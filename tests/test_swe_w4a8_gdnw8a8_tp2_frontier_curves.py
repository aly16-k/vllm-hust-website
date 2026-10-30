import json
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 五档齐全, C16 是 max-num-seqs=16 的饱和档, 不能被 C8 顶替。
LEVELS = (1, 2, 4, 8, 16)
CHIPS = 2


def load_renderer():
    return runpy.run_path(str(ROOT / "scripts/render_swe_w4a8_gdnw8a8_tp2_curves.py"))


def frontier():
    return json.loads((ROOT / "data/leaderboard_frontier.json").read_text())


def swe_evidence():
    return json.loads(
        (ROOT / "data/leaderboard_frontier_swe_evidence.json").read_text()
    )


def tp2_points(data, renderer):
    return [
        p
        for p in data["points"]
        if p["cohort_id"] == renderer["COHORT"]
        and p["configuration"]["parameters"].get("max_num_seqs") == 16
        and p["load"].get("concurrency_series") == renderer["SERIES"]
        and p["configuration"]["hardware"]["accelerator_count"] == CHIPS
    ]


def test_published_w4a8_gdn_tp2_swe_curve_matches_the_measured_series():
    renderer = load_renderer()
    data = frontier()
    rendered = renderer["render"](data)
    assert (
        rendered
        == (
            ROOT / "assets/frontier-qwen35-w4a8-gdnw8a8-tp2-swe-concurrency.svg"
        ).read_text()
    )
    ids = {
        node.attrib["data-point"]
        for node in ET.fromstring(rendered).iter()
        if "data-point" in node.attrib
    }
    assert ids == {point["id"] for point in tp2_points(data, renderer)}
    # 本 cohort 只有一条 real-MTP2 线, 五个点必须全部被画出来。
    assert len(ids) == len(LEVELS)


def test_w4a8_gdn_tp2_curve_declares_its_svg_on_the_cohort_contract():
    renderer = load_renderer()
    cohort = next(c for c in frontier()["cohorts"] if c["id"] == renderer["COHORT"])
    url = cohort["workload"]["contract"]["concurrency_curves_url"]
    assert url.startswith(
        "./assets/frontier-qwen35-w4a8-gdnw8a8-tp2-swe-concurrency.svg?v="
    )
    assert (ROOT / url.split("?")[0].lstrip("./")).is_file()


def test_w4a8_gdn_tp2_cohort_does_not_collide_with_any_other_cohort():
    """cohort 唯一性 key 是 [model, precision, workload, context]

    (assets/leaderboard-frontier-model.js)。本 cohort 的 workload.id 与其它 TP2
    cohort 相同 (同一份 prepared workload 的双卡口径), 靠自有的 precision.id
    重开这个键, 否则站点 validate() 抛 duplicate。

    这里只对本 cohort 自己作断言 —— 不依赖任何别的 cohort 在不在基线里。
    """
    renderer = load_renderer()
    cohorts = frontier()["cohorts"]
    mine = next(c for c in cohorts if c["id"] == renderer["COHORT"])

    def key(c):
        return (
            c["model"]["id"],
            c["precision"]["id"],
            c["workload"]["id"],
            c["context_tokens"],
        )

    assert sum(1 for c in cohorts if key(c) == key(mine)) == 1
    # 同 model 下不得有第二条 cohort 复用我们的 precision.id。
    same_model = [c for c in cohorts if c["model"]["id"] == mine["model"]["id"]]
    assert (
        sum(1 for c in same_model if c["precision"]["id"] == mine["precision"]["id"])
        == 1
    )


def test_w4a8_gdn_tp2_swe_curve_covers_c1_through_c16_on_one_series():
    """五档必须在同一 cohort + 同一 concurrency_series 内, 且只由客户端 C 区分。"""
    renderer = load_renderer()
    points = tp2_points(frontier(), renderer)
    assert sorted(p["load"]["concurrency"] for p in points) == list(LEVELS)
    assert {p["load"]["concurrency_series"] for p in points} == {renderer["SERIES"]}
    assert {p["cohort_id"] for p in points} == {renderer["COHORT"]}
    assert {p["load"]["session_rotation_depth"] for p in points} == {1}
    assert {json.dumps(p["configuration"]["mods"]) for p in points} == {
        json.dumps(renderer["MODS"])
    }
    for point in points:
        params = point["configuration"]["parameters"]
        assert params["mtp_draft_tokens"] == 2
        assert params["max_num_seqs"] == 16
        assert params["quantization"] == "ascend"
        # 本曲线是双卡: 并行规模、卡数与卡位都要落在 TP2 上。
        assert params["tensor_parallel_size"] == CHIPS
        assert point["configuration"]["hardware"]["accelerator_count"] == CHIPS
        # 本次测量固定在 6,7 上完成 (卡 0 为常驻服务, 1/2/3/5 有别的作业, 4 被新起的
        # TP2 作业占走)。
        assert params["physical_devices"] == [6, 7]
        # 专家层 int4 与 GDN 五投影 int8 是本 cohort 的 precision 主张, 必须逐点写明。
        assert "int4" in params["weight_precision"]
        assert "in_proj_a" in params["weight_precision"]
        for key, expected in renderer["CALIBER"].items():
            assert params[key] == expected, (key, params[key], expected)
        assert params["qualification"]["concurrency"] == point["load"]["concurrency"]
        assert params["qualification"]["measurement_seconds"] == 60


def test_w4a8_gdn_tp2_kv_budget_is_declared_per_point():
    """本 cohort 的服务口径与已发布 W8A8 TP2 点逐字一致: 显式 KV 预算逐点自述。

    kv_cache_memory_bytes 是 deepcopy 造新 cohort 时最容易带错的一处 —— 值必须与
    渲染器声明的 CALIBER 逐点一致, 且每个点都要在 caliber_note 里自述这个值。
    """
    renderer = load_renderer()
    points = tp2_points(frontier(), renderer)
    declared = renderer["CALIBER"]["kv_cache_memory_bytes"]
    for point in points:
        params = point["configuration"]["parameters"]
        assert params["kv_cache_memory_bytes"] == declared
        assert str(declared) in params["caliber_note"]
        assert "24.25" in params["caliber_note"]


def test_w4a8_gdn_tp2_c16_point_is_its_own_saturated_900s_window():
    """C16 档专项: 一个独立 run_id 的真实满载窗口, 不是 C8 的复制或标签。"""
    renderer = load_renderer()
    points = {p["load"]["concurrency"]: p for p in tp2_points(frontier(), renderer)}
    assert set(points) == set(LEVELS)
    c16, c8 = points[16], points[8]

    runs = {r["run_id"]: r for r in swe_evidence()["runs"]}
    run = runs[c16["evidence"]["run_ids"][0]]

    assert run["client"]["concurrency"] == 16
    assert run["client"]["chips"] == CHIPS
    assert run["summary"]["valid"] is True
    assert run["summary"]["aborted"] is False
    assert run["summary"]["failed_requests"] == 0
    assert run["summary"]["measurement_seconds"] == 900
    assert run["summary"]["planned_measurement_seconds"] == 900
    assert run["summary"]["decode_speed_samples"] > 0
    assert run["summary"]["full_concurrency_fraction"] > 0.9
    assert run["summary"]["mean_client_inflight"] > 15

    qualification = run["qualification_run"]
    assert qualification["concurrency"] == 16
    assert qualification["duration"] == 60
    assert qualification["summary"]["failed_requests"] == 0
    assert run["validation"]["protocol_qualification_passed"] is True
    assert run["validation"]["protocol_qualification_run_id"] == qualification["run_id"]

    assert c16["evidence"]["run_ids"] != c8["evidence"]["run_ids"]
    assert c16["metrics"] != c8["metrics"]
    assert c16["metrics"] == run["metrics"]
    assert c16["metrics"]["output_tps"] == (
        run["summary"]["observed_output_tokens_in_window"] / 900
    )

    # 五档各有一个独立 run, 且都与点内联的 metrics 一致。
    seen = set()
    for level, point in points.items():
        assert len(point["evidence"]["run_ids"]) == 1
        run_id = point["evidence"]["run_ids"][0]
        assert run_id not in seen
        seen.add(run_id)
        row = runs[run_id]
        assert row["client"]["concurrency"] == level
        assert row["client"]["chips"] == CHIPS
        assert row["summary"]["measurement_seconds"] == 900
        assert row["summary"]["failed_requests"] == 0
        assert row["metrics"] == point["metrics"]


def test_w4a8_gdn_tp2_runtime_evidence_is_this_cohorts_own():
    """运行期证据必须来自本 cohort 自己的采样, 不能是造 cohort 时继承来的。

    runtime_evidence 是深拷贝母本时最隐蔽的一处: counters 与 sources 都会
    原样继承, 从而把别的点的计数器与日志路径当成这里的证据提交。
    """
    renderer = load_renderer()
    points = tp2_points(frontier(), renderer)
    runs = {
        r["run_id"]: r
        for r in swe_evidence()["runs"]
        if r.get("point_id", "").startswith("qwen35-sweprefix-w4a8-gdnw8a8-ascend-tp2")
    }
    assert len(runs) == len(LEVELS)
    for point in points:
        run = runs[point["evidence"]["run_ids"][0]]
        evidence = run["runtime_evidence"]
        sources = evidence["sources"]
        # 三条来源路径都必须指向本批次自己的采样文件 (编排与采样日志是 w4a8gdn,
        # 服务端日志是 gdn_tp2_server_20261030T140254Z), 且不得残留别的批次的时间戳。
        assert set(sources) == {"orchestration_log", "metrics_tsv", "server_log"}
        assert all("/data/whk/models/logs/" in path for path in sources.values())
        assert "w4a8gdn" in sources["orchestration_log"]
        assert "w4a8gdn" in sources["metrics_tsv"]
        assert "20261030T140254Z" in sources["server_log"]
        assert all("20261023" not in path for path in sources.values())
        counters = evidence["prefix_cache_counters"]
        assert counters["prefix_cache_queries_total"]
        assert counters["prefix_cache_hits_total"]
        assert evidence["prefix_cache_observed"] is True
        assert evidence["spec_decode_metrics_seen"] is True
        assert evidence["campaign_finished_ok"] is True
