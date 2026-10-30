# Qwen3.5-35B-A3B W4A8 (专家层 int4 + GDN 五投影 W8A8) · 双卡 (TP2) SWE prefix reuse 并发曲线 (C1 / C2 / C4 / C8 / C16)

本报告由实测产物直接生成, 数值零推断。每一档是一个独立的 900s 固定窗口观测, 五档在**同一个服务实例**内按 C1→C16 连续完成 (档间不重启服务、不重置 prefix cache),
因此档间唯一变量就是客户端并发 C。

**本页对应 series `swe-w4a8-gdnw8a8-tp2-20261030-mtp2` (TP2, 2×910B3)**。与已发布的 W8A8 TP2 曲线
(`swe-w8a8-tp2-20260929-mtp2-r2`) 和 W8A8-GDN TP2 曲线 (`swe-w8a8-gdn-tp2-20261016-mtp2`) 使用同一份
prepared workload、同一个客户端、同一协议、 **逐字相同的显式 KV 预算** (26038239232 = 24.25 GiB/卡), 差别是量化配方与所用卡对。

## 0. 与已发布两条 TP2 曲线的配方差异

| 模块                                                           | 已发布 W8A8 TP2              | W8A8-GDN TP2 | 本 cohort                                   |
| -------------------------------------------------------------- | ---------------------------- | ------------ | ------------------------------------------- |
| `*self_attn*` (除 o_proj)                                      | W8A8                         | 同左         | 同左                                        |
| `*mlp.experts*`                                                | W8A8 (int8 权重 + int8 激活) | 同左         | **int4 权重 (ssz per-channel) + int8 激活** |
| GDN `linear_attn.in_proj_qkv` / `in_proj_z` / `out_proj`       | BF16                         | W8A16        | **W8A8**                                    |
| GDN `linear_attn.in_proj_a` / `in_proj_b`                      | BF16                         | BF16         | **W8A8**                                    |
| 其余 (o_proj / shared_expert / router / embed / lm_head / MTP) | BF16                         | 同左         | 同左                                        |
| KV cache                                                       | bf16                         | 同左         | 同左                                        |

两对融合层必须同精度: `in_proj_qkv` 与 `in_proj_z` 在加载时融合成 `in_proj_qkvz`, `in_proj_a` 与 `in_proj_b` 融合成
`in_proj_ba` (`qwen3_5.py` 的 `packed_modules_mapping`), 而
`modelslim_config.is_layer_skipped_ascend()` 会在同一融合层分片精度不一致时直接抛错拒载。 本产物成对同精度,
`quant_model_description.json` 实测 W4A8_DYNAMIC 122880 / W8A8_DYNAMIC 541 / FLOAT 1551 项, **零个 W8A16
条目**。

## 1. 裁定口径

| 项           | 取值                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 硬件         | Ascend 910B3 × 2 (physical devices [6, 7], 同一 CPU 亲和组、HCCS 全互联)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 引擎         | vLLM + vLLM-Ascend `0.28.1.post1.dev143+gf18cf803c / 0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 模型         | Qwen3.5-35B-A3B @ `712cf74392b05026a6db2bf213d343747d1f6d45`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 精度         | **W4A8**: MLP 专家层 (gate/up/down_proj) int4 权重 (ssz per-channel) + int8 激活; 10 层 `self_attn` 的 q/k/v 与 30 层 GDN `linear_attn` 的 `in_proj_qkv` / `in_proj_z` / `in_proj_a` / `in_proj_b` / `out_proj` 为 int8 权重 + int8 激活 (动态 per-token, minmax); `lm_head` 保持 FLOAT; msModelSlim ascendv1 离线量化, 6 个 `quant_model_weights` 分片; KV auto (bf16; KV 未量化)                                                                                                                                                                                                                                                                                                                                     |
| 权重产物指纹 | `quant_model_description.json` sha256 `2b3f6cbb166a8cf5…`; `quant_model_weights.safetensors.index.json` sha256 `a2039f38cf7db77c…`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 并行         | TP2 · PP1 · DP1 · EP=False                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 并发座位     | max-num-seqs 16                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| KV 预算      | **`26038239232` (24.25 GiB/卡, 显式) — 262144 上下文下最大并发 7.91x** (服务端实测 GPU KV cache size 2,072,810 token)。与已发布 TP2 点逐字相同; 本产物权重只有 23 GiB, 仍**不放大** KV, 以保持同口径                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 投机解码     | real MTP, num_speculative_tokens=2 (真实接受/拒绝, 无 synthetic 覆盖); 服务端日志实测 Mean acceptance length 2.869–2.911 (每档 n=89–90 采样)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| MOD          | `ascend-mtp-contract-2patch` + `ascend-npugraph-ex-attr-shim` — 三处 engine 本地最小改动, 与同机的 W8A8 / W8A8-GDN TP2 点完全相同: (1) `vllm_ascend/platform.py` NPUPlatform.check_runner_kv_caches_multi_layer() no-op; (2) `vllm_ascend/worker/model_runner_v1.py:898` 改传 `self._get_mamba_state_copy_funcs()` (dict, 0.28.1 契约); (3) `vllm_ascend/compilation/compiler_interface.py` 的 npugraph_ex 属性 stand-in —— 其生效条件是编译图中存在 `npu.npu_weight_quant_batchmatmul` (W8A16) 节点, 而本产物零个 W8A16 条目, 故对模型**惰性**。MTP2 为真实接受/拒绝, 无 synthetic sampler 覆盖。另: engine 树内 `vllm_ascend/plugins/a8w4_compact` 默认关闭, 本轮未设置 `VLLM_ASCEND_A8W4_COMPACT`, 对本曲线为 no-op |
| Graph        | FULL_DECODE_ONLY (请求 FULL, 平台因 AscendGDNAttentionBackend 仅支持 UNIFORM_BATCH 而降级); max_cudagraph_capture_size=256                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| runtime 提交 | vllm `f18cf803c5f6` / vllm-ascend `74f0c0a27237`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| runtime 版本 | vllm 0.28.1.post1.dev143+gf18cf803c; vllm-ascend 0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27; torch 2.13.0+cpu; torch_npu 2.13.0.rc1; Ascend CANN 9.1.0 (逐进程临时环境: /data/jxd/Ascend-9.1.0/{ascend-toolkit,nnal/atb}; 系统自带 /usr/local/Ascend CANN 8.5.0 未改动、未覆盖), ATB CXX ABI=1                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 客户端       | swe-prefix-reuse `6861242dbd9f17b707003191e4200b7752911d7c` (0.1.2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| workload     | `https://github.com/vLLM-HUST/swe-prefix-reuse` prepared/qwen35.json sha256 `ac1fbf9885076c369fd25c5525d1163ff8b7945972cdf4c0c6d2d523ab7c1172`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 窗口         | 每档 60s 协议/缓存资格 + 900s 正式窗口; 总吞吐只计窗口内收到的实际输出 token                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 坐标轴       | x = P90 across fully completed in-window requests of (N-1)/(last-token-time-first-token-time); grouped SSE token times, not inverse P90 TPOT; y = In-window generated tokens / 900 seconds / every allocated chip                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

## 2. 结果

| C   | run_id         | output tps | tps/卡   | P90 decode tps | TTFT p95 (s) | 窗口内完成 / 启动 | 在途均值 | 满载占比 | 最大 prompt | drain (s) | MTP 接受长度 (n) |
| --- | -------------- | ---------- | -------- | -------------- | ------------ | ----------------- | -------- | -------- | ----------- | --------- | ---------------- |
| C1  | `ec4a80ba9aa3` | 84.2622    | 42.1311  | 94.7066        | 0.863        | 142 / 143         | 0.9995   | 0.9995   | 52505       | 23.17     | 2.902 (90)       |
| C2  | `9a84f5872364` | 129.6222   | 64.8111  | 80.5949        | 1.420        | 179 / 181         | 1.9992   | 0.9992   | 52505       | 13.52     | 2.884 (89)       |
| C4  | `c9ac150d2669` | 204.3578   | 102.1789 | 69.8606        | 1.775        | 292 / 296         | 3.9984   | 0.9984   | 67634       | 14.38     | 2.911 (90)       |
| C8  | `4b7108f6d360` | 303.7989   | 151.8994 | 59.7799        | 2.005        | 445 / 453         | 7.9976   | 0.9977   | 78752       | 48.80     | 2.887 (90)       |
| C16 | `31a8793fbb00` | 402.4000   | 201.2000 | 38.4014        | 2.989        | 607 / 623         | 15.9974  | 0.9975   | 59957       | 71.66     | 2.869 (90)       |

末列是服务端日志里 `metrics.py` 每 ~10s 打点的 `SpecDecoding metrics: Mean acceptance length` 在本档 900s 正式窗内的均值 (n
= 采样条数), 用以证明 real MTP 真的在跑。

吞吐单调升到 C16, C16 就是本条曲线的峰值 (也是 max-num-seqs=16 的饱和档)。显式 KV 预算让 SWE 前缀在 C16 仍然留在缓存里 (满载占比 0.9975、 P90
decode 38.40 tokens/s 仍在秒级排队区), 因此这里没有出现高并发回落 —— 这是本口径下的观测, 不是对其它 KV 预算的断言。

## 3. 每档工作量与资格窗口

| C   | 完成会话 | 覆盖轨迹 | 最大 turn | 60s 资格窗口 run | 资格失败请求 | 正式窗口失败请求 | 正式窗口有效 |
| --- | -------- | -------- | --------- | ---------------- | ------------ | ---------------- | ------------ |
| C1  | 5        | 6 / 8    | 38        | `5c67832fad92`   | 0            | 0                | 是           |
| C2  | 5        | 7 / 8    | 38        | `b5d426ef0cc0`   | 0            | 0                | 是           |
| C4  | 7        | 8 / 8    | 45        | `8cbecfaa3093`   | 0            | 0                | 是           |
| C8  | 9        | 8 / 8    | 59        | `eb2886b6186e`   | 0            | 0                | 是           |
| C16 | 11       | 8 / 8    | 41        | `1cebfd5aa208`   | 0            | 0                | 是           |

## 4. 曲线与站点数据

cohort `qwen35-35b-a3b-w4a8-gdnw8a8-tp2-2x910b3-full-decode-only-sweprefix-900s-v1`; precision
`w4a8-expert-int4-linearattn-5proj-w8a8-weights-compute-bf16-kv`; workload
`sweprefix-qwen35-eight-traces-900s-v1` (与已发布 W8A8 TP2 点相同, 靠自有 precision 重开 cohort 唯一性 key); 渲染器
`scripts/render_swe_w4a8_gdnw8a8_tp2_curves.py`; SVG
`assets/frontier-qwen35-w4a8-gdnw8a8-tp2-swe-concurrency.svg`; 测试
`tests/test_swe_w4a8_gdnw8a8_tp2_frontier_curves.py`。

数据本身在 benchmark 仓库的 `website-publication/data/` 提交, 站点只按 pin 的 benchmark revision 投影, 不在本站 checkout
里携带数据。

## 5. 口径限制 (必须与数值一起引用)

- 客户端与 engine 同机 (127.0.0.1), 客户端调度/JSON/网络开销仍在观测路径内
- 未做预热: 每个 play 使用新 cache_salt 冷启动, 窗口结束排空在途请求
- 本轮为 60s 资格窗口 + 900s 固定窗口两个显式测量, 无自动认证语义
- 首次冷编译: 因关闭 torch AOT autograd 缓存, 启动要重跑 torch.compile + ACL 图捕获, 该时间不计入测量窗口
- CANN 9.1 为逐进程临时环境 (未覆盖系统 8.5.0); 引擎实际加载的 CANN 版本见启动日志的 ASCEND_HOME_PATH
- 样例 workload 仅 8 条轨迹/360 轮, 非统计代表性 SWE 样本; 最长会话 141269 tokens, 未触及 262144 上限
- MTP 特有: 请求的 cudagraph_mode=FULL 被平台降级为 FULL_DECODE_ONLY (AscendGDNAttentionBackend 仅声明
  UNIFORM_BATCH)
- MTP 特有: 每次带 MTP 的运行都会留下一条不可用的 AOT 编译缓存条目, 故每轮实验前需隔离 `torch_aot_compile/*`
- MTP 特有: 前缀复用可用, 但受 EAGLE/MTP 尾块丢弃语义影响, 每个请求比无 MTP 时少复用 1 个 block (2048 token)
- 三个 engine 补丁均为本地最小改动, 未进入上游; 第三处 (npugraph_ex 属性垫片) 对本产物为惰性 (零个 W8A16 条目)
- 配方特有: 专家 int4 + GDN 五投影 int8。与已发布 W8A8 TP2 点、W8A8-GDN TP2 点相比各有两个以上变量同时变化 (专家位宽 / GDN 精度 / 卡对),
  因此本页只报自己的观测, 不声明跨配方的加速比
- 卡位特有: 本轮用 NPU 6,7; 已发布 W8A8 TP2 点用 0,1, W8A8-GDN TP2 点用 2,3 (本机 NPU0 为常驻服务, 1/2/3/5
  有其它作业)。未做同卡对重复实验, 跨 cohort 的差值不能唯一归因
- 同一服务实例内串行测 5 档, 各档不是统计独立重复; 不做重复性估计

服务启动命令 (五档同一实例):

```
/data/jxd/envs/vllm-hust-v1-cann91/bin/vllm serve /data/whk/models/Qwen3.5-35B-A3B-W4A8-GDN-W8A8 \
  --served-model-name qwen35-a3b-w4a8gdn --tensor-parallel-size 2 --max-model-len 262144 \
  --gpu-memory-utilization 0.95 --trust-remote-code --host 127.0.0.1 --port 8302 \
  --quantization ascend --max-num-seqs 16 --kv-cache-memory-bytes 26038239232 \
  --max-num-batched-tokens 4096 --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --compilation-config '{"max_cudagraph_capture_size":256,"cudagraph_mode":"FULL"}'
```

## 6. 与同 workload 已发布曲线的并排读数

| C   | 本 cohort 总 throughput | 本 cohort tps/卡 | W8A8 TP2 tps/卡 | 差值    | W8A8-GDN TP2 tps/卡 | 差值   |
| --- | ----------------------- | ---------------- | --------------- | ------- | ------------------- | ------ |
| C1  | 84.2622                 | 42.1311          | 45.7878         | -7.99%  | 42.7028             | -1.34% |
| C2  | 129.6222                | 64.8111          | 72.0911         | -10.10% | 64.4772             | +0.52% |
| C4  | 204.3578                | 102.1789         | 109.5400        | -6.72%  | 97.5200             | +4.78% |
| C8  | 303.7989                | 151.8994         | 167.9200        | -9.54%  | 149.6500            | +1.50% |
| C16 | 402.4000                | 201.2000         | 220.4344        | -8.73%  | 198.8250            | +1.19% |

**差值列不是配方效应。** 三条线在同一 workload、同一协议、同一服务 KV 口径下测得, 但彼此之间同时有多个变量: 专家位宽 (int8 vs int4)、 GDN 投影精度 (BF16
/ W8A16 / W8A8)、以及卡对 (0,1 / 2,3 / 6,7)。没有同卡对的配对重复实验, 因此本表只描述**本次观测到的位置**, 不构成跨配方的通用加速比声明。要分离配方效应,
需在同一卡对上把两个配方各跑一遍。

## 7. 精度代价 (不在 Frontier 队列内, 仅记录)

同一 wikitext-2 协议 (`ppl_wikitext2.py`, chunk-tokens 1000, 并发 8, 同一 session 内 W8A8 基线) 实测: W8A8
**7.1764** / 本产物 **7.4029** 与 **7.4052** (两次, 极差 0.03%), 即专家 int4 + GDN 五投影 int8 相对 W8A8 多付
**+3.16%** PPL。 本产物的配方与 2026-10-23 单卡 W4A8-GDN 产物不同 (后者未量化 `in_proj_a` / `in_proj_b`), 两者的 PPL
不可直接比较。
