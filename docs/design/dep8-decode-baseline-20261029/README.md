# DEP8 Decode 自然输出主测试结果（2026-10-29）

本轮完成 9 档客户端并发，每档 90 秒，全部完成请求自然 stop，未使用 ignore_eos，未出现请求错误。每档截止即取消在途请求，不等待排空。所有压测客户端已退出，指标采集已停止，模型服务保留运行。

## 配置与统计口径

TY19 TP1/DP8/EP8，DSpark5，现有图模式；每 rank max_num_seqs=132，总运行序列上限1056。TY18四DEP2提供Prefill。约128k输入、四种固定合成长回答任务，8192输出上限，完成输出平均约1997–2077 token。实测缓存命中约99.906%，不是之前90%命中的整链路测试。

总输出 TPS 使用八个 Decode rank 的生成计数增量，窗口为共同启动后30秒至最早截止点内的指标样本，约56–58秒。单请求 TPS 使用全部成功完成请求的输出 token 数除以首次至末次非空 SSE 的时间；TTFT P95 包含已观察到首输出的取消请求。两个口径窗口和样本不同，不能用并发乘单请求均值精确还原总吞吐。

## 并发结果

| 客户端并发 | D running均值 | 总输出token/s | 单请求TPS均值 | P5 | P50 | P95 | TTFT P95(s) | 完成/取消 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 15.7 | 2,316 | 139.20 | 108.52 | 125.47 | 203.70 | 0.287 | 88/16 |
| 64 | 63.2 | 9,072 | 139.79 | 107.45 | 127.08 | 205.85 | 0.438 | 361/64 |
| 128 | 126.0 | 17,889 | 136.98 | 105.20 | 123.14 | 204.17 | 0.573 | 718/128 |
| 256 | 252.4 | 35,305 | 133.82 | 102.39 | 120.86 | 199.22 | 0.850 | 1396/256 |
| 384 | 375.8 | 48,083 | 122.97 | 93.65 | 111.42 | 183.73 | 1.071 | 1893/384 |
| 512 | 504.9 | 55,807 | 106.69 | 81.68 | 97.03 | 158.34 | 1.290 | 2158/512 |
| 768 | 756.9 | 67,387 | 86.35 | 66.28 | 78.54 | 128.87 | 1.843 | 2540/768 |
| 1024 | 1011.5 | 72,061 | 69.65 | 53.24 | 63.84 | 105.05 | 2.333 | 2629/1024 |
| 1280 | 1053.9 | 101,162 | 90.51 | 69.45 | 81.55 | 135.59 | 7.335 | 3705/1280 |

## 每卡输出吞吐

单位为输出 token/s。平均每卡=整机吞吐÷8；R0–R7为原始指标的DP engine rank，TP1下一rank对应一卡，编号不额外声称已核对物理GPU UUID。EP8各卡协同计算，下表按输出所属rank记账。

| 并发 | 总输出TPS | 平均每卡 | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2,316 | 290 | 281 | 259 | 269 | 337 | 253 | 329 | 306 | 281 |
| 64 | 9,072 | 1,134 | 1,139 | 1,050 | 1,173 | 1,132 | 1,116 | 1,116 | 1,217 | 1,129 |
| 128 | 17,889 | 2,236 | 2,206 | 2,246 | 2,262 | 2,274 | 2,260 | 2,274 | 2,218 | 2,149 |
| 256 | 35,305 | 4,413 | 4,296 | 4,445 | 4,489 | 4,309 | 4,482 | 4,350 | 4,510 | 4,424 |
| 384 | 48,083 | 6,010 | 5,901 | 5,772 | 5,841 | 6,328 | 6,213 | 6,028 | 6,028 | 5,972 |
| 512 | 55,807 | 6,976 | 6,899 | 7,091 | 6,798 | 6,871 | 6,764 | 7,339 | 7,186 | 6,859 |
| 768 | 67,387 | 8,423 | 8,503 | 8,175 | 8,737 | 8,558 | 8,568 | 8,218 | 8,310 | 8,318 |
| 1,024 | 72,061 | 9,008 | 8,892 | 8,947 | 8,803 | 8,992 | 9,255 | 9,038 | 9,254 | 8,881 |
| 1,280 | 101,162 | 12,645 | 12,717 | 12,570 | 12,713 | 12,992 | 12,339 | 12,653 | 12,350 | 12,829 |

## 结果解读与边界

16–256并发时单请求均值基本维持134–140 TPS；384开始下降。1024并发达到72,061 token/s，单请求均值69.65 TPS。1280并发达到本轮最高101,162 token/s，单请求均值90.51 TPS，但TTFT P95达到7.335秒。

1280档不能解释为普通的平滑扩展：与1024相比，D running只从1011.5增至1053.9，投机token接受率却从25.65%升至44.57%，每轮接受token数加1从2.282升至3.228。两档均每次draft 5 token；各任务单请求TPS均有提升，输出长度接近。吞吐跳升与接受率跳升同时发生，但根因未确认。本轮未做profile或修改代码，不能据此断言某个kernel、通信或调度逻辑是根因。

1280档D running P95=1056，平均waiting=217.7，P waiting均值=0，说明已触及当前配置的运行序列上限，压力主要积累在D。1024档D waiting均值7.1。仅一个饱和档不足以证明绝对硬件吞吐上限或稳定平台；本轮峰值是该固定负载、配置下的观测值。

每档只有一次90秒观测；高并发取消数量较多，完成样本可能偏向短/快请求。取消请求不计作错误、不计算完成TPS。详细JSON同时保留完全落在测量窗口内的请求分布。四种固定任务和高缓存命中不代表线上混合负载，投机接受率尤其依赖文本与执行条件。

## 启动配置

本轮复用现有模型服务，没有重启或调整服务参数。以下服务入口参数在测试后从原容器读取；完整复现还需要对应 release 的源码挂载、模型、环境变量、sidecar 与路由配置。

- 模型：`DeepSeek-V4.1-Flash-1017`，容器内 `/model`。
- Decode：TY19 `tuoyu-cep8-dev-decode`，release `/data/services/tuoyu-cep8-dev-20261028/service-r22`。
- Prefill：TY18 `tuoyu-shared-l2-perf-prefill-{0..3}`，四个 TP1/DP2/EP2 实例，release `/data/services/tuoyu-shared-l2-perf-20261029/service-r1`，Shared L2 优化提交 `f15d7e3ef2`。
- Decode 镜像 ID：`sha256:ad1bf5e74168f03f841d701f2b4e1fe183e9341b4389f67db6ee803c3ade61eb`，基础镜像标签 `vllm/vllm-openai:nightly-fe4df3df0eb703862c686b256e9fe44fa94c8945`，实际代码包含 service-r22 挂载覆盖。
- 请求从 TYCPU Gateway `http://127.0.0.1:51003` 的隔离渠道16进入；Dynamo frontend `http://172.2.242.19:38080`，namespace `tuoyu_cep8_dev_20261028`。

### Decode 完整入口参数

```sh
python3 /opt/tuoyu/no-thp-exec.py vllm-rs serve /model \
  --kv-transfer-config '{"kv_connector": "NixlConnector", "kv_connector_module_path": "vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_as_nixl", "kv_role": "kv_consumer", "kv_load_failure_policy": "fail", "kv_buffer_device": "cuda", "kv_connector_extra_config": {"backends": ["UCX"], "enforce_handshake_compat": false, "kv_lease_duration": 120, "enable_push_cancellation": true}}' \
  --served-model-name deepseek-v4.1-flash \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser None \
  --kv-cache-dtype fp8 \
  --max-model-len 1048576 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --no-enable-cumem-allocator \
  --enable-ep-weight-filter \
  --cpu-distributed-timeout-seconds 7200 \
  --attention-config '{"backend":"FLASHMLA_MEGA_ATTN_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --engram-config '{"cpu_offload":true,"embedding_across_dp":true,"use_thp":false}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}' \
  --prefill-schedule-interval 2 \
  --gpu-memory-utilization 0.95 \
  --kernel-config '{"enable_flashinfer_autotune":true,"moe_backend":"deep_gemm_mega_moe","sparse_indexer_topk_backend":"deep_select"}' \
  --data-parallel-size 8 \
  --enable-eplb \
  --eplb-config '{"communicator":"symm_mem","num_redundant_experts":16,"window_size":1000,"step_interval":3000,"use_async":true,"log_balancedness":true,"log_balancedness_interval":3000}' \
  --no-enable-prefix-caching \
  --numa-bind \
  --numa-bind-nodes 0 0 0 0 1 1 1 1 \
  --max-num-seqs 132 \
  --max-num-batched-tokens 1320 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":792,"cudagraph_capture_sizes":[6,12,18,24,30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144,150,156,162,168,174,180,186,192,198,204,210,216,222,228,234,240,246,252,258,264,270,276,282,288,294,300,306,312,318,324,330,336,342,348,354,360,366,372,378,384,390,396,402,408,414,420,426,432,438,444,450,456,462,468,474,480,504,528,552,576,600,624,648,672,696,720,744,768,792]}' \
  --limit-mm-per-prompt '{"image": 600, "video": 0}' \
  --data-parallel-size-local 8 \
  --data-parallel-address 172.2.242.19 \
  --data-parallel-rpc-port 24345 \
  --host 127.0.0.1 \
  --port 28000 \
  --grpc-port 29000
```

### Prefill 实例0完整入口参数

```sh
python3 /opt/tuoyu/no-thp-exec.py vllm-rs serve /model \
  --kv-transfer-config '{"kv_connector": "MultiConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"connectors": [{"kv_connector": "NixlConnector", "kv_connector_module_path": "vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_as_nixl", "kv_role": "kv_both", "kv_load_failure_policy": "fail", "kv_buffer_device": "cuda", "kv_connector_extra_config": {"backends": ["UCX"], "enforce_handshake_compat": false, "kv_lease_duration": 120, "enable_push_cancellation": true}}, {"kv_connector": "SharedCPUOffloadConnector", "kv_connector_module_path": "vllm.distributed.kv_transfer.kv_connector.v1.shared_cpu_offload_connector", "kv_role": "kv_both", "kv_connector_extra_config": {"pool_bytes": 274877906944, "manager_socket": "/run/tuoyu-l2/perf-r1.sock", "cache_namespace": "DeepSeek-V4.1-Flash-1017-layout-v1", "max_store_bytes_per_step": 134217728, "store_bytes_per_second_per_rank": 2147483648}}]}}' \
  --served-model-name deepseek-v4.1-flash \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser None \
  --kv-cache-dtype fp8 \
  --max-model-len 1048576 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --no-enable-cumem-allocator \
  --enable-ep-weight-filter \
  --cpu-distributed-timeout-seconds 7200 \
  --attention-config '{"backend":"FLASHMLA_MEGA_ATTN_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --engram-config '{"cpu_offload":true,"embedding_across_dp":true,"use_thp":false}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}' \
  --prefill-schedule-interval 2 \
  --gpu-memory-utilization 0.95 \
  --kernel-config '{"enable_flashinfer_autotune":true,"moe_backend":"deep_gemm_mega_moe","sparse_indexer_topk_backend":"deep_select"}' \
  --data-parallel-size 2 \
  --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6,12,18,24,30,36,48,60,72,96,120,144,192,240,288,384,480,576,768,1020,1536,2046],"decoder_replay_cudagraph_capture_sizes":[6,12,18,24,30,36,48,60,72,96,120,144,192,240,288,384,480,576,768,1020,1536,2048],"max_cudagraph_capture_size":2046}' \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --long-prefill-token-threshold 8192 \
  --prefix-cache-retention-interval 32768 \
  --numa-bind \
  --limit-mm-per-prompt '{"image": 600, "video": 0}' \
  --data-parallel-size-local 2 \
  --data-parallel-address 172.2.242.18 \
  --data-parallel-rpc-port 14345 \
  --host 127.0.0.1 \
  --port 18000 \
  --grpc-port 19000 \
  --numa-bind-nodes 0 0 \
  --enable-prefix-caching \
  --kv-events-config '{"publisher": "zmq", "topic": "kv-events", "enable_kv_cache_events": true, "endpoint": "tcp://172.2.242.18:21000"}'
```

### 压测客户端

两个 Docker 客户端，每个使用 3 CPU、4 GiB 内存、host 网络、128 PID 上限。`CONCURRENCY` 是整机目标值，脚本每个 shard 创建其一半的闭环请求槽。每槽请求完成后立即补发。四种任务轮转：中文分布式推理说明、Python队列实现与测试、虚构故障报告、数据平台容量规划。

```sh
# 在 TYCPU 执行；示例为 1024 并发，目录需预先准备 client.py/deps/prompts.json/key.json。
# key.json 仅保留在远端，不写入本文档。
BENCH_ROOT=/opt/tai-gateway/validation/dep8-decode-baseline-20261029
for SHARD in 0 1; do
  docker run -d --name "tuoyu-dep8-baseline-c1024-s${SHARD}" \
    --network host --cpus 3 --memory 4g --pids-limit 128 \
    --entrypoint python3 -v "${BENCH_ROOT}:/bench" \
    -e PYTHONPATH=/bench/deps -e PYTHONUNBUFFERED=1 \
    -e SHARD="${SHARD}" -e CONCURRENCY=1024 -e STAGE=c1024 -e DURATION=90 \
    tuoyu/vllm-bench:cpu-light /bench/client.py
done
```

请求配置：

```json
{
  "model": "deepseek-v4.1-flash",
  "messages": [
    {
      "role": "user",
      "content": "<预热后的固定长输入及任务>"
    }
  ],
  "temperature": 0,
  "max_tokens": 8192,
  "enable_thinking": false,
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}
```

先执行32次预热，覆盖4个P实例×2个rank×4个任务（预热输出上限16，不计入正式测试）。然后依次执行16、64、128、256、384、512、768、1024，每档90秒；追加1280时不重复预热。每档截止立即取消在途请求；客户端超过100秒仍未退出则kill并终止扫档。Prometheus每2秒、GPU约每5秒采样。

## 数据留存与验证

共 19,920 条记录，完成 15,488，截止取消 4,432，错误0。已校验18个shard的请求计数守恒、ID唯一、完成原因均为stop、八个rank吞吐为正。测试结束后客户端和采集器退出，模型服务保留运行。

本提交仅包含本文档。原始请求、完整统计、CSV、脚本、曲线、配置快照和SHA256清单保留在以下本地目录，不随Git提交：

```text
/mnt/gjkj/wsz/code/tuoyu/deployments/dep8-decode-baseline-20261029
```

重点文件：`results/summary.json`、`results/concurrency-summary.csv`、`results/per-request.csv.gz`、`results/cXXXX.tar.gz`、`results/{decode,prefill}-metrics.jsonl.gz`、`artifact-sha256.json`。每请求保存TPS、TTFT、token数、取消/结束原因及非空SSE块时序；输出正文只保留首尾摘要和hash。
