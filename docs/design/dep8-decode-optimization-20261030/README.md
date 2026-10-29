# DEP8 Decode 优化与实验归档（2026-10-30）

## 源码与部署边界

本次 vLLM 代码提交只引入单节点 DP 元数据 SHM 同步，默认关闭。启用 `VLLM_DP_METADATA_SHM=1` 后，用既有 MessageQueue 交换六个字段；不能共享 SHM 时初始化选择原 Gloo 路径，协议异常或超时不会在半步中切换 backend。EP dispatch/combine 未修改。

DeepGEMM MoE 分块优化位于独立仓库 `/tmp/dep8-deepgemm-source` 的 `perf/dep8-b300-small-batch` 分支，提交 `4e2c9a9`，基线 `1e1842a833699298f7afc02eefb2ed168fff6938`。vLLM 不内置这项源码；仅快进 vLLM dev 不会自动安装优化后的 DeepGEMM 扩展。可复现补丁附后，构建步骤和脚本保留在本地实验目录。

TY19 开发实例保留 SHM 和已验证的 MoE candidate.so。其 SHA256 为 `c14a91db28ab64def3d52102518b59015ca3ba3dee58a88c1c59710e039570a0`。运行镜像原有额外改动，部署采用运行文件叠加 SHM 修改，不能宣称当前 dev 与运行镜像全部源码完全一致。最新一次只增加 profiler 启动配置，采集已停止；未部署 Indexer 新补丁。

## 结果索引

- [SHM A/B](shm.md)：128 并发总输出 TPS 17958.38→22646.65（+26.11%），256 为36697.74→37943.95（+3.40%）；含逐轮、每卡、单请求吞吐及启动和回退说明。
- [SHM 后 profile](profile.md)：32份trace；融合MoE约7.28/7.58ms每rank每步，kernel覆盖率91.1%/98.6%。Indexer投影漏调优的单算子筛选仅部分尺寸获益，未部署。最终无采样复测22446.08/37840.11 TPS，0错误。
- [MoE A/B](moe.md)：M64候选在256并发两轮平均+4.43%，128/512没有一致收益；EPLB映射和autotune未固定。
- [质量冒烟检查](quality.md)：376/376标准题通过、8篇长文全文检查；发现普通回答技术错误，尚无Gloo配对准确率评测。用户要求暂缓进一步语义评测。

SHM 已完成15360次协议值/布局检查及1536次真实旧/新DP函数比较，含空闲/不均匀rank、序号错误、缺失rank超时、padding与ubatching。服务使用正常CUDA Graph。测试脚本和原始结果存放于本地实验目录，当前提交仅归档Markdown，不包含trace、gzip、JSON数据或二进制。

此前Gloo内部profile将重点定位到每步约9–10ms的CPU collective。它与GPU执行可能重叠，不能直接当成GPU净节省；本次profile也没有分离融合MoE内部计算和EP等待。下一步优先拆解融合MoE并关联batch/EPLB，同时调查128并发图间隙。

## 当前启动配置

```sh
docker compose \
  -f /data/services/tuoyu-cep8-dev-20261028/service-r22/TY19.compose.yaml \
  -f /data/services/tuoyu-dep8-moe-ab-20261029/candidate.override.json \
  -f /data/services/tuoyu-dep8-dp-shm-ab-20261030/shm.override.json \
  -f /data/services/tuoyu-dep8-shm-profile-20261030/profile.override.json \
  up -d --no-deps decode
```

移除profile override并重建可去掉profiler启动配置；当前采集已经关闭，无需通过重启停止采集。移除SHM override可回退Gloo，移除MoE override可回退镜像内扩展。sidecar在Decode health200后启动。模型/DP8/EP8/DSpark5/EPLB和完整参数见各报告及已有[基线文档](../dep8-decode-baseline-20261029/README.md)。

## DeepGEMM 可复现源码补丁

```diff
diff --git a/csrc/jit_kernels/heuristics/mega_moe.hpp b/csrc/jit_kernels/heuristics/mega_moe.hpp
index e018a33..1b046b8 100644
--- a/csrc/jit_kernels/heuristics/mega_moe.hpp
+++ b/csrc/jit_kernels/heuristics/mega_moe.hpp
@@ -100,6 +100,16 @@ static std::tuple<int, int, int, int, int> get_block_config_for_mega_moe(
                 break;
         }
     }
+    // B300 EP8 decode: cover skewed routing without reloading the same
+    // expert weights for a second M tile. Keep the public tile query and
+    // kernel launch on this shared selection path (also used for shared SFs).
+    if (mma_kind == MmaKind::MXFP8FP4 and num_ranks == 8 and
+        num_experts == 400 and num_topk == 6 and
+        num_tokens >= 160 and num_tokens <= 200 and
+        jit->device.get_arch_major() == 10 and jit->device.get_arch_minor() == 3 and
+        runtime->get_num_sms() == 148)
+        block_m = 64;
+
     const int store_block_m = block_m <= 16 ? 8 : block_m <= 64 ? 16 : block_m <= 192 ? 32 : 40;
     const int block_k = 128 / get_num_mma_elem_bytes(mma_kind);
 
```
