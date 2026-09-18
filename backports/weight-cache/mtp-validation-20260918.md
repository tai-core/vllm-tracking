# MTP 双 owner 组合验收：10 保留新服务，视频限制待处理

日期：2026-09-18。本记录区分工程门、视频答案质量与部署决策。**08 的限定
组合工程验收 PASS；10 已完成切换，用户明确选择保留新服务、后续处理视频
问题，部署状态为 `accepted-with-video-limitation`。** 这不表示 10 的整轮
请求或视频精度全部通过。

## 冻结版本与配置

- Release：`weight-cache-mtp-r1`。
- Core：`cd93fdfaabbb2f9ed9ccb8f4ef76ce8b75ac5317`。
- Plugin：`49ea0bed7a775eadc74f1043483e4c959e28dc76`。
- 模型：GLM-5.3 Flash W8A8；单机 TP8/PP1/DP1/DCP1，64K、C8。
- MTP：k=4，`use_local_argmax_reduction=true`；mHC：AITER-RNE。
- KV 预算：每卡 8 GiB；IPC：`zero_copy`、`fallback=false`。

主模型与 MTP 使用两个 TP8 owner group、同源 checkpoint；双方使用同一版
role-aware 协议。该验收没有复用旧的 target-only owner，也不证明任意跨
版本缓存兼容。上游 PR57312 的来源及评估时 OPEN/未合并状态见
[来源说明](README.md#mtp-extension-pinned-open-upstream-feature) 和
[manifest](manifest.json)。

## 08 构造、映射与内存

| 检查 | 结果 |
| --- | --- |
| 主/草稿共 16 个 role constructor | 全部完成；unsupported=0，恢复 hook 检查均为 True |
| 主模型每 rank 导出 | 1,633 entries，2,379 aliases |
| MTP 每 rank 导出 | 33 entries，25 aliases |
| 实际 IPC 映射 | 16 条 role 映射日志均已观察到 |
| 日志中的模型加载时间中位数 | 约 6.27 s |
| 主模型 state storage | 每卡 38.2568 GiB |
| MTP state storage | 每卡 2.42485 GiB |
| 两组 owners 启动后最小可用显存 | 17.24 GiB |

state storage 不包含 allocator/context 开销，不能当作进程显存总量。
MTP 组加载的是草稿模型，没有再复制一个完整主模型；k=4 控制草稿迭代，
不代表存在四份草稿模型。

KV 初始化估算为 566,051 tokens，约 8.64 份 64K。这个值来自启动时的
容量估算，**不是 8 条同时满 64K 请求的容量实测**，也不是最大 TPM 或
SLO 容量结论。

## 08 请求、推测解码与共享状态

16/16 请求工程验收 **PASS**，实际观察到 engine 并发峰值 8。MTP 计数为
208 次 draft、832 draft tokens、329 accepted tokens，证实 k4 推测路径
实际执行；该小样本不构成吞吐或接受率的生产预测。

普通文本、图像和单色视频输出正常。双色视频的 `known_answer` 仍为
**FAIL**，属于此前 10 节点已记录的同类质量问题，输出仍 coherent。
“工程 PASS”没有把这些已知答案失败改写为精度 PASS。

前后审计覆盖选定的 target MLA/router tensor，以及**全部 draft tensor**，
内容和 metadata 均保持一致，**PASS**。target 的检查范围并非全部主模型
tensor；不把选择性主模型审计表述为全模型逐项审计。

08 随后完成一次暖重启，追加 8 项功能检查 PASS；前后选定 target 与全部
draft tensor 审计 PASS。最后清理全部实验资源，8 卡显存恢复本轮初始值，
资源回收 PASS。这个清理结论不适用于仍按用户要求运行的 10 新服务。

## RNE 精度边界

历史 RNE 严格全张量数值矩阵仍是 **8/12 PASS、4/12 FAIL**。用户已明确
接受本次部署使用该候选；原数值门限没有放宽，历史 FAIL 没有改成 PASS。
此次正常输出和共享状态检查不能替代该严格数值门，也不证明双色视频
答案质量问题已经解决。

## 10 最终运行快照与独立验收

新容器 `glm53-weight-cache-10` 为 **running**，CID：
`0e03d629e86d7d3e7d50dca6fd7e8aa6171235da867b38608541725a0e7c8991`。
daemon PID 为 22，engine PID 为 2125，health 返回 200。服务名为
`glm-5.3-flash`，同时保留 `glm-5.3-flash-w8a8` 兼容别名。配置为上述
64K/C8、MTP k4/local argmax、AITER-RNE、KV 8 GiB。旧容器 `e026`（CID
前缀）保留为 stopped，供回退使用。

16 个 owner socket 均存在，socket 检查 **PASS**。另对 8 个 engine worker
核对实际 RNE 模块映射与选择日志，均 **PASS**。每 rank 主模型映射
1,633 entries，草稿映射 33 entries。8 rank 的模型加载日志
中位数为 **11.058649 s**，这是模型加载阶段，**不是整服启动耗时**。
KV 启动估算仍为 566,051 tokens、约 8.64 份 64K，不改变前述满长容量
尚未实测的边界。

| 10 独立检查 | 结果 |
| --- | --- |
| C8 并发门 | 8/8 PASS；实际 engine 并发峰值 8 |
| C8 期间 MTP 计数 | 207 次 draft、828 draft tokens、310 accepted tokens |
| 基础文字、算术、中文、红图、蓝图、单红视频 | 共 6 项 PASS |
| 原红蓝视频问法，512-token 预算 | FAIL：仅 reasoning，没有最终答案 |
| 使用原问法单独发起的复现请求 | 仍 FAIL，没有最终答案 |
| 改为一句话问法 | 返回连贯最终答案，但遗漏蓝色；`known_answer` 仍 FAIL |

业务重试配置为 0；上述复现是单独发起的诊断请求，不是自动业务重试。
本轮没有进行 Torch mHC 对照，尚未把视频现象归因于 RNE、MTP 或模型本身。
因此不能将这些结果汇总成“整轮请求全部 PASS”或“视频精度 PASS”。用户在
了解上述结果后明确选择**保留新服务，视频问题后续处理**；这是有明确视频
限制的部署接受，没有豁免或改写视频质量结果。

## 检查与发布状态

Core Draft PR21 已发布；本地源码冻结时的 CPU 合约、维护工具和 pre-commit
结果见 README。GitHub CI 因账户 payment/spending-limit 限制未启动，状态
是 **NOT_RUN**，不能记为代码检查 PASS。此次文档更新也不代表 PR 已合并。

本次文档更新没有修改产品源码、测试、运行脚本或模型权重，也没有执行
节点操作。本文只保留汇总，不包含原始响应、完整日志或 vendor 源码片段。
