# GPU weight-cache 回移验收（2026-09-18）

**结果：单机 BW1000 TP8/C1、64K、W8A8 的限定功能门 PASS。** 保留独立 source/config 版本和上游来源；不等于所有模型、并行或上下文配置都已验收。

## 代码与可维护性

core 从 `glm53-bw1000@11df7d1d629d9a8da6cd22d8971bf54062bc5988` 开始，按来源回移上游 #54921/#56010/#55656/#56669/#56472。每个提交保留 `cherry-pick -x`，上下文差异见 manifest。测试兼容、daemon workspace 初始化、spawn general-plugin 初始化单独提交；HCU UUID、量化恢复、GLM router 引用修复放在插件。

core `tools/check_backports.py` 只读检查上游 SHA 是否进入目标版本；源提交已包含不等于本地 HCU 适配可直接删除。原始来源保留在 `manifest.json`，不依赖会在 squash/rebase 中变化的本地 SHA。以后升级先用上游原生实现，再逐项复核 MANUAL_REVIEW。

## 精确测试来源

- Core runtime 源码：`325a6a5f8f`；core PR 文档 HEAD 后续变化不改变此 runtime subtree。
- Daemon/普通磁盘基线插件：`1ac06f4d73ba91ffd051b63ac3c2a466957b3992`。
- 两次成功的 cache engine 插件：`064908bf97c8b9ee6a017fea78c68b000a6123c9`。
- 单容器、Torch 2.11.0 / HIP 6.3.26113，复用原镜像和 v029-r1 native 库，不构建镜像。
- TP8/PP1/DP1/DCP1，max_model_len=65536，C1，batch=8192，block 参数256（引擎按 hybrid cache 调整），BF16，W8A8 slimquant，MTP 关闭，eager、prefix cache/async scheduling关闭，每卡 KV 固定4 GiB。
- daemon/engine 共享影响后端选择的参数。磁盘 gpu-memory-utilization=0.87；cache engine 按 daemon-ready 空闲显存用0.25。这是 admission 比例，KV 字节预算相同。

客户端修复仅在 preprocessed-weights 上下文重绑非 Module router 的参数引用，不改变 owner 权重布局或普通加载。本次受控实验保持原 daemon/源码不动，仅使用独立 client overlay，结合生成前后指纹验证。最终统一版源码覆盖层已准备，但**没有再对统一最终版执行另一轮从零 daemon 冷加载**；其新增 normal-load hook 为 no-op，CPU 回归覆盖。一般切换模型或框架/插件版本仍要求重建 daemon。

## 实测结果

| 项目 | 结果 |
|---|---|
| 维护工具 | 8 项 Git 行为测试 PASS；真实官方目标的 ancestry 检查通过 |
| core 兼容回归 | 真实运行容器5项 PASS |
| 插件回归 | 47项 CPU行为测试 PASS，使用fake平台/导入依赖 |
| 设备标识 | 8卡UUID唯一，HIP可见卡7/0换序 PASS |
| 模型构造 | TP8 meta 8/8，TP1 Triton后备路径 gate PASS |
| 普通加载基线 | 中文、算术、合成图片3请求 PASS；C1、重试0 |
| 缓存恢复 | 两次独立 engine，均8个 rank 各映射1633 tensors，zero_copy、fallback=false，不重载checkpoint张量；仍读取config/头部元数据 |
| 输出对照 | 两次共6请求，完整token序列和消息与基线一致；每轮输出13/3/17 tokens |
| MLA 共享内容 | 每rank198条指纹/视图记录，覆盖11层KV投影、W_UV/W_UK_T及4类scale和别名；两次生成后均8/8位级不变 |
| 显存与生命周期 | 每次 engine 退出后，8卡物理显存逐字节回到同一 daemon-only 记录；最终所有实验进程退出、每卡空闲63.681640625 GiB（含测量进程自身context） |

## 加载耗时和显存解释

磁盘权重装载日志为1561.37秒。按8 rank 的 `Model loading took` 日志取中位数：普通加载1645.53秒，两次缓存恢复81.42/81.17秒，约20倍的**模型加载阶段**差异。

从启动脚本日志名时间到API路由注册，普通加载1997秒，两次缓存263/262秒。这是同机顺序实验，包含已有JIT/文件缓存热态差异，不是严格隔离的单因素基准，也不代表稳态TPM收益。

普通加载报告模型40.6 GiB/卡；cache engine报告新增0.57 GiB/卡，**不代表模型总共只占0.57 GiB**，权重仍由daemon持有。按相同测量方式，daemon+engine运行时比单体基线多用约2.25–2.80 GiB/卡。每次engine退出后，这些客户端占用回收，daemon-only空闲字节与初始完全一致。

## 发现并修复的问题

1. 独立daemon未经过GPUWorker，缺少workspace初始化。meta构造失败后补齐真实入口，并通过回归与完整daemon加载。
2. spawn子进程不继承父进程general-plugin callbacks。真实状态对照证实GLM/INT8/MLA回调缺失；rank入口显式加载插件。
3. 首次cache engine虽完成IPC映射，但profile forward发生HSA nil fault。真实binder CPU红测定位：非nn.Module的router持有旧meta correction-bias，注册Parameter已恢复却未更新该引用。插件模型级hook在缓存恢复阶段重绑，之后两次服务启动及输出对照通过。

`moe_align_block_size` 的launch-bounds警告在成功磁盘基线也存在；同绑定的真实GPU balanced/skewed/random三例对齐校验全部通过，不能将该警告作为本次缓存故障原因。失败记录均保留，未隐去失败尝试。

## 启动与退出边界

脚本：`/gjkj/scripts/glm53/weight-cache/`，包含 prepare_runtime、init_container、start_daemon、start_engine，以及独立验收工具。最终默认源码覆盖层为 `/gjkj/data/cch/glm53-bw1000/releases/weight-cache-r1`。脚本不重建镜像、不默默回退磁盘；启动engine前需按daemon就绪后的实测空闲显存设置预算。

08验收资源已回收并保留实验容器/日志；10未变。MTP、1M缓存模式、C8、跨容器/多机、FP8/BF16模型缓存及完整质量/吞吐基准均 NOT_RUN。W8A8/FP8/BF16三个checkpoint目录已单独完成复制和逐文件SHA256核对，均在 `/gjkj/models`；这不等于后两种模型的GPU缓存已验收。

小型证据汇总：`/gjkj/data/cch/glm53-bw1000/ops/weight-cache-20260918/acceptance-summary.json`。原始响应、完整日志和设备转储仅留在节点，不进入PR。
