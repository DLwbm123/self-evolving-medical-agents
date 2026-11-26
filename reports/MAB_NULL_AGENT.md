# MedAgentBench 空 FINISH 审计

2026-11-26，Stanford MedAgentBench v2 官方 300 题。评分器来自上游仓库 README 提供的 `refsol.py`，环境是仓库指定的官方 FHIR 镜像。每题的 agent 历史只有 `FINISH([])`，评分器接收原版任务运行时会提取的 `[]`。评分器查询本机 FHIR 以计算参考结果。运行记录只保留计数，300 题均完成评分。

| 范围 | 通过 | 总数 | 通过率 |
|---|---:|---:|---:|
| 官方全集 | 70 | 300 | 23.33% |
| 只读子集 | 0 | 150 | 0.00% |
| task5 | 28 | 30 | 93.33% |
| task9 | 29 | 30 | 96.67% |
| task10 | 13 | 30 | 43.33% |

其余七个类别各 0/30。三个出现空通过的类别都有条件执行分支。当前结果支持对这些类别检查 Agent 是否真的查询了相关证据。[官方仓库](https://github.com/stanfordmlgroup/MedAgentBench) · [既有审计论文](https://arxiv.org/abs/2607.01470)

### 读取证据门控的对照

对官方评分通过的 70 道题加一项证据条件。Agent 的历史必须包含成功的 FHIR Observation GET，患者和检验项目都与任务匹配。空历史通过 0/70。把评分器实际成功的同患者、同项目查询用作参考正对照，70/70 满足证据条件。查询归属到另一患者的负对照为 0/70。这测的是门控判定，完整 Agent 评测仍需把通过校验的读取事件从运行时传入评分器。

本机 FHIR 的 695 个 Patient 资源均带有 identifier 和 meta.source，其中 98 个资源能按现有官方任务标识直接匹配。其余资源是一批待核验的候选。评估其作为训练患者的资格仍需确认去重、病例身份、来源及使用条件。

聚合结果位于 `.benchmarks/logs/mab-null-official.json`。审计入口为 `benchmarks/audit_medagentbench.py`。

已有研究在 600 个任务上报告直接空提交通过率为 41.7%。本项目 300 题上的 23.33% 是另一任务范围的实测。本文审计可用作奖励设计对照；尚未检验读取证据后是否真正利用它，以及加入奖励后是否改善留出任务的主动查询和正确决策。[World Feedback for Clinical Agents: Diagnosing RL in FHIR Environments](https://arxiv.org/abs/2607.01470)
