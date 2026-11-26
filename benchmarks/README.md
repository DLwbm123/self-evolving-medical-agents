# 医学 Benchmark 准备记录

## 医学 Agent

- **MedAgentBench 上游**。Stanford 官方代码版本为 `99260117137b09f04837a8c18d18a1107efa55ae`，目录为 `.benchmarks/source/c/`。官方 v2 含 300 个任务。根据官方评分器筛出 150 个只读任务，ID 清单位于 `data/medagentbench/query_task_ids.json`，涵盖 task1、task2、task4、task6 和 task7。
- **首轮 FHIR 验收**。此前在本机 `bench-1126-c` 容器运行。FHIR metadata 返回 HTTP 200。使用参考评分器验证了一条真实 GET 任务，正确答案通过，错误答案被拒绝。历史配置位于 `.benchmarks/source/c/configs/tasks/query-local.yaml`，验收日志位于 `.benchmarks/logs/c-smoke.log`。服务器恢复时需按实际容器地址配置 FHIR 入口。
- **VAPA 原协议**。论文匿名仓库返回 HTTP 410，作者的 FHIR 到 EHR 动作映射、预算和评分器暂缺。上游 150 条只读任务的清单按照 Stanford 评分器生成，和 VAPA 适配器的一致性待作者工件恢复后核验。
- **MIMIC-IV v3.1 与 EHRSHOT**。等待核实授权和 VAPA 的构造清单。官方入口为 [PhysioNet MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) 和 [Stanford EHRSHOT](https://som-shahlab.github.io/ehrshot-website/docs/intro/dataset/)。
