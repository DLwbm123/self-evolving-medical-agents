# Self-evolving medical agents

医学 Agent 研究代码，研究可验证的步骤奖励、轨迹分叉和错误恢复训练。当前实现覆盖平均动脉压（MAP）和体质指数（BMI）的查询、证据绑定、计算和提交，以及 MedAgentBench 的只读证据审计。[实验进展](reports/PROGRESS.md)

在仓库根目录检查确定性环境和优势计算，只需 Python 3.10+。

```sh
python -m unittest discover -s tests -v
```

以下命令在装有 PyTorch、Transformers 和 PEFT 的 CUDA 服务器上运行同一组合成留出任务。`MODEL` 指向已下载的 Qwen2.5-0.5B-Instruct，`--synthetic` 模式在内存中生成所有数值。

```sh
export PYTHONPATH=.
python benchmarks/sample_model_rollouts.py "$MODEL" . runs/clean --synthetic --synthetic-holdout-seed 4 --device cuda --prompt menu --warmup --warmup-epochs 3
python benchmarks/sample_model_rollouts.py "$MODEL" . runs/recovery --synthetic --synthetic-holdout-seed 4 --device cuda --prompt menu --warmup --warmup-epochs 3 --recovery-warmup
```

`prepare_medcalc_split.py` 接受官方训练和测试 CSV，按病历关联组划分数据。`audit_medagentbench.py` 需要用户提供官方代码及可用的 FHIR 服务。`medical/credit.py` 实现轨迹优势及分叉、状态碰撞和同轮次分组的局部优势。分叉轨迹共享的前缀不重复计入采样。
