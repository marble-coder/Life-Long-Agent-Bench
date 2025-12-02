# DB-Bench 本地 HF + LoRA：先贪心评测，再采样 GRPO 训练

本说明汇总了为 DB-Bench 接入 rLLM 组件并实现“先 greedy 再采样、立刻 GRPO+LoRA 更新”的改动和用法。

## 新增组件
- `src/rllm_integration/dbbench_dataset.py`：从 `data/db_bench.json` 读取标准数据，注册为 rLLM Dataset（含 Verl parquet）。
- `src/rllm_integration/dbbench_agent.py`：多轮累积对话 Agent，使用 `chat_history_items/standard/db_bench.json` 作为开场提示，配合 rLLM 掩码工具，仅对 assistant token 计损。
- `src/rllm_integration/dbbench_env.py`：MySQL 环境封装，3 轮限制；执行 SQL、判定 Answer，奖励：正确 +1、错误 -1、超轮 -0.8、非法 -0.2。
- `src/rllm_integration/train_dbbench_rllm.py`：纯 rLLM `AgentTrainer` 入口（无 greedy 预评测）。
- `src/rllm_integration/run_dbbench_greedy_grpo.py`：本地 HF + LoRA 单机脚本，按“先 greedy 记录准确率，再 group_size 采样、即时 GRPO 更新”的 on-policy 流程；采样时记录 rollout 的 logprob 作为 old_logprob（无参考模型时使用），保持标准 GRPO 比率。

## 运行先贪心再 GRPO 的脚本
依赖：本地 HF 权重、CUDA、Docker（MySQL 镜像）。
```bash
export PYTHONPATH=./
python3 -m src.rllm_integration.run_dbbench_greedy_grpo \
  --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --group_size 4 \
  --max_new_tokens 512 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --learning_rate 2e-5 --weight_decay 0.01 \
  --beta 0.04 --clip_param 0.2 --grad_accum 1 --max_grad_norm 1.0 --num_epochs 1 \
  --reference_model_path <可选: 冻结基座做 KL 参考> \
  --save_dir outputs/dbbench_grpo_lora
```
要点：
- 每条样本先跑一次 greedy（do_sample=False）计准确率，然后对同一条样本采样 `group_size` 次（do_sample=True）并立即 GRPO+LoRA 更新，后续样本使用更新后的权重，保证 on-policy。
- 仅 assistant token 参与 logprob/loss（用 `convert_messages_to_tokens_and_masks` + chat_template）。
- 如果提供 `--reference_model_path`，KL 参考使用一份冻结基座（不挂 LoRA）；否则使用 rollout 时记录的 logprob 作为 old_logprob，仍能得到非 1:1 的 PPO/GRPO 比率。
- 结果：终端打印 greedy accuracy；LoRA/分词器权重保存在 `--save_dir`。

## 若需改超参/显存
- 减少显存：降低 `max_new_tokens`、`group_size`，或调小 LoRA r。
- 强化训练：提高 `num_epochs`/`grad_accum`，或调整 `temperature/top_p`。

## 与 rLLM AgentTrainer 的区别
- `train_dbbench_rllm.py`：直接用 rLLM `AgentTrainer` 跑 GRPO（无 greedy 预评测）。
- `run_dbbench_greedy_grpo.py`：自管 rollout + GRPO 更新，满足“先贪心测评、后采样训练”的在线策略，仍复用 rLLM Agent/Env/掩码逻辑。 可以在后续迁移到 rLLM 的 rollout engine 以分布式扩展。
