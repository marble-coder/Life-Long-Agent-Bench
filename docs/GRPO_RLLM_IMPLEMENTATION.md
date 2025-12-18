# GRPO Training Callback with rllm Framework Integration

## 概述

本文档记录了使用rllm框架实现的GRPO (Group Relative Policy Optimization) 训练回调的所有新增和修改内容。

---

## 新增文件

### 1. 主实现文件
**路径**: `src/callbacks/instance/grpo_training_callback_rllm.py`

**功能**: GRPO训练回调的核心实现

**框架集成**:
| 组件 | 来源 | 用途 |
|------|------|------|
| `convert_messages_to_tokens_and_masks` | rllm.agents.utils | 消息tokenization和action mask生成 |
| `ChatTemplateParser` | rllm.parser | 聊天模板解析 |
| `compute_policy_loss` | verl.trainer.ppo.core_algos | PPO clipped loss计算 |
| `kl_penalty` | verl.trainer.ppo.core_algos | KL散度计算 |
| LoRA | peft | 高效微调 |
| TensorBoard | torch.utils.tensorboard | 实时监控 |

**核心类**: `GRPOTrainingCallbackRLLM`

**关键方法**:
```python
on_session_create()      # 初始化模型和LoRA
on_agent_inference()     # 缓存采样时的logprobs（稳定cache key）
on_task_complete()       # 收集attempt，触发训练
_train_on_group()        # GRPO训练核心逻辑
_calc_reward()           # 奖励计算（正确+1.0, 完成+0.5）
```

**数值稳定性保护**:
- logprob差值clamp到[-20, 20]
- loss clamp保护
- 异常检测（loss > 50跳过）

---

### 2. Callback定义配置
**路径**: `configs/components/callbacks/grpo_training_callback_rllm.yaml`

```yaml
grpo_training_callback_rllm:
  module: "src.callbacks.instance.grpo_training_callback_rllm.GRPOTrainingCallbackRLLM"
  parameters:
    # config_path: path to GRPO config YAML (required in custom_parameters)
```

---

### 3. GRPO超参数配置
**路径**: `configs/components/rl/db_bench_grpo_rllm.yaml`

```yaml
group_size: 4                    # 每个样本采样次数
best_metric_strategy: best_reward

generation:
  do_sample: true
  temperature: 0.8
  top_p: 0.95
  max_new_tokens: 512
  num_beams: 1

grpo:
  beta: 0.1                      # KL系数
  clip_param: 0.2                # PPO clip参数
  normalize_rewards: true        # GRPO核心：组内归一化
  use_best_of_n: false          # 使用全部样本
  reference_model_path: null

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

optim:
  learning_rate: 2.0e-5
  weight_decay: 0.01
  max_grad_norm: 0.5
  num_train_epochs: 1
  gradient_accumulation_steps: 1

save:
  lora_output_dir: "outputs/{TIMESTAMP}/grpo_rllm_lora"

monitoring:
  tensorboard: true              # 启用TensorBoard
  log_interval: 1
```

---

### 4. 实验配置文件
**路径**: `configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/grpo_rllm.yaml`

```yaml
import:
  - ../task.yaml
  - ../../../agent.yaml
  - ../../../../../../definition.yaml

assignment_config:
  callback_dict:
    callback_0:
      name: current_session_saving_callback
    callback_1:
      name: consecutive_abnormal_agent_inference_process_handling_callback
    callback_grpo_rllm:
      name: grpo_training_callback_rllm
      custom_parameters:
        config_path: "./configs/components/rl/db_bench_grpo_rllm.yaml"
  output_dir: outputs/{TIMESTAMP}
  sample_order: default

environment_config:
  use_task_client_flag: false
```

---

## 修改的文件

### 1. Callback导入注册
**路径**: `src/callbacks/instance/__init__.py`

**修改内容**: 添加了新callback的导入和导出

```python
# 新增导入
from .grpo_training_callback_rllm import GRPOTrainingCallbackRLLM

# 新增导出
__all__ = [
    # ... 其他callbacks
    "GRPOTrainingCallbackRLLM",
]
```

---

### 2. Constructor注册
**路径**: `src/callbacks/constructor.py`

**修改内容**: 在match-case中添加新callback的处理分支

```python
case GRPOTrainingCallbackRLLM.__name__:
    unique_flag = GRPOTrainingCallbackRLLM.is_unique()
```

---

### 3. 全局定义注册
**路径**: `configs/definition.yaml`

**修改内容**: 在callback_dict中注册新callback

```yaml
callback_dict:
  import:
    # ... 其他callbacks
    - ./components/callbacks/grpo_training_callback_rllm.yaml # GRPO + LoRA using rllm/verl framework
```

---

## 算法细节

### 奖励函数
```python
def _calc_reward(session: Session) -> float:
    reward = 0.0
    if outcome == CORRECT:
        reward += 1.0      # 结果正确
    if status == COMPLETED:
        reward += 0.5      # 状态完成
    return reward          # 纯正向，不扣分
```

### PPO Clipped Loss
```python
# 参考: https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py

# ratio = π(a|s) / π_old(a|s)
ratio = exp(new_logps - old_logps)
clipped_ratio = clamp(ratio, 1-ε, 1+ε)

# L^CLIP = min(ratio * A, clip(ratio) * A)
policy_loss = -min(ratio * advantage, clipped_ratio * advantage).mean()
```

### KL Penalty (Reverse KL)
```python
# KL(π || π_ref) ≈ r - log(r) - 1, where r = π/π_ref
ratio_ref = exp(new_logps - ref_logps)
per_token_kl = ratio_ref - log(ratio_ref) - 1.0
kl_loss = beta * per_token_kl.mean()
```

### 采样Logprobs缓存
```python
# 使用agent turn count作为稳定的cache key
# 解决了chat_history长度在on_agent_inference和on_task_complete之间变化的问题
turn_count = len([item for item in chat_history if item.role == AGENT])
cache_key = f"{sample_index}_turn_{turn_count}"
```

---

## 监控输出

### TensorBoard指标
| 指标 | 含义 |
|------|------|
| train/loss_total | 总loss |
| train/loss_policy | 策略loss |
| train/loss_kl | KL惩罚 |
| train/group_accuracy | 组内准确率 |
| train/ratio_mean | PPO ratio均值 |
| train/ratio_max | PPO ratio最大值 |
| train/kl_mean | KL散度均值 |
| train/grad_norm | 梯度范数 |
| train/reward_mean | 奖励均值 |
| train/rewards | 奖励分布直方图 |

### TSV日志列
```
global_step | sample_index | attempt_idx | loss_total | loss_policy | loss_kl |
reward | group_acc | ratio_mean | ratio_max | kl_mean | grad_norm | train_started
```

---

## 使用方法

### 运行训练
```bash
python src/run_experiment.py \
    --config_path "configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/grpo_rllm.yaml" \
    --max_samples 10
```

### 监控训练
```bash
tensorboard --logdir outputs/*/tensorboard
```

---

## 依赖

```
torch
transformers
peft
rllm
tensorboard
pyyaml
```

---

## 文件结构总结

```
实验代码版本控制/
├── src/callbacks/
│   ├── constructor.py                           [修改] 添加match-case分支
│   └── instance/
│       ├── __init__.py                          [修改] 添加import
│       └── grpo_training_callback_rllm.py       [新增] 主实现 (~580行)
├── configs/
│   ├── definition.yaml                          [修改] 注册callback
│   ├── components/
│   │   ├── callbacks/
│   │   │   └── grpo_training_callback_rllm.yaml [新增] callback定义
│   │   └── rl/
│   │       └── db_bench_grpo_rllm.yaml          [新增] 超参数配置
│   └── assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/
│       └── grpo_rllm.yaml                       [新增] 实验配置
└── docs/
    └── GRPO_RLLM_IMPLEMENTATION.md              [新增] 本文档
```

---

## 参考

- [verl PPO core_algos](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py)
- [rllm framework](https://github.com/rllm-project/rllm)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
