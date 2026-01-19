# Reflection GRPO 系统架构文档

## 概述

Reflection GRPO 是一个嵌套的强化学习系统，用于训练一个 **reflection model**（反思模型），该模型能够分析 agent 的执行轨迹并生成高质量的 insights，从而指导后续的任务执行。

## 核心架构

### 1. 嵌套 GRPO 结构

```
每个样本:
├── Greedy Evaluation (1次，用于 metric)
│   ├── 贪婪解码，确定性
│   ├── 结果用于计算准确率
│   └── 轨迹 + reflection 存储到 memory
│
└── Reflection GRPO 训练
    ├── 外层: k1=4 个 reflections (训练 reflection model)
    │   ├── 基于 greedy trajectory 生成 reflection
    │   ├── 采样模式 (temperature=0.7, top_p=0.9)
    │   └── Reward = group_acc (对应的 k2 个 rollouts 的准确率)
    │
    └── 内层: 每个 reflection 对应 k2=4 个 rollouts (训练 base model)
        ├── Reflection insight 注入到 prompt
        ├── 使用 virtual_sample_index 确保正确分组
        └── 由 grpo_training_callback_rllm 处理
```

### 2. 两种 Reflection 的区分

系统中有**两种**不同目的的 reflection：

#### Memory Reflection (长期存储用)
- **生成时机**: `on_state_save` 时自动生成
- **生成方式**: 贪婪解码 (`do_sample=false`, `temperature=0.0`)
- **生成器**: `previous_sample_embedding_callback._call_reflection_api()`
- **目的**: 稳定的知识存储，与 trajectory 一起召回
- **格式**: JSON 格式的结构化分析
  ```json
  {
    "diagnosis_reasoning": "分析失败/成功的根本原因...",
    "error_type": "Syntax Error | Logic Error | ...",
    "insight": "提炼的通用规则或建议 (<50 words)",
    "tags": ["相关概念"]
  }
  ```

#### GRPO Training Reflection (训练用)
- **生成时机**: Reflection GRPO 循环中生成 k1 次
- **生成方式**: 采样模式 (`temperature=0.7`, `top_p=0.9`)
- **生成器**: `reflection_grpo_training_callback.generate_reflection()`
- **目的**: 探索不同的分析角度，训练 reflection model
- **特点**:
  - 多样性（每次生成不同）
  - 基于 greedy trajectory 分析
  - Reward = 对应 k2 个 rollouts 的 group_acc

## 完整执行流程

### Step 1: Greedy Evaluation
```python
# run_experiment.py line 529-556
greedy_session = Session(...)
# 使用贪婪解码执行
agent._inference_config_dict = {
    "do_sample": False,
    "num_beams": 1,
    "temperature": None,
    ...
}
task.complete(greedy_session)  # finish_reason = "GREEDY_EVAL"
```

**结果**:
- `greedy_session` 记录到 `session_list`（用于 metric）
- 暂存到 `_pending_session_to_store`（延迟存储）

### Step 2: Reflection GRPO 训练循环

```python
# run_experiment.py line 579-716
greedy_correct = (greedy_session.evaluation_record.outcome == CORRECT)

# 外层循环: k1=4 个 reflections
for reflection_id in range(k1):
    # 生成 reflection (采样模式)
    reflection_text, insight_text, messages = reflection_grpo_cb.generate_reflection(
        current_query=query,
        greedy_trajectory=trajectory,
        greedy_correct=greedy_correct,
        sample_index=sample_index,
        reflection_id=reflection_id
    )

    # 注册到 reflection GRPO callback
    reflection_grpo_cb.register_reflection_generation(...)

    # 内层循环: k2=4 个 base model rollouts
    for rollout_id in range(k2):
        virtual_sample_index = f"{sample_index}_r{reflection_id}"
        session = Session(sample_index=virtual_sample_index)

        # 注入 reflection insight 到 prompt
        setattr(session, '_reflection_text', insight_text)

        # 执行 rollout
        agent.inference(session)
        task.interact(session)
        task.complete(session)

        # 注册结果
        reflection_grpo_cb.register_rollout_result(
            sample_index, reflection_id, is_correct, session
        )
```

**结果**:
- Reflection model 根据 group_acc 进行训练
- Base model 根据 best_reward 进行训练（由 grpo_training_callback_rllm 处理）

### Step 3: 存储到 Memory

```python
# previous_sample_embedding_callback.py line 395-438
def on_state_save(callback_args):
    pending_session, outcome, error_message = self._pending_session_to_store
    query_text = self._extract_query_text(pending_session)

    # 自动生成 reflection (贪婪解码)
    if self.enable_reflection:
        trajectory_text = self._extract_trajectory_text(pending_session, ...)
        reflection = self._call_reflection_api(
            query_text, trajectory_text, outcome, error_message
        )

    # 存储到 memory
    record = _SessionRecord(
        session=pending_session,
        query_text=query_text,
        embedding=embedding,
        reflection=reflection,  # Memory Reflection
        outcome=outcome,
        error_message=error_message,
    )
    self._stored_records.append(record)
```

**结果**:
- `{query, trajectory, reflection, outcome}` 存储到 memory 数据库
- 保存到 `utilized_session_list.json`

### Step 4: 召回和注入（下一个 sample）

```python
# previous_sample_embedding_callback.py line 320-374
def on_task_reset(callback_args):
    current_query = self._extract_query_text(session)

    # Embedding 召回 top-k 相似轨迹
    selected_records = self._select_topk_records(current_query, ...)

    # 渲染 example_text (包含 trajectory + reflection)
    example_text = self._render_example_text(selected_records, ...)

    # 注入 task-specific reflection (如果是 Reflection GRPO rollout)
    reflection_text = getattr(session, '_reflection_text', '')
    if reflection_text:
        reflection_section = f"\n**Task-Specific Hint:**\n{reflection_text}\n\n"
        example_text = example_text + reflection_section

    # 替换 prompt
    first_user_prompt = self.original_first_user_prompt.replace(
        self.pattern, example_text
    )
```

**结果**:
- Prompt 包含 4 条历史轨迹（每条包含 trajectory + reflection）
- 如果是 Reflection GRPO rollout，还包含当前的 task-specific reflection

## Reflection 生成的 Prompt

### Chain-of-Thought Analysis Prompt

```
System:
You are an expert AI Agent Analyst and Prompt Engineer specialized in
optimizing Large Language Model agents for complex reasoning tasks.

User:
### Input Context
**1. User Query:**
{current_query}

**2. Agent Trajectory:**
{greedy_trajectory}

**3. Evaluation Outcome:**
{Success/Failure}
{error_message}

---

### Analysis Instructions (Chain of Thought)
**Step 1: Diagnosis (Root Cause Analysis)**
- Pinpoint the exact turn where logic diverged (failure) or
  identify the critical decision that made the solution work (success)

**Step 2: Abstraction (Generalization)**
- Formulate a general heuristic (not specific variable names)
- Example: "When querying metric X, always perform an INNER JOIN..."

**Step 3: Refinement (Actionability)**
- Condense into a single, high-impact tip (<50 words)

### Output Format
{
  "diagnosis_reasoning": "...",
  "error_type": "Syntax Error | Logic Error | ...",
  "insight": "The refined, generalizable rule or tip.",
  "tags": ["relevant_concept"]
}
```

## 训练细节

### Reflection Model 训练
- **模型**: Llama-3.1-8B-Instruct (LoRA r=16, alpha=32)
- **Reward**: Group accuracy (k2 个 rollouts 中正确的比例)
- **Loss**: GRPO loss (对齐 verl 实现)
  ```python
  pg_loss = compute_policy_loss(old_logprob, logprob, advantages, ...)
  kl_loss = kl_penalty(logprob, ref_logprob, mode="k3")
  total_loss = pg_loss + beta * kl_loss  # beta=0.0 (不使用 KL loss)
  ```
- **组内归一化**: 每个 group (k1=4 reflections) 内部归一化 advantage
- **累积训练**: 每累积 2 个 group 训练一次
- **Warmup**: 前 50 个样本只评估不训练

### Base Model 训练
- **模型**: Llama-3.1-8B-Instruct (LoRA r=16, alpha=32)
- **Group size**: k2=4 rollouts per reflection
- **处理器**: `grpo_training_callback_rllm`
- **Virtual sample_index**: 使用 `f"{sample_index}_r{reflection_id}"` 确保分组正确

## 额外功能：成功轨迹存储（可配置）

如果 greedy decode 失败，但 k1×k2 rollouts 中有成功的，可以选择存储**第一条成功轨迹**到 memory。

### 配置参数

```yaml
# configs/components/rl/reflection_grpo.yaml
store_rollout_success_to_memory: true  # 默认 true，可设为 false 禁用
```

### 工作原理

```python
# 1. register_rollout_result 记录第一个成功的 session
if self.store_rollout_success_to_memory:  # ← 可配置
    greedy_correct = self._greedy_results.get(sample_index, True)
    if is_correct and not greedy_correct and self._first_success_candidate.get(sample_index) is None:
        self._first_success_candidate[sample_index] = session.model_copy(deep=True)

# 2. 在所有 reflections 完成后，存储到 memory
if cb.store_rollout_success_to_memory:  # ← 可配置
    success_candidate = cb.get_success_candidate_for_memory(sample_index)
    if success_candidate is not None:
        mem_cb._pending_session_to_store = (success_candidate, "success", "")
```

### 使用场景

**启用 (`true`)** - 适合以下情况：
- 训练早期，memory 中成功案例较少
- 希望快速积累成功经验
- Reflection 能够引导模型找到 greedy 未发现的解法

**禁用 (`false`)** - 适合以下情况：
- Memory 已经有足够多的成功案例
- 只关心 greedy decode 的性能提升
- 避免 rollout 的"侥幸成功"污染 memory

## 配置文件

### 实验配置
```yaml
# configs/assignments/.../reflection_grpo_k1_4_k2_4.yaml
callback_2:  # Memory + Reflection 存储
  name: previous_sample_embedding_callback
  custom_parameters:
    utilized_sample_count: 4
    enable_reflection: true  # 自动为 greedy trajectory 生成 reflection
    reflection_use_local_model: true
    reflection_local_do_sample: false  # 贪婪解码

callback_grpo_base:  # Base Model GRPO (k2 rollouts)
  name: grpo_training_callback_rllm
  custom_parameters:
    config_path: "./configs/components/rl/base_model_grpo.yaml"

callback_reflection_grpo:  # Reflection Model GRPO (k1 reflections)
  name: reflection_grpo_training_callback
  custom_parameters:
    config_path: "./configs/components/rl/reflection_grpo.yaml"
```

### Base Model GRPO 配置
```yaml
# configs/components/rl/base_model_grpo.yaml
model_name_or_path: /mnt/ssd2/models/Meta-Llama-3.1-8B-Instruct
group_size: 4  # k2 = 4 (每个 reflection 对应 4 个 rollouts)
accumulate_samples: 2
grpo:
  beta: 0.04  # KL 惩罚系数
  clip_param: 0.2
  clip_ratio_c: 3.0
lora:
  r: 16
  alpha: 32
  dropout: 0.05
optim:
  learning_rate: 2e-5
  max_grad_norm: 1.0
save:
  lora_output_dir: outputs/{TIMESTAMP}/base_model_lora
```

### Reflection GRPO 配置
```yaml
# configs/components/rl/reflection_grpo.yaml
reflection_model_path: /mnt/ssd2/models/Meta-Llama-3.1-8B-Instruct
k1: 4  # 每个样本生成 4 个 reflections
k2: 4  # 每个 reflection 对应 4 个 rollouts
accumulate_samples: 2
warmup_samples: 50
store_rollout_success_to_memory: true  # 存储成功轨迹到 memory
grpo:
  beta: 0.0  # 不使用 KL loss (节省显存)
  clip_param: 0.2
  clip_ratio_c: 3.0
lora:
  r: 16
  alpha: 32
  dropout: 0.05
optim:
  learning_rate: 2e-5
save:
  lora_output_dir: outputs/{TIMESTAMP}/reflection_lora
```

## 运行实验

```bash
# 运行 Reflection GRPO 训练
python src/run_experiment.py \
  --config_path configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/reflection_grpo_k1_4_k2_4.yaml \
  --max_samples 100

# 查看训练日志
tail -f outputs/reflection_grpo_k1_4_k2_4/*/callback_state/callback_reflection_grpo/reflection_train_log.tsv

# 查看 TensorBoard
tensorboard --logdir outputs/reflection_grpo_k1_4_k2_4/*/tensorboard_reflection
```

## 输出文件

```
outputs/reflection_grpo_k1_4_k2_4/{TIMESTAMP}/
├── config.yaml                          # 实验配置
├── runs.json                           # 所有 session 记录
├── metric.json                         # 评估指标
├── callback_state/
│   ├── callback_2/
│   │   └── utilized_session_list.json  # Memory 数据库
│   ├── callback_grpo_base/
│   │   ├── grpo_state.json            # Base model 训练状态
│   │   └── train_log.tsv              # Base model 训练日志
│   └── callback_reflection_grpo/
│       ├── reflection_state.json       # Reflection model 训练状态
│       └── reflection_train_log.tsv    # Reflection model 训练日志
├── base_model_lora/                    # Base model LoRA 权重
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── reflection_lora/                    # Reflection model LoRA 权重
    ├── adapter_config.json
    └── adapter_model.safetensors
```

## 关键优势

1. **嵌套 GRPO**: 同时训练 reflection model 和 base model
2. **两种 Reflection**:
   - Memory reflection（确定性，长期存储）
   - GRPO training reflection（多样性，探索优化）
3. **Chain-of-Thought 分析**: 结构化的轨迹反思，提炼通用规则
4. **成功轨迹存储**: 确保 memory 包含成功案例
5. **Virtual Sample Index**: 确保 GRPO 分组正确
6. **完全对齐 verl**: 使用 verl 的核心算法计算 loss

## 技术细节

### Virtual Sample Index 机制
为了让 `grpo_training_callback_rllm` 正确分组，使用：
```python
virtual_sample_index = f"{sample_index}_r{reflection_id}"
```
这样，每个 reflection 的 k2 个 rollouts 会被视为独立的 group。

### Reflection 注入位置
```
[Historical Trajectories]  # 从 memory 召回
  - Question 1: ... Trajectory: ... Insight: ...
  - Question 2: ... Trajectory: ... Insight: ...
  - ...

[Task-Specific Hint]       # 从 reflection model 生成（仅 GRPO rollouts）
**Task-Specific Hint:**
{insight_text}

[Task Description]         # 原始 prompt
```

### 显存优化
- `beta: 0.0`: 不加载 ref model，节省显存
- LoRA: 只训练少量参数
- `niuload`: 均匀分配多 GPU 显存

## 故障排查

### 常见问题

**Q: Reflection 没有注入到 prompt？**
A: 检查 session 是否有 `_reflection_text` 属性，并查看 `on_task_reset` 日志。

**Q: GRPO 训练不收敛？**
A: 检查 reward 方差是否太小（`reward_std < epsilon` 会被丢弃）。

**Q: Memory 没有存储 reflection？**
A: 确认 `enable_reflection: true` 并检查 `on_state_save` 日志。

**Q: Virtual sample index 导致分组错误？**
A: 确认使用 `f"{sample_index}_r{reflection_id}"` 格式。

## 未来改进方向

1. **自适应 k1/k2**: 根据任务难度动态调整 reflection 数量
2. **Reflection Quality 评估**: 训练一个 reflection quality 模型
3. **多模态 Reflection**: 支持图表、代码等多模态输入
4. **增量学习**: 在线更新 reflection model
5. **Reflection Reuse**: 复用高质量 reflections
