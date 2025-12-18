# VERL 框架中批处理概念澄清

本文档基于 VERL/RLLM 框架源代码分析，澄清数据批处理、GRPO Group、Mini-batch、Micro-batch 等关键概念的关系。

---

## 一、核心概念定义

### 1. `data.train_batch_size` - 训练数据批大小

**定义**: 单个训练步骤从数据集中抽取的唯一样本（prompt）数量。

**作用范围**: 直接影响数据加载器（DataLoader）

**数值范围**: 通常为 256 到 1024

**配置位置**: `_generated_agent_ppo_trainer.yaml` 第 224 行

```yaml
data:
  train_batch_size: 1024  # 从数据集抽取1024条不同的prompt
```

**关键特性**:
- 来自数据加载器的是 **不同的 prompt**
- 每个 prompt 只有一份，用来对应多个生成结果（via `actor_rollout_ref.rollout.n`）
- 这是整个训练 epoch 的数据单位

---

### 2. `actor_rollout_ref.rollout.n` - Rollout 采样数 (GRPO 的 group_size)

**定义**: 对每个 prompt 进行多少次独立采样/生成，每组采样后才进行一次模型更新。

**GRPO 术语**: 这就是 GRPO 论文中的 **`group_size`** 或 **`N`** (同一 prompt 的多个轨迹)

**典型值**: 4（标准 GRPO 配置）

**配置位置**: `_generated_agent_ppo_trainer.yaml` 第 95 行和第 250 行

```yaml
actor_rollout_ref:
  rollout:
    'n': 4  # 对每个prompt生成4个不同的回答

critic:
  rollout_n: ${oc.select:actor_rollout_ref.rollout.n,1}
```

**关键特性**:
- 在 GRPO 中，这 4 个响应组成一个 **"Group"**
- 4 个响应具有**不同的奖励值**，这是 GRPO "Group Relative" 的基础
- 使用数据重复机制实现: `batch.repeat(repeat_times=n, interleave=True)`

**工作流程**（来自代码第 120-123 行）:

```python
# 原始批：train_batch_size个样本
batch = DataProto.from_single_dict(batch_dict)  # shape: (1024,)

# 扩展批：1024 * 4 = 4096个总轨迹
batch = batch.repeat(
    repeat_times=self.config.actor_rollout_ref.rollout.n,  # n=4
    interleave=True,  # 交错排列保证同一prompt相邻
)  # shape: (4096,)

# 采样组织: 每4条轨迹来自同一prompt
# prompt_0: [traj_0, traj_1, traj_2, traj_3]
# prompt_1: [traj_4, traj_5, traj_6, traj_7]
# ...
# prompt_1023: [traj_4092, traj_4093, traj_4094, traj_4095]
```

---

### 3. `ppo_mini_batch_size` - PPO Mini-Batch 大小

**定义**: 在一个训练步骤内，PPO 优化器分多少个小批处理不同的"独立样本"进行梯度累积和更新。

**单位**: **独立样本（prompt）的数量**，不是轨迹数

**典型值**: 256

**配置位置**: `_generated_agent_ppo_trainer.yaml` 第 4 行和 281 行

```yaml
actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 256  # 每个mini-batch处理256个不同prompt的数据

critic:
  ppo_mini_batch_size: ${oc.select:actor_rollout_ref.actor.ppo_mini_batch_size,256}
```

**约束关系**: 必须满足整除关系

```python
# 来自代码第 157 行
assert ppo_train_batch_size % ppo_mini_batch_size == 0
```

**如果 train_batch_size=1024, ppo_mini_batch_size=256**:
- 需要 1024 / 256 = **4 次 mini-batch 迭代**来处理这 1024 个 prompt

---

### 4. `ppo_micro_batch_size_per_gpu` - Micro-Batch 大小（每 GPU）

**定义**: 在单个 GPU 上同时前向传播的轨迹数量，与显存配置相关。

**单位**: **轨迹（trajectory）**，通常与模型推理流程相关

**是否必需**: 否（可为 null，由框架自动计算）

**配置位置**: `_generated_agent_ppo_trainer.yaml` 第 6 行、60 行、90 行

```yaml
actor_rollout_ref:
  actor:
    ppo_micro_batch_size: null  # 可不指定
    ppo_micro_batch_size_per_gpu: null  # 可不指定

  ref:
    log_prob_micro_batch_size_per_gpu: null

  rollout:
    log_prob_micro_batch_size_per_gpu: null
```

**自动计算逻辑**: 如果为 null，框架基于：
- 可用 GPU 显存
- `ppo_max_token_len_per_gpu` (第 8 行: 16384 tokens)
- 分布式并行大小

**不同于 mini-batch 的原因**:
- Mini-batch = 逻辑优化单位（促进梯度稳定性）
- Micro-batch = 物理内存单位（受硬件限制）

---

## 二、批处理层级关系图

```
数据层 (Data Layer)
├─ 数据集 (Dataset): N 条不同的 prompt
│
└─ train_batch_size = 1024
   └─ 从数据集一次性抽取 1024 个不同 prompt

生成层 (Generation Layer - Rollout)
├─ actor_rollout_ref.rollout.n = 4
│  └─ 对每个 prompt 采样 4 次
│     (这 4 个响应 = GRPO 的 "Group")
│
└─ 总轨迹数 = 1024 × 4 = 4096

训练层 (Training Layer - Actor)
├─ ppo_mini_batch_size = 256
│  └─ 将 1024 个 prompt 分成 4 个 mini-batch
│     每个 mini-batch 处理 256 个 prompt 的数据
│
├─ 每个 mini-batch 的轨迹数:
│  └─ 256 prompt × 4 responses = 1024 轨迹
│
└─ ppo_micro_batch_size_per_gpu ≈ ?
   └─ 单 GPU 上的实际批大小（由框架自动管理）
      与 PPO 计算的分布式划分相关
```

---

## 三、GRPO 中 "Group" 的精确定义

在 `/home/marble/实验代码版本控制/configs/components/rl/db_bench_grpo.yaml`:

```yaml
group_size: 4  # GRPO中同一prompt的采样数量
```

在 `/home/marble/实验代码版本控制/src/callbacks/instance/grpo_training_callback_rllm.py` 的代码中：

```python
class GRPOTrainingCallbackRLLM(Callback):
    def __init__(self, config_path: str):
        # ...
        self.group_size: int = int(self.config.get("group_size", 1))  # 第77行

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        # ...
        if len(self.pending_attempts[key]) >= self.group_size:
            self._train_on_group(key, callback_args)  # 第291-292行

    def _train_on_group_local(self, sample_index, attempts, callback_args):
        # 4个attempt（轨迹）组成一个group，进行GRPO训练
        raw_rewards = torch.tensor(
            [a.reward for a in attempts],  # [reward_0, reward_1, reward_2, reward_3]
            device=device, dtype=torch.float32
        )  # 第646-648行

        # GRPO核心：group内相对奖励归一化（不使用GAE）
        if normalize and len(train_rewards) > 1:
            advantages = (train_rewards - train_rewards.mean()) / train_rewards.std()
```

**Group 的计算流程**:

```python
# 4个同prompt的响应
attempts = [attempt_0, attempt_1, attempt_2, attempt_3]
rewards = [0.0, 1.0, 1.5, 1.0]

# GRPO核心：组内相对归一化而非GAE
mean_reward = mean(rewards) = 0.875
std_reward = std(rewards) ≈ 0.535
advantages = (rewards - mean_reward) / std_reward
           = [-0.163, 0.023, 1.177, 0.023]

# 4个advantage值用于PPO更新
```

---

## 四、Mini-Batch 与 Micro-Batch 的对应关系

### 在 VERL 训练循环中的实际执行（来自代码第 115-249 行）

```python
for epoch in range(total_epochs):
    for batch_iter, batch_dict in enumerate(train_dataloader):
        # 1. 数据加载
        batch = DataProto.from_single_dict(batch_dict)  # 1024个prompt

        # 2. 生成扩展（每个prompt 4次采样）
        batch = batch.repeat(repeat_times=4, interleave=True)  # 4096个轨迹

        # 3. 触发生成（vLLM/sglang 负责 micro-batch 调度）
        for trajectory in generate_agent_trajectories():
            # vLLM 在此自动进行 micro-batch 调度
            pass

        # 4. PPO 训练循环：Mini-batch 迭代
        ppo_mini_batch_size = 256
        num_loops = 1024 // 256  # = 4

        for mini_batch_iter in range(num_loops):  # 第162-249行
            # 每个mini-batch收集256个prompt及其4个responses
            trajectories = []
            for _ in range(ppo_mini_batch_size):  # 256次
                _, trajes = replay_queue.get()  # 获取一个prompt的4个轨迹
                trajectories.extend(trajes)  # 添加到列表

            # 现在有 256 × 4 = 1024 条轨迹
            mini_batch = transform_trajectories(trajectories)  # 1024轨迹

            # 5. 在 mini_batch 上进行 PPO 更新
            actor_output = actor_wg.update_actor_mini_batch(mini_batch)
            # 这个update内部可能继续分 micro-batch 处理
```

**层级关系总结表**:

| 层级 | 单位 | 数量 | 配置参数 | 用途 |
|------|------|------|---------|------|
| Data | Prompt | 1024 | `data.train_batch_size` | 数据加载 |
| Rollout | Response | 4096 (1024×4) | `actor_rollout_ref.rollout.n` | 生成采样 |
| Group | Same-prompt responses | 4 | `group_size` (GRPO) | 相对奖励计算 |
| Mini-batch | Prompts | 256 | `ppo_mini_batch_size` | PPO梯度累积 |
| Mini-batch total | Trajectories | 1024 (256×4) | - | 单次参数更新 |
| Micro-batch | Trajectories/GPU | ≤1024 | `ppo_micro_batch_size_per_gpu` | GPU内存管理 |

---

## 五、VERL 批处理在不同模块中的应用

### 5.1 Rollout（生成）模块

使用 `actor_rollout_ref.rollout.n` 进行数据重复：

**文件**: `rllm/rllm/trainer/verl/agent_ppo_trainer_pipeline.py` 第 120-123 行

```python
batch = batch.repeat(
    repeat_times=self.config.actor_rollout_ref.rollout.n,  # n=4
    interleave=True,  # 关键：交错排列使同一prompt相邻
)
```

**为什么要 interleave=True**:
- 保证生成器返回轨迹时，相同 prompt 的轨迹被分组
- 便于后续构建 GRPO group

**代码第 136-146 行展示了分组机制**:

```python
uid_to_trajectories = {}  # 存储按 uid 分组的轨迹

for trajectory in generator:  # 按返回顺序遍历
    uid = trajectory["idx"] // n  # 计算原始prompt的id

    if uid not in uid_to_trajectories:
        uid_to_trajectories[uid] = []

    uid_to_trajectories[uid].append(trajectory)

    # 当集合了 n 个同 uid 的轨迹，放入队列
    if len(uid_to_trajectories[uid]) == n:
        q.put((batch_iter_val, uid_to_trajectories[uid]))
        del uid_to_trajectories[uid]
```

### 5.2 Actor（优化）模块

**Mini-batch 循环** (`agent_ppo_trainer_pipeline.py` 第 162-249 行):

```python
ppo_step_minibatch_iter = train_batch_size // ppo_mini_batch_size
# = 1024 // 256 = 4

for mini_batch_iter in range(ppo_step_minibatch_iter):
    # 第168-171行：从队列获取 ppo_mini_batch_size 个 group
    trajectories = []
    for _ in range(ppo_mini_batch_size):  # 256次
        _, trajes = replay_queue.get()  # 得到1个prompt的[traj_0, traj_1, traj_2, traj_3]
        trajectories.extend(trajes)  # 添加4个轨迹

    # 现在 trajectories 包含 256×4=1024 条轨迹
    mini_batch = self._transform_agent_trajectories(trajectories)

    # 计算优势和损失
    mini_batch = compute_advantage(...)

    # PPO更新（内部可能使用micro-batch）
    actor_output = self.actor_wg.update_actor_mini_batch(mini_batch)
```

### 5.3 Ref Model（参考模型）模块

使用相同的 micro-batch 配置：

**文件**: `_generated_agent_ppo_trainer.yaml` 第 57-70 行

```yaml
ref:
  strategy: ${actor_rollout_ref.actor.strategy}
  log_prob_micro_batch_size: null
  log_prob_micro_batch_size_per_gpu: null
  log_prob_use_dynamic_bsz: ${oc.select:actor_rollout_ref.actor.use_dynamic_bsz,false}
  log_prob_max_token_len_per_gpu: ${oc.select:actor_rollout_ref.actor.ppo_max_token_len_per_gpu,16384}
```

---

## 六、实际配置示例解读

从 `db_bench_grpo_rllm.yaml`:

```yaml
group_size: 4                      # GRPO group_size = 4

generation:
  do_sample: true
  temperature: 0.8
  max_new_tokens: 512              # 每条轨迹最多512个新tokens

lora:
  r: 64
  alpha: 128

optim:
  learning_rate: 2.0e-5
  num_train_epochs: 1
  gradient_accumulation_steps: 1    # 无梯度累积（每step更新）
```

配合 `_generated_agent_ppo_trainer.yaml`:

```yaml
data:
  train_batch_size: 1024            # 数据批大小

actor_rollout_ref:
  rollout:
    'n': 1                          # 仅采样1次(或可改为4)
    max_num_seqs: 1024              # vLLM最多1024个并发序列
    max_num_batched_tokens: 8192    # vLLM最多8192个tokens/batch

  actor:
    ppo_mini_batch_size: 256        # mini-batch大小
    ppo_micro_batch_size_per_gpu: null  # 自动
```

---

## 七、性能调优建议

### 显存不足时

**问题**: "CUDA out of memory"

**调整策略**:

1. **减小 ppo_mini_batch_size**
   ```yaml
   ppo_mini_batch_size: 128  # 从256→128，显存减半
   ```

2. **减小 actor_rollout_ref.rollout.n**
   ```yaml
   'n': 2  # 从4→2，总轨迹减半
   ```

3. **减小 max_num_seqs**
   ```yaml
   max_num_seqs: 512  # 从1024→512
   ```

### 吞吐量不足时

**问题**: GPU 利用率低，收敛慢

**调整策略**:

1. **增加 ppo_mini_batch_size**（若显存允许）
   ```yaml
   ppo_mini_batch_size: 512  # 从256→512
   ```

2. **增加 data.train_batch_size**
   ```yaml
   train_batch_size: 2048  # 从1024→2048
   ```

3. **启用梯度累积**
   ```yaml
   gradient_accumulation_steps: 2  # 2步后更新
   ```

---

## 八、本地 GRPO 实现参考

在 `src/callbacks/instance/grpo_training_callback_rllm.py`:

```python
class GRPOTrainingCallbackRLLM(Callback):
    def __init__(self, config_path: str):
        self.group_size = 4  # 从YAML读取
        self.pending_attempts = {}  # sample_index -> [attempt_0, ..., attempt_3]

    def on_task_complete(self, session):
        reward = self._calc_reward(session)
        attempt = self._build_attempt_record(session, reward)

        key = session.sample_index
        if key not in self.pending_attempts:
            self.pending_attempts[key] = []
        self.pending_attempts[key].append(attempt)  # 第289行

        # 当集齐group_size个尝试
        if len(self.pending_attempts[key]) >= self.group_size:
            self._train_on_group(key)  # GRPO训练
            self.pending_attempts[key] = []
```

---

## 九、总结表

| 概念 | 单位 | 默认值 | 作用 | 配置位置 |
|------|------|--------|------|---------|
| `data.train_batch_size` | 不同 prompt 数 | 1024 | 数据加载 | `_generated_agent_ppo_trainer.yaml` L224 |
| `actor_rollout_ref.rollout.n` | 同prompt采样数 | 1 | 生成采样/GRPO group | `_generated_agent_ppo_trainer.yaml` L95 |
| `group_size` | 同prompt响应数 | 4 | GRPO相对奖励 | `db_bench_grpo_rllm.yaml` L1 |
| `ppo_mini_batch_size` | prompt数 | 256 | PPO优化单位 | `_generated_agent_ppo_trainer.yaml` L4 |
| `ppo_micro_batch_size_per_gpu` | 轨迹数 | auto | GPU内存管理 | `_generated_agent_ppo_trainer.yaml` L6 |

---

## 十、参考源代码位置

1. **批处理重复机制**: `/rllm/rllm/trainer/verl/agent_ppo_trainer_pipeline.py` L120-123
2. **Group分组机制**: `/rllm/rllm/trainer/verl/agent_ppo_trainer_pipeline.py` L136-146
3. **Mini-batch循环**: `/rllm/rllm/trainer/verl/agent_ppo_trainer_pipeline.py` L155-249
4. **GRPO本地实现**: `/src/callbacks/instance/grpo_training_callback_rllm.py` L77, L289-292, L630-821
5. **GRPO配置**: `/configs/components/rl/db_bench_grpo_rllm.yaml`
6. **VERL官方配置**: `/rllm/rllm/trainer/config/_generated_agent_ppo_trainer.yaml`

