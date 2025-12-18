import argparse
import json
import math
import os
import sys
import yaml
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure local rLLM package is importable when not installed system-wide
try:
    from rllm.agents.utils import convert_messages_to_tokens_and_masks
    from rllm.parser import ChatTemplateParser
except ModuleNotFoundError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = [
        repo_root,
        os.path.join(repo_root, "rllm"),
        os.path.abspath(os.path.join(repo_root, "..", "rllm")),
    ]

    def _inject(path: str) -> bool:
        if os.path.exists(os.path.join(path, "rllm", "__init__.py")):
            if path not in sys.path:
                sys.path.insert(0, path)
            return True
        return False

    for cand in candidates:
        if _inject(cand):
            break

    from rllm.agents.utils import convert_messages_to_tokens_and_masks
    from rllm.parser import ChatTemplateParser

from src.rllm_integration.dbbench_agent import DBBenchAgent, load_prelude_messages
from src.rllm_integration.dbbench_dataset import load_dbbench_standard
from src.rllm_integration.dbbench_env import DBBenchEnv


@dataclass
class Attempt:
    messages: List[Dict[str, str]]
    action_mask: torch.Tensor  # bool [1, L]
    input_ids: torch.Tensor  # [1, L]
    attention_mask: torch.Tensor  # [1, L]
    gen_logprobs: torch.Tensor  # [num_action_tokens]
    sampling_logprobs: torch.Tensor  # [num_action_tokens] - logprobs from sampling
    reward: float
    correct: bool
    completed: bool


@dataclass
class SuccessfulTrajectory:
    """存储成功的轨迹用于历史召回"""
    messages: List[Dict[str, str]]
    question: str


def load_grpo_config(config_path: str) -> Dict[str, Any]:
    """加载GRPO配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def build_model(model_path: str, lora_config: Dict[str, Any]) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    lora_cfg = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    return model, tokenizer


def build_reference_model(ref_path: Optional[str], dtype, device_map="auto"):
    if ref_path is None:
        return None
    ref_model = AutoModelForCausalLM.from_pretrained(ref_path, device_map=device_map, torch_dtype=dtype)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    return ref_model


def generate(model, tokenizer, messages: List[Dict[str, str]], gen_kwargs: Dict[str, Any]) -> str:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.strip()


def token_logprobs(model, input_ids, attention_mask, enable_grad: bool = False):
    ctx = torch.enable_grad if enable_grad else torch.no_grad
    with ctx():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        log_probs = logits.log_softmax(dim=-1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs


def calc_reward(correct: bool, completed: bool) -> float:
    """
    纯正向奖励函数：
    - 结果正确：+1.0
    - 状态是COMPLETED：+0.5
    - 其他情况：0（不扣分）
    """
    reward = 0.0
    if correct:
        reward += 1.0
    if completed:
        reward += 0.5
    return reward


def inject_history_to_prelude(
    prelude_messages: List[Dict[str, str]],
    history_trajectories: List[SuccessfulTrajectory],
) -> List[Dict[str, str]]:
    """将历史轨迹注入到prelude中"""
    if not history_trajectories:
        return prelude_messages

    # 构建历史轨迹文本
    example_text = "\n"
    for traj in history_trajectories:
        example_text += f"Question {traj.question}:\n"
        # 跳过前2条（system和OK），从第3条开始
        for msg in traj.messages[2:]:
            role = msg["role"]
            content = msg["content"]
            example_text += f"{role}: {content}\n"

    # 替换prelude中的占位符
    new_prelude = []
    for msg in prelude_messages:
        if "{previous_sample_utilization_target_position}" in msg["content"]:
            new_content = msg["content"].replace(
                "{previous_sample_utilization_target_position}",
                example_text
            )
            new_prelude.append({"role": msg["role"], "content": new_content})
        else:
            new_prelude.append(msg)

    return new_prelude


def rollout(
    env: DBBenchEnv,
    agent: DBBenchAgent,
    model,
    tokenizer,
    gen_kwargs: Dict[str, Any],
    record_sampling_logprobs: bool = False,
) -> tuple[Attempt, Dict[str, Any]]:
    parser = ChatTemplateParser.get_parser(tokenizer)
    observation, _ = env.reset()
    agent.reset()
    agent.update_from_env(observation, 0.0, False, {})
    done = False
    reward_final = 0.0
    completed = False

    while not done:
        response = generate(model, tokenizer, agent.chat_completions, gen_kwargs)
        action = agent.update_from_model(response).action
        observation, step_reward, done, info = env.step(action)
        reward_final = step_reward
        if done:
            completed = info.get("finish_reason") not in ["MAX_TURNS", None]
            break
        agent.update_from_env(observation, step_reward, done, info)

    if done and observation:
        agent.update_from_env(observation, reward_final, done, {})

    messages = agent.chat_completions
    token_list, mask_list = convert_messages_to_tokens_and_masks(
        messages,
        tokenizer=tokenizer,
        parser=parser,
        contains_first_msg=True,
        contains_generation_msg=True,
    )
    input_ids = torch.tensor(token_list, dtype=torch.long, device=model.device).unsqueeze(0)
    action_mask = torch.tensor(mask_list, dtype=torch.bool, device=model.device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    # 计算生成时的logprobs
    gen_logps_full = token_logprobs(model, input_ids, attention_mask, enable_grad=False)
    action_mask_shift = action_mask[:, 1:]
    gen_logps = gen_logps_full[action_mask_shift].detach()

    # 如果需要记录sampling logprobs（用于训练）
    sampling_logps = gen_logps if record_sampling_logprobs else gen_logps

    correct = getattr(env, "correct", False)
    reward = calc_reward(correct, completed)

    return Attempt(
        messages, action_mask, input_ids, attention_mask,
        gen_logps, sampling_logps, reward, correct, completed
    ), {"correct": correct, "reward": reward, "completed": completed}


def grpo_update(
    model,
    optimizer,
    attempts: List[Attempt],
    config: Dict[str, Any],
    ref_model=None,
) -> Optional[Dict[str, float]]:
    if len(attempts) == 0:
        return None

    device = model.device
    grpo_config = config.get("grpo", {})
    optim_config = config.get("optim", {})

    beta = float(grpo_config.get("beta", 0.04))
    clip_param = float(grpo_config.get("clip_param", 0.2))
    normalize = bool(grpo_config.get("normalize_rewards", False))
    use_best_of_n = bool(grpo_config.get("use_best_of_n", False))
    grad_accum = int(optim_config.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(optim_config.get("max_grad_norm", 1.0))
    num_epochs = int(optim_config.get("num_train_epochs", 1))

    raw_rewards = torch.tensor([a.reward for a in attempts], device=device, dtype=torch.float32)

    # Best-of-N选择机制
    if use_best_of_n and len(attempts) > 1:
        best_idx = raw_rewards.argmax().item()
        worst_idx = raw_rewards.argmin().item()
        if raw_rewards[best_idx] > raw_rewards[worst_idx]:
            train_attempts = [attempts[best_idx], attempts[worst_idx]]
            train_rewards = torch.tensor([raw_rewards[best_idx], raw_rewards[worst_idx]], device=device)
        else:
            # 即使全部相同也训练，使用全部样本
            train_attempts = attempts
            train_rewards = raw_rewards
    else:
        # 使用全部样本训练（即使全正确或全错误）
        train_attempts = attempts
        train_rewards = raw_rewards

    # 奖励归一化
    if normalize and len(train_rewards) > 1:
        rewards = (train_rewards - train_rewards.mean()) / (train_rewards.std() + 1e-6)
    else:
        rewards = train_rewards

    model.train()
    loss_sum = 0.0
    policy_sum = 0.0
    kl_sum = 0.0
    loss_steps = 0

    for _ in range(num_epochs):
        backward_calls = 0
        for idx, attempt in enumerate(train_attempts):
            action_mask = attempt.action_mask
            if action_mask.sum().item() == 0:
                continue

            # 计算当前策略的logprobs
            token_logps_full = token_logprobs(model, attempt.input_ids, attempt.attention_mask, enable_grad=True)
            action_mask_shift = action_mask[:, 1:]
            new_logps = token_logps_full[action_mask_shift]

            # 使用采样时的logprobs计算PPO ratio
            old_logps = attempt.sampling_logprobs.to(device)

            # 计算参考模型的logprobs（如果有）
            if ref_model is not None:
                ref_logps_full = token_logprobs(ref_model, attempt.input_ids, attempt.attention_mask, enable_grad=False)
                ref_logps = ref_logps_full[action_mask_shift]
            else:
                ref_logps = attempt.gen_logprobs.to(device)

            advantage = rewards[idx]
            ratio = torch.exp(new_logps - old_logps)
            clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
            policy_component = torch.min(ratio * advantage, clipped_ratio * advantage)

            # KL散度惩罚
            per_token_kl = new_logps - ref_logps
            policy_loss = -policy_component.mean()
            kl_loss = beta * per_token_kl.mean()
            loss = (policy_loss + kl_loss) / grad_accum

            loss.backward()
            loss_sum += float(loss.item())
            policy_sum += float(policy_loss.item())
            kl_sum += float(kl_loss.item())
            loss_steps += 1
            backward_calls += 1

            if (idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        if backward_calls % grad_accum != 0 and backward_calls > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

    if loss_steps > 0:
        return {
            "loss": loss_sum / loss_steps,
            "policy_loss": policy_sum / loss_steps,
            "kl_loss": kl_sum / loss_steps,
        }
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="HF本地模型路径")
    parser.add_argument("--config_path", required=True, help="GRPO配置文件路径（如configs/components/rl/db_bench_grpo.yaml）")
    parser.add_argument("--prelude_path", type=str, default="chat_history_items/previous_sample_utilization/db_bench.json", help="开场提示文件路径")
    parser.add_argument("--data_path", type=str, default="data/db_bench.json", help="数据集路径")
    parser.add_argument("--max_samples", type=int, default=None, help="限制运行的样本数量")
    parser.add_argument("--save_dir", type=str, default=None, help="输出根目录（默认outputs/{TIMESTAMP}）")
    parser.add_argument("--log_path", type=str, default=None, help="逐样本轨迹/损失日志文件")
    parser.add_argument("--metrics_path", type=str, default=None, help="整体指标输出文件")
    parser.add_argument("--utilized_sample_count", type=int, default=4, help="使用的历史轨迹数量")
    args = parser.parse_args()

    # 加载GRPO配置
    grpo_config = load_grpo_config(args.config_path)

    # 提取配置参数
    group_size = grpo_config.get("group_size", 4)
    gen_config = grpo_config.get("generation", {})
    lora_config = grpo_config.get("lora", {})
    optim_config = grpo_config.get("optim", {})
    ref_model_path = grpo_config.get("grpo", {}).get("reference_model_path")

    # 构建输出目录
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_out = args.save_dir or os.path.join("outputs", timestamp)
    os.makedirs(base_out, exist_ok=True)
    log_path = args.log_path or os.path.join(base_out, "dbbench_grpo_history_log.jsonl")
    metrics_path = args.metrics_path or os.path.join(base_out, "dbbench_grpo_history_metrics.json")
    lora_out = os.path.join(base_out, "lora")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    # 构建模型
    model, tokenizer = build_model(args.model_path, lora_config)
    ref_model = build_reference_model(ref_model_path, dtype=model.dtype, device_map="auto")
    optimizer = AdamW(
        model.parameters(),
        lr=optim_config.get("learning_rate", 2e-5),
        weight_decay=optim_config.get("weight_decay", 0.01)
    )

    # 加载数据集和prelude
    dataset = load_dbbench_standard(data_path=args.data_path)
    base_prelude = load_prelude_messages(path=args.prelude_path)

    # 维护历史成功轨迹列表
    history_trajectories: List[SuccessfulTrajectory] = []

    greedy_correct = 0
    total = 0
    if os.path.exists(log_path):
        os.remove(log_path)

    samples = dataset.get_data()
    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    for sample in samples:
        total += 1
        sample_index = sample.get("sample_index", total)

        # 注入历史轨迹
        prelude_with_history = inject_history_to_prelude(base_prelude, history_trajectories)

        # ----- Greedy evaluation pass -----
        env = DBBenchEnv(sample, max_turns=3)
        agent = DBBenchAgent(prelude_with_history)
        greedy_kwargs = {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": gen_config.get("max_new_tokens", 512)
        }
        greedy_attempt, greedy_info = rollout(env, agent, model, tokenizer, greedy_kwargs, record_sampling_logprobs=False)
        if greedy_info.get("correct"):
            greedy_correct += 1
        print(f"[Sample {sample_index}] Greedy correct={greedy_info.get('correct')} reward={greedy_info.get('reward'):.2f}")

        # ----- GRPO sampling passes -----
        attempts: List[Attempt] = []
        sample_kwargs = {
            "do_sample": gen_config.get("do_sample", True),
            "temperature": gen_config.get("temperature", 0.8),
            "top_p": gen_config.get("top_p", 0.95),
            "max_new_tokens": gen_config.get("max_new_tokens", 512),
        }

        for attempt_id in range(group_size):
            env_t = DBBenchEnv(sample, max_turns=3)
            agent_t = DBBenchAgent(prelude_with_history)
            attempt, attempt_info = rollout(env_t, agent_t, model, tokenizer, sample_kwargs, record_sampling_logprobs=True)
            attempts.append(attempt)
            print(f"[Sample {sample_index}] Attempt {attempt_id + 1}/{group_size}: reward={attempt.reward:.2f}, correct={attempt.correct}, completed={attempt.completed}")

        print(f"[Sample {sample_index}] Collected {len(attempts)} attempts, rewards={[a.reward for a in attempts]}")

        # ----- GRPO update -----
        update_metrics = grpo_update(
            model=model,
            optimizer=optimizer,
            attempts=attempts,
            config=grpo_config,
            ref_model=ref_model,
        )
        print(f"[Sample {sample_index}] GRPO update done.")

        # 更新历史轨迹列表（只保存正确且完成的轨迹）
        best_attempt = max(attempts, key=lambda a: a.reward)
        if best_attempt.correct and best_attempt.completed:
            history_trajectories.append(SuccessfulTrajectory(
                messages=best_attempt.messages,
                question=sample.get("instruction", "")
            ))
            if len(history_trajectories) > args.utilized_sample_count:
                history_trajectories.pop(0)

        # 记录日志（包含完整轨迹）
        sample_log = {
            "sample_index": sample_index,
            "instruction": sample.get("instruction", ""),
            # Greedy解码信息
            "greedy_correct": bool(greedy_info.get("correct")),
            "greedy_reward": float(greedy_info.get("reward", 0.0)),
            "greedy_completed": bool(greedy_info.get("completed", False)),
            "greedy_trajectory": greedy_attempt.messages,  # 完整的贪心解码轨迹
            # GRPO采样信息
            "attempt_rewards": [float(a.reward) for a in attempts],
            "attempt_corrects": [bool(a.correct) for a in attempts],
            "attempt_completed": [bool(a.completed) for a in attempts],
            "attempt_trajectories": [a.messages for a in attempts],  # 所有采样轨迹
            # 训练信息
            "loss": update_metrics.get("loss") if update_metrics else None,
            "policy_loss": update_metrics.get("policy_loss") if update_metrics else None,
            "kl_loss": update_metrics.get("kl_loss") if update_metrics else None,
            "history_count": len(history_trajectories),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample_log, ensure_ascii=False) + "\n")

    acc = greedy_correct / max(1, total)
    print(f"Greedy accuracy: {greedy_correct}/{total} = {acc:.4f}")

    # 保存模型
    model.save_pretrained(lora_out)
    tokenizer.save_pretrained(lora_out)

    # 保存指标
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"greedy_accuracy": acc, "total": total, "correct": greedy_correct}, f, indent=2)


if __name__ == "__main__":
    main()
