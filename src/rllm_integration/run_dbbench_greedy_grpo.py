import argparse
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
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
    gen_logprobs: torch.Tensor  # [num_action_tokens]
    reward: float


def build_model(model_path: str, lora_r: int, lora_alpha: int, lora_dropout: float) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return decoded.strip()


def rollout(env: DBBenchEnv, agent: DBBenchAgent, model, tokenizer, gen_kwargs: Dict[str, Any]) -> tuple[Attempt, Dict[str, Any]]:
    parser = ChatTemplateParser.get_parser(tokenizer)
    observation, _ = env.reset()
    agent.reset()
    agent.update_from_env(observation, 0.0, False, {})
    done = False
    reward = 0.0
    while not done:
        response = generate(model, tokenizer, agent.chat_completions, gen_kwargs)
        action = agent.update_from_model(response).action
        observation, step_reward, done, info = env.step(action)
        reward += step_reward
        if done:
            break
        agent.update_from_env(observation, step_reward, done, info)
    if done and observation:
        agent.update_from_env(observation, reward, done, {})
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
    gen_logps_full = token_logprobs(model, input_ids, attention_mask, enable_grad=False)
    action_mask_shift = action_mask[:, 1:]
    gen_logps = gen_logps_full[action_mask_shift].detach()
    return Attempt(messages, action_mask, input_ids, gen_logps, reward), {"correct": getattr(env, "correct", False), "reward": reward}


def token_logprobs(model, input_ids, attention_mask, enable_grad: bool = False):
    ctx = torch.enable_grad if enable_grad else torch.no_grad
    with ctx():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        log_probs = logits.log_softmax(dim=-1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs


def grpo_update(model, optimizer, attempts: List[Attempt], beta: float, clip_param: float, grad_accum: int, max_grad_norm: float, num_epochs: int, ref_model=None):
    if len(attempts) == 0:
        return None
    raw_rewards = torch.tensor([a.reward for a in attempts], device=model.device, dtype=torch.float32)
    if (raw_rewards > 0).all() or (raw_rewards <= 0).all():
        return None
    rewards = (raw_rewards - raw_rewards.mean()) / (raw_rewards.std() + 1e-6)
    model.train()
    loss_sum = 0.0
    policy_sum = 0.0
    kl_sum = 0.0
    loss_steps = 0
    for _ in range(num_epochs):
        backward_calls = 0
        for idx, attempt in enumerate(attempts):
            action_mask = attempt.action_mask
            if action_mask.sum().item() == 0:
                continue
            attention_mask = torch.ones_like(attempt.input_ids, dtype=torch.long, device=model.device)
            token_logps_full = token_logprobs(model, attempt.input_ids, attention_mask, enable_grad=True)
            action_mask_shift = action_mask[:, 1:]
            new_logps = token_logps_full[action_mask_shift]
            if ref_model is not None:
                ref_logps_full = token_logprobs(ref_model, attempt.input_ids, attention_mask, enable_grad=False)
                old_logps = ref_logps_full[action_mask_shift]
            else:
                old_logps = attempt.gen_logprobs.to(model.device)
            advantage = rewards[idx]
            ratio = torch.exp(new_logps - old_logps)
            clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
            policy_component = torch.min(ratio * advantage, clipped_ratio * advantage)
            per_token_kl = torch.exp(old_logps - new_logps) - (old_logps - new_logps) - 1
            policy_loss = -policy_component.mean()
            kl_loss = (beta * per_token_kl).mean()
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
    parser.add_argument("--model_path", required=True, help="HF 本地模型路径")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--reference_model_path", type=str, default=None, help="可选：冻结参考模型，用于 KL（不挂 LoRA）")
    parser.add_argument("--save_dir", type=str, default=None, help="输出根目录（默认 outputs/{TIMESTAMP}）")
    parser.add_argument("--prelude_path", type=str, default="chat_history_items/standard/db_bench.json", help="开场提示文件路径")
    parser.add_argument("--log_path", type=str, default=None, help="逐样本轨迹/损失日志文件（默认在输出目录下）")
    parser.add_argument("--metrics_path", type=str, default=None, help="整体指标输出文件（默认在输出目录下）")
    args = parser.parse_args()

    # build timestamped output dir
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_out = args.save_dir or os.path.join("outputs", timestamp)
    os.makedirs(base_out, exist_ok=True)
    log_path = args.log_path or os.path.join(base_out, "dbbench_grpo_log.jsonl")
    metrics_path = args.metrics_path or os.path.join(base_out, "dbbench_grpo_metrics.json")
    lora_out = os.path.join(base_out, "lora")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    model, tokenizer = build_model(args.model_path, args.lora_r, args.lora_alpha, args.lora_dropout)
    ref_model = build_reference_model(args.reference_model_path, dtype=model.dtype, device_map="auto")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    dataset = load_dbbench_standard()
    prelude = load_prelude_messages(path=args.prelude_path)

    greedy_correct = 0
    total = 0
    if os.path.exists(log_path):
        os.remove(log_path)
    for sample in dataset.get_data():
        total += 1
        env = DBBenchEnv(sample, max_turns=3)
        agent = DBBenchAgent(prelude)
        greedy_kwargs = {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens}
        _, info = rollout(env, agent, model, tokenizer, greedy_kwargs)
        if info.get("correct"):
            greedy_correct += 1
        print(f"[Sample {sample.get('sample_index', total)}] Greedy correct={info.get('correct')} reward={info.get('reward')}")

        attempts: List[Attempt] = []
        sample_kwargs = {
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        }
        for _ in range(args.group_size):
            env_t = DBBenchEnv(sample, max_turns=3)
            agent_t = DBBenchAgent(prelude)
            attempt, _ = rollout(env_t, agent_t, model, tokenizer, sample_kwargs)
            attempts.append(attempt)
        print(f"[Sample {sample.get('sample_index', total)}] Collected {len(attempts)} attempts, rewards={[a.reward for a in attempts]}")
        update_metrics = grpo_update(
            model=model,
            optimizer=optimizer,
            attempts=attempts,
            beta=args.beta,
            clip_param=args.clip_param,
            grad_accum=args.grad_accum,
            max_grad_norm=args.max_grad_norm,
            num_epochs=args.num_epochs,
            ref_model=ref_model,
        )
        print(f"[Sample {sample.get('sample_index', total)}] GRPO update done.")
        # log sample-level info
        sample_log = {
            "sample_index": sample.get("sample_index", total),
            "greedy_correct": bool(info.get("correct")),
            "greedy_reward": float(info.get("reward", 0.0)),
            "attempt_rewards": [float(a.reward) for a in attempts],
            "loss": update_metrics.get("loss") if update_metrics else None,
            "policy_loss": update_metrics.get("policy_loss") if update_metrics else None,
            "kl_loss": update_metrics.get("kl_loss") if update_metrics else None,
            "trajectory": attempts[0].messages if attempts else [],
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample_log) + "\n")
    acc = greedy_correct / max(1, total)
    print(f"Greedy accuracy: {greedy_correct}/{total} = {acc:.4f}")

    model.save_pretrained(lora_out)
    tokenizer.save_pretrained(lora_out)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"greedy_accuracy": acc, "total": total, "correct": greedy_correct}, f, indent=2)


if __name__ == "__main__":
    main()
