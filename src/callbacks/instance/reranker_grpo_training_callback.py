"""
Reranker GRPO Training Callback - 对齐 verl 实现

训练 reranker 模型，使用对比奖励：
- Baseline (无记忆) 正确 + With Memory 正确 → 小奖励
- Baseline (无记忆) 正确 + With Memory 错误 → 正常惩罚
- Baseline (无记忆) 错误 + With Memory 正确 → 正常奖励
- Baseline (无记忆) 错误 + With Memory 错误 → 小惩罚
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import yaml
import niuload
from peft import LoraConfig, get_peft_model, PeftModel
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

from rllm.agents.utils import convert_messages_to_tokens_and_masks
from rllm.parser import ChatTemplateParser

# verl 核心算法（完全对齐 verl 的 loss 计算）
from verl.trainer.ppo.core_algos import (
    compute_policy_loss,
    kl_penalty,
    agg_loss,
)
import verl.utils.torch_functional as verl_F

from src.callbacks.callback import Callback, CallbackArguments
from src.typings import SampleStatus, Session, SessionEvaluationOutcome


# 全局实例注册（用于跨 callback 通信）
_RERANKER_GRPO_INSTANCE: Optional["RerankerGRPOTrainingCallback"] = None


def get_reranker_grpo_instance() -> Optional["RerankerGRPOTrainingCallback"]:
    """获取全局 RerankerGRPOTrainingCallback 实例"""
    return _RERANKER_GRPO_INSTANCE


@dataclass
class RerankerAttemptRecord:
    """Reranker 的一次尝试记录"""
    input_ids: torch.Tensor  # reranker 完整输入 (prompt + output)
    attention_mask: torch.Tensor
    action_mask: torch.Tensor  # 哪些 token 是 reranker 生成的
    gen_logprobs: torch.Tensor  # 当前策略的 action logprobs
    ref_logprobs: torch.Tensor  # 参考策略的 action logprobs
    reward: float  # 对比奖励
    sample_index: str | int
    baseline_correct: bool
    with_memory_correct: bool
    sampling_logprobs: Optional[torch.Tensor] = None  # 采样时的 logprobs


class RerankerGRPOTrainingCallback(Callback):
    """
    Reranker GRPO 训练回调 - 完全对齐 verl 实现

    核心流程：
    1. 每个样本先跑一遍不带记忆的 baseline（由外部协调）
    2. Reranker 选择记忆后，保存 input/output/logprobs
    3. 带记忆跑完后，对比结果计算 reward
    4. 累积足够样本后用 GRPO 训练
    """

    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        self.config: Dict[str, Any] = self._load_config(config_path)

        # Reward 配置
        reward_cfg = self.config.get("reward", {})
        self.reward_baseline_correct_memory_correct = float(reward_cfg.get("baseline_correct_memory_correct", 0.2))
        self.reward_baseline_correct_memory_wrong = float(reward_cfg.get("baseline_correct_memory_wrong", -1.0))
        self.reward_baseline_wrong_memory_correct = float(reward_cfg.get("baseline_wrong_memory_correct", 1.0))
        self.reward_baseline_wrong_memory_wrong = float(reward_cfg.get("baseline_wrong_memory_wrong", -0.2))
        # 格式奖励
        self.reward_format_correct = float(reward_cfg.get("format_correct", 0.2))
        self.reward_format_wrong = float(reward_cfg.get("format_wrong", -0.2))

        # GRPO 配置
        self.group_size: int = int(self.config.get("group_size", 4))
        self.accumulate_samples: int = int(self.config.get("accumulate_samples", 1))
        self.warmup_samples: int = int(self.config.get("warmup_samples", 0))  # 前 N 个样本只评估不训练
        self.grpo_config: Dict[str, Any] = self.config.get("grpo", {})
        self.lora_config: Dict[str, Any] = self.config.get("lora", {})
        self.optim_config: Dict[str, Any] = self.config.get("optim", {})
        self.save_config: Dict[str, Any] = self.config.get("save", {})
        self.monitor_config: Dict[str, Any] = self.config.get("monitoring", {})

        # Reranker 模型配置
        self.reranker_model_path: str = self.config.get("reranker_model_path", "")
        self.reranker_device: str = self.config.get("reranker_device", "auto")

        # 状态
        self.pending_attempts: Dict[str | int, List[RerankerAttemptRecord]] = {}
        self.accumulated_groups: List[List[RerankerAttemptRecord]] = []
        self.trained_steps: int = 0
        self._state_file: Optional[str] = None
        self.log_path: Optional[str] = None

        # 模型相关
        self.reranker_model = None
        self.ref_model = None
        self.tokenizer = None
        self.chat_parser = None
        self.optimizer: Optional[AdamW] = None
        self.lora_applied: bool = False
        self.device = None

        # Baseline 结果缓存: sample_index -> is_correct
        self._baseline_results: Dict[str | int, bool] = {}

        # Warmup 计数器
        self._processed_samples: int = 0
        self._seen_sample_indices: set = set()  # 用于去重计数

        # Reranker 生成缓存: sample_index -> {input_ids, action_mask, logprobs, ...}
        self._reranker_generation_cache: Dict[str | int, Dict] = {}

        # 采样时的 logprobs 缓存
        self._sampling_logprobs_cache: Dict[str, torch.Tensor] = {}

        # TensorBoard
        self.writer: Optional[SummaryWriter] = None
        self.enable_tensorboard = self.monitor_config.get("tensorboard", True)

        # 日志头部（对齐 verl 格式）
        self._log_header = (
            "global_step\tsample_index\tepoch\tloss_total\tpg_loss\tkl_loss\t"
            "ppo_kl\tclipfrac\tmean_reward\tmean_adv\tbaseline_acc\tmemory_acc\tgrad_norm\n"
        )

        # 注册全局实例
        global _RERANKER_GRPO_INSTANCE
        _RERANKER_GRPO_INSTANCE = self
        print("[RerankerGRPO] Registered global instance")

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        now = datetime.now()
        placeholders = {
            "TIMESTAMP": now.strftime("%Y-%m-%d-%H-%M-%S"),
            "TIMESTAMP_DATE": now.strftime("%Y-%m-%d"),
            "TIMESTAMP_TIME": now.strftime("%H-%M-%S"),
        }

        def _replace(value: Any) -> Any:
            if isinstance(value, str):
                return value.format(**placeholders)
            if isinstance(value, dict):
                return {k: _replace(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_replace(v) for v in value]
            return value

        return _replace(cfg)

    def restore_state(self) -> None:
        if self._state_file is None:
            self._state_file = os.path.join(self.get_state_dir(), "reranker_state.json")
        if self._state_file and os.path.exists(self._state_file):
            try:
                state = json.load(open(self._state_file, "r"))
                self.trained_steps = state.get("trained_steps", 0)
                self._processed_samples = state.get("processed_samples", 0)
                self._seen_sample_indices = set(state.get("seen_sample_indices", []))
            except Exception:
                pass
        if self.log_path is None:
            self.log_path = os.path.join(self.get_state_dir(), "reranker_train_log.tsv")
        self._ensure_log_header()

    def _ensure_reranker_model(self) -> bool:
        """确保 reranker 模型已加载"""
        if self.reranker_model is not None:
            return True

        if not self.reranker_model_path:
            print("[RerankerGRPO] Error: reranker_model_path not configured")
            return False

        print(f"[RerankerGRPO] Loading reranker model from {self.reranker_model_path}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.reranker_model_path, trust_remote_code=True
        )
        self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer)

        # 使用 niuload 均匀分配显存
        device_map = niuload.balanced_load(self.reranker_model_path, return_device_map_only=True)

        # 加载 policy 模型
        self.reranker_model = AutoModelForCausalLM.from_pretrained(
            self.reranker_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.device = next(self.reranker_model.parameters()).device

        # 加载 ref model (frozen) - 仅当 beta > 0 时需要
        beta = float(self.grpo_config.get("beta", 0.0))
        if beta > 0:
            ref_path = self.grpo_config.get("reference_model_path", self.reranker_model_path)
            ref_device_map = niuload.balanced_load(ref_path, return_device_map_only=True)
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                ref_path,
                torch_dtype=torch.bfloat16,
                device_map=ref_device_map,
                trust_remote_code=True,
            )
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            print(f"[RerankerGRPO] Loaded ref model from {ref_path}")
        else:
            self.ref_model = None
            print("[RerankerGRPO] beta=0, skipping ref model loading (no KL loss)")

        # 应用 LoRA
        if not self.lora_applied and self.lora_config.get("enabled", True):
            lora_cfg = LoraConfig(
                r=self.lora_config.get("r", 16),
                lora_alpha=self.lora_config.get("alpha", 32),
                lora_dropout=self.lora_config.get("dropout", 0.05),
                target_modules=self.lora_config.get(
                    "target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                ),
                task_type="CAUSAL_LM",
            )
            save_dir = self.save_config.get("lora_output_dir")
            if save_dir and os.path.exists(save_dir):
                self.reranker_model = PeftModel.from_pretrained(
                    self.reranker_model, save_dir, is_trainable=True
                )
            else:
                self.reranker_model = get_peft_model(self.reranker_model, lora_cfg)
            self.reranker_model.print_trainable_parameters()
            self.lora_applied = True

        # 初始化优化器
        if self.optimizer is None:
            self.optimizer = AdamW(
                self.reranker_model.parameters(),
                lr=float(self.optim_config.get("learning_rate", 2e-5)),
                weight_decay=float(self.optim_config.get("weight_decay", 0.0)),
            )

        # 初始化 TensorBoard
        if self.enable_tensorboard and self.writer is None:
            save_dir = self.save_config.get("lora_output_dir", "outputs/reranker")
            tb_dir = os.path.join(os.path.dirname(save_dir), "tensorboard_reranker")
            os.makedirs(tb_dir, exist_ok=True)
            self.writer = SummaryWriter(tb_dir)
            print(f"[RerankerGRPO] TensorBoard logging to: {tb_dir}")

        print(f"[RerankerGRPO] Reranker model initialized on {self.device}")
        return True

    def compute_reward(self, baseline_correct: bool, with_memory_correct: bool, format_correct: bool = True) -> float:
        """根据 baseline 和 with_memory 结果计算对比奖励，加上格式奖励"""
        # 基础对比奖励
        if baseline_correct and with_memory_correct:
            base_reward = self.reward_baseline_correct_memory_correct
        elif baseline_correct and not with_memory_correct:
            base_reward = self.reward_baseline_correct_memory_wrong
        elif not baseline_correct and with_memory_correct:
            base_reward = self.reward_baseline_wrong_memory_correct
        else:
            base_reward = self.reward_baseline_wrong_memory_wrong

        # 格式奖励
        format_reward = self.reward_format_correct if format_correct else self.reward_format_wrong

        return base_reward + format_reward

    def register_baseline_result(self, sample_index: str | int, is_correct: bool) -> None:
        """注册 baseline (无记忆) 的结果"""
        self._baseline_results[sample_index] = is_correct
        print(f"[RerankerGRPO] Registered baseline result for {sample_index}: {'correct' if is_correct else 'wrong'}")

    def register_reranker_generation(
        self,
        sample_index: str | int,
        messages: List[Dict[str, str]],
        output_text: str,
        format_correct: bool = True,
    ) -> None:
        """
        注册 reranker 的生成结果（由 previous_sample_embedding_callback 调用）

        Args:
            sample_index: 样本索引
            messages: reranker 的输入 messages (system + user)
            output_text: reranker 生成的文本
            format_correct: 输出格式是否正确
        """
        if not self._ensure_reranker_model():
            return

        # Tokenize 输入 + 输出
        full_messages = messages + [{"role": "assistant", "content": output_text}]

        try:
            token_list, mask_list = convert_messages_to_tokens_and_masks(
                full_messages,
                tokenizer=self.tokenizer,
                parser=self.chat_parser,
                contains_first_msg=True,
                contains_generation_msg=True,
            )
        except Exception as e:
            print(f"[RerankerGRPO] Failed to tokenize reranker generation: {e}")
            return

        input_ids = torch.tensor(token_list, dtype=torch.long, device=self.device).unsqueeze(0)
        action_mask = torch.tensor(mask_list, dtype=torch.bool, device=self.device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # 计算采样时的 logprobs（eval 模式避免 dropout 影响）
        was_training = self.reranker_model.training
        self.reranker_model.eval()
        with torch.no_grad():
            sampling_logps_full = self._token_logprobs(self.reranker_model, input_ids, attention_mask)
            action_mask_shift = action_mask[:, 1:]
            sampling_logps = sampling_logps_full[action_mask_shift]
        if was_training:
            self.reranker_model.train()

        self._reranker_generation_cache[sample_index] = {
            "input_ids": input_ids.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "action_mask": action_mask.detach().cpu(),
            "sampling_logprobs": sampling_logps.detach().cpu(),
            "messages": messages,
            "output_text": output_text,
            "format_correct": format_correct,
        }
        print(f"[RerankerGRPO] Registered reranker generation for {sample_index}, "
              f"action_tokens={action_mask.sum().item()}, format_correct={format_correct}")

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        """样本完成时，计算 reward 并可能触发训练"""
        session = callback_args.current_session
        key = session.sample_index

        # 跳过 baseline 评估的结果（它会单独注册）
        if getattr(session, 'finish_reason', None) == "BASELINE_EVAL":
            return

        # 跳过 greedy 评估（只用于 metric，不参与训练）
        if getattr(session, 'finish_reason', None) == "GREEDY_EVAL":
            return

        # 检查是否有 reranker 生成记录
        if key not in self._reranker_generation_cache:
            return

        # 检查是否有 baseline 结果
        if key not in self._baseline_results:
            print(f"[RerankerGRPO] Warning: No baseline result for sample {key}, skipping")
            return

        if not self._ensure_reranker_model():
            return

        baseline_correct = self._baseline_results[key]
        # 只看 evaluation outcome，不需要检查 sample_status
        with_memory_correct = session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT

        # 获取格式信息
        cache = self._reranker_generation_cache[key]
        format_correct = cache.get("format_correct", True)

        # 计算对比奖励（包含格式奖励）
        reward = self.compute_reward(baseline_correct, with_memory_correct, format_correct)

        # 构建 attempt record
        attempt = self._build_attempt_record(cache, reward, key, baseline_correct, with_memory_correct)

        if attempt is None:
            print(f"[RerankerGRPO] Failed to build attempt record for {key}")
            return

        # 添加到 pending attempts
        if key not in self.pending_attempts:
            self.pending_attempts[key] = []
        self.pending_attempts[key].append(attempt)

        print(f"[RerankerGRPO] Sample {key}: baseline={'✓' if baseline_correct else '✗'}, "
              f"with_memory={'✓' if with_memory_correct else '✗'}, format={'✓' if format_correct else '✗'}, "
              f"reward={reward:.2f}")

        # 检查是否完成一个 group
        if len(self.pending_attempts[key]) >= self.group_size:
            group = self.pending_attempts.pop(key)

            # 更新样本计数器（用于 warmup）
            if key not in self._seen_sample_indices:
                self._seen_sample_indices.add(key)
                self._processed_samples += 1

            self._maybe_accumulate_group(group, callback_args)

        # 清理缓存（只在 group 完成后清理 baseline 结果）
        del self._reranker_generation_cache[key]

    def _build_attempt_record(
        self,
        cache: Dict,
        reward: float,
        sample_index: str | int,
        baseline_correct: bool,
        with_memory_correct: bool,
    ) -> Optional[RerankerAttemptRecord]:
        """构建 attempt record"""
        input_ids = cache["input_ids"].to(self.device)
        attention_mask = cache["attention_mask"].to(self.device)
        action_mask = cache["action_mask"].to(self.device)
        sampling_logprobs = cache["sampling_logprobs"]

        if action_mask.sum().item() == 0:
            return None

        # 计算当前策略的 logprobs（eval 模式）
        was_training = self.reranker_model.training
        self.reranker_model.eval()
        gen_logps_full = self._token_logprobs(self.reranker_model, input_ids, attention_mask)
        action_mask_shift = action_mask[:, 1:]
        gen_logps = gen_logps_full[action_mask_shift]
        if was_training:
            self.reranker_model.train()

        # 计算参考模型的 logprobs（仅当 ref_model 存在时）
        if self.ref_model is not None:
            with torch.no_grad():
                ref_logps_full = self._token_logprobs(self.ref_model, input_ids, attention_mask)
                ref_logps = ref_logps_full[action_mask_shift]
        else:
            # 没有 ref_model 时，使用 sampling_logprobs 或 gen_logps 作为占位
            ref_logps = sampling_logprobs if sampling_logprobs is not None else gen_logps.detach()

        return RerankerAttemptRecord(
            input_ids=input_ids.detach().cpu(),
            attention_mask=attention_mask.detach().cpu(),
            action_mask=action_mask.detach().cpu(),
            gen_logprobs=gen_logps.detach().cpu(),
            ref_logprobs=ref_logps.detach().cpu() if isinstance(ref_logps, torch.Tensor) else ref_logps,
            reward=reward,
            sample_index=sample_index,
            baseline_correct=baseline_correct,
            with_memory_correct=with_memory_correct,
            sampling_logprobs=sampling_logprobs,
        )

    @staticmethod
    def _token_logprobs(
        model, input_ids: torch.Tensor, attention_mask: torch.Tensor, enable_grad: bool = False
    ) -> torch.Tensor:
        """计算 token 级别的 log probabilities"""
        ctx = torch.enable_grad if enable_grad else torch.no_grad
        with ctx():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            target_ids = input_ids[:, 1:]
            log_probs = logits.log_softmax(dim=-1)
            token_log_probs = torch.gather(
                log_probs, dim=-1, index=target_ids.unsqueeze(-1)
            ).squeeze(-1)
        return token_log_probs

    def _maybe_accumulate_group(self, group: List[RerankerAttemptRecord], callback_args: CallbackArguments) -> None:
        """可能累积 group 并触发训练"""
        # Warmup 期间跳过训练
        if self._processed_samples <= self.warmup_samples:
            print(f"[RerankerGRPO] Warmup phase: {self._processed_samples}/{self.warmup_samples} samples, "
                  f"skipping training (collecting memories)")
            # 清理 baseline 结果
            for attempt in group:
                if attempt.sample_index in self._baseline_results:
                    del self._baseline_results[attempt.sample_index]
            return

        # 检查 reward 方差
        group_rewards = torch.tensor([a.reward for a in group], dtype=torch.float32)
        reward_std = group_rewards.std().item()
        epsilon = 1e-6

        if reward_std < epsilon:
            print(f"[RerankerGRPO] Discarding group with zero variance "
                  f"(all rewards = {group_rewards[0].item():.2f})")
            # 清理 baseline 结果
            for attempt in group:
                if attempt.sample_index in self._baseline_results:
                    del self._baseline_results[attempt.sample_index]
            return

        self.accumulated_groups.append(group)
        print(f"[RerankerGRPO] Accumulated group {len(self.accumulated_groups)}/{self.accumulate_samples} "
              f"(reward_std={reward_std:.4f})")

        # 达到累积数量后训练
        if len(self.accumulated_groups) >= self.accumulate_samples:
            self._train_on_accumulated_groups(callback_args)
            self.accumulated_groups = []
            # 训练完成后清理 baseline 结果
            for g in self.accumulated_groups:
                for attempt in g:
                    if attempt.sample_index in self._baseline_results:
                        del self._baseline_results[attempt.sample_index]

    def _train_on_accumulated_groups(self, callback_args: CallbackArguments) -> None:
        """
        在累积的多个 group 上进行批训练 - 完全对齐 verl
        """
        assert self.reranker_model is not None
        assert self.optimizer is not None

        if len(self.accumulated_groups) == 0:
            return

        if self.log_path is None:
            self.log_path = os.path.join(self.get_state_dir(), "reranker_train_log.tsv")
        self._ensure_log_header()
        device = self.device

        # 超参数（对齐 verl）
        beta = float(self.grpo_config.get("beta", 0.04))
        clip_ratio = float(self.grpo_config.get("clip_param", 0.2))
        clip_ratio_c = float(self.grpo_config.get("clip_ratio_c", 3.0))
        kl_penalty_mode = str(self.grpo_config.get("kl_penalty_mode", "k3"))
        loss_agg_mode = str(self.grpo_config.get("loss_agg_mode", "token-mean"))
        max_grad_norm = float(self.optim_config.get("max_grad_norm", 1.0))
        num_epochs = int(self.optim_config.get("num_train_epochs", 1))
        save_dir = self.save_config.get("lora_output_dir")
        epsilon = 1e-6

        # 收集所有 attempts 和计算 per-group 归一化的 advantages
        all_attempts: List[RerankerAttemptRecord] = []
        all_advantages: List[float] = []
        total_reward = 0.0
        baseline_correct_count = 0
        memory_correct_count = 0

        for group in self.accumulated_groups:
            if len(group) == 0:
                continue

            group_rewards = torch.tensor([a.reward for a in group], dtype=torch.float32)
            total_reward += group_rewards.sum().item()

            for a in group:
                if a.baseline_correct:
                    baseline_correct_count += 1
                if a.with_memory_correct:
                    memory_correct_count += 1

            # GRPO 组内归一化
            if len(group_rewards) == 1:
                group_advantages = [0.0]
            else:
                reward_mean = group_rewards.mean()
                reward_std = group_rewards.std()
                if reward_std < epsilon:
                    group_advantages = [0.0] * len(group)
                else:
                    normalized = (group_rewards - reward_mean) / (reward_std + epsilon)
                    group_advantages = normalized.tolist()

            for i, attempt in enumerate(group):
                all_attempts.append(attempt)
                all_advantages.append(group_advantages[i])

        if len(all_attempts) == 0:
            return

        advantages = torch.tensor(all_advantages, device=device, dtype=torch.float32)

        # 统计信息
        num_groups = len(self.accumulated_groups)
        total_attempts = len(all_attempts)
        mean_reward = total_reward / total_attempts
        baseline_acc = baseline_correct_count / total_attempts
        memory_acc = memory_correct_count / total_attempts

        print(f"[RerankerGRPO] Training on {num_groups} groups, {total_attempts} attempts")
        print(f"[RerankerGRPO] Baseline acc: {baseline_acc:.2%}, Memory acc: {memory_acc:.2%}")

        # 构建批处理张量
        batch_size = len(all_attempts)
        response_lengths = [a.gen_logprobs.shape[0] for a in all_attempts]
        max_response_len = max(response_lengths)

        old_log_prob = torch.zeros(batch_size, max_response_len, device=device)
        ref_log_prob = torch.zeros(batch_size, max_response_len, device=device)
        response_mask = torch.zeros(batch_size, max_response_len, device=device)
        token_level_advantages = torch.zeros(batch_size, max_response_len, device=device)

        for i, attempt in enumerate(all_attempts):
            seq_len = response_lengths[i]
            if attempt.sampling_logprobs is not None:
                old_log_prob[i, :seq_len] = attempt.sampling_logprobs.to(device)
            else:
                old_log_prob[i, :seq_len] = attempt.gen_logprobs.to(device)
            ref_log_prob[i, :seq_len] = attempt.ref_logprobs.to(device)
            response_mask[i, :seq_len] = 1.0
            token_level_advantages[i, :seq_len] = advantages[i]

        input_ids_list = [a.input_ids.to(device) for a in all_attempts]
        attention_mask_list = [a.attention_mask.to(device) for a in all_attempts]
        action_mask_list = [a.action_mask.to(device) for a in all_attempts]

        self.reranker_model.train()
        step_count = 0

        for epoch in range(num_epochs):
            # 梯度累积模式
            self.optimizer.zero_grad()

            total_pg_loss = 0.0
            total_kl_loss = 0.0
            total_ppo_kl = 0.0
            total_clipfrac = 0.0
            valid_samples = 0

            for i in range(batch_size):
                input_ids = input_ids_list[i]
                attention_mask = attention_mask_list[i]
                action_mask = action_mask_list[i]

                token_logps_full = self._token_logprobs(
                    self.reranker_model, input_ids, attention_mask, enable_grad=True
                )
                action_mask_shift = action_mask[:, 1:]
                seq_logps = token_logps_full[action_mask_shift]
                seq_len = seq_logps.shape[0]

                single_new_log_prob = torch.zeros(1, max_response_len, device=device)
                single_new_log_prob[0, :seq_len] = seq_logps

                single_old_log_prob = old_log_prob[i:i+1]
                single_ref_log_prob = ref_log_prob[i:i+1]
                single_response_mask = response_mask[i:i+1]
                single_advantages = token_level_advantages[i:i+1]

                # 使用 verl 的 compute_policy_loss
                pg_loss_i, pg_clipfrac_i, ppo_kl_i, _ = compute_policy_loss(
                    old_log_prob=single_old_log_prob,
                    log_prob=single_new_log_prob,
                    advantages=single_advantages,
                    response_mask=single_response_mask,
                    cliprange=clip_ratio,
                    cliprange_low=clip_ratio,
                    cliprange_high=clip_ratio,
                    clip_ratio_c=clip_ratio_c,
                    loss_agg_mode=loss_agg_mode,
                )

                # 使用 verl 的 kl_penalty
                per_token_kl_i = kl_penalty(
                    logprob=single_new_log_prob,
                    ref_logprob=single_ref_log_prob,
                    kl_penalty=kl_penalty_mode,
                )
                kl_loss_i = agg_loss(
                    loss_mat=per_token_kl_i,
                    loss_mask=single_response_mask,
                    loss_agg_mode=loss_agg_mode,
                )

                sample_loss = (pg_loss_i + beta * kl_loss_i) / batch_size

                if not (torch.isnan(sample_loss) or torch.isinf(sample_loss)):
                    sample_loss.backward()
                    total_pg_loss += pg_loss_i.item()
                    total_kl_loss += kl_loss_i.item()
                    total_ppo_kl += ppo_kl_i.item()
                    total_clipfrac += pg_clipfrac_i.item()
                    valid_samples += 1

            if valid_samples == 0:
                print(f"[RerankerGRPO] Warning: No valid samples, skipping")
                continue

            pg_loss = total_pg_loss / valid_samples
            kl_loss = total_kl_loss / valid_samples
            ppo_kl = total_ppo_kl / valid_samples
            pg_clipfrac = total_clipfrac / valid_samples
            total_loss = pg_loss + beta * kl_loss

            total_norm = torch.nn.utils.clip_grad_norm_(self.reranker_model.parameters(), max_grad_norm)
            self.optimizer.step()
            step_count += 1

            global_step = self.trained_steps + step_count
            mean_adv = advantages.mean().item()

            # 日志记录
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{global_step}\tbatch_{num_groups}groups\tepoch_{epoch}\t"
                    f"{total_loss:.6f}\t"
                    f"{pg_loss:.6f}\t"
                    f"{kl_loss:.6f}\t"
                    f"{ppo_kl:.6f}\t"
                    f"{pg_clipfrac:.4f}\t"
                    f"{mean_reward:.4f}\t"
                    f"{mean_adv:.4f}\t"
                    f"{baseline_acc:.4f}\t"
                    f"{memory_acc:.4f}\t"
                    f"{float(total_norm):.4f}\n"
                )

            # TensorBoard
            if self.writer:
                self.writer.add_scalar("reranker/loss_total", total_loss, global_step)
                self.writer.add_scalar("reranker/pg_loss", pg_loss, global_step)
                self.writer.add_scalar("reranker/kl_loss", kl_loss, global_step)
                self.writer.add_scalar("reranker/ppo_kl", ppo_kl, global_step)
                self.writer.add_scalar("reranker/pg_clipfrac", pg_clipfrac, global_step)
                self.writer.add_scalar("reranker/baseline_accuracy", baseline_acc, global_step)
                self.writer.add_scalar("reranker/memory_accuracy", memory_acc, global_step)
                self.writer.add_scalar("reranker/reward_mean", mean_reward, global_step)
                self.writer.add_scalar("reranker/grad_norm", total_norm, global_step)

            print(f"[RerankerGRPO] step={global_step} | groups={num_groups} | epoch={epoch} | "
                  f"loss={total_loss:.4f} | pg_loss={pg_loss:.4f} | kl_loss={kl_loss:.4f} | "
                  f"baseline_acc={baseline_acc:.2%} | memory_acc={memory_acc:.2%}")

        self.trained_steps += step_count

        # 保存模型
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.reranker_model.save_pretrained(save_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(save_dir)
            print(f"[RerankerGRPO] Saved checkpoint to {save_dir}")

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        if self._state_file is None:
            return
        state = {
            "trained_steps": self.trained_steps,
            "processed_samples": self._processed_samples,
            "seen_sample_indices": list(self._seen_sample_indices),
        }
        with open(self._state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _ensure_log_header(self) -> None:
        if self.log_path is None:
            return
        need_header = False
        if not os.path.exists(self.log_path):
            need_header = True
        else:
            try:
                if os.path.getsize(self.log_path) == 0:
                    need_header = True
            except Exception:
                need_header = True
        if need_header:
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write(self._log_header)
