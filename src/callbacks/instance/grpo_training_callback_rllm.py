import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import yaml
from peft import LoraConfig, get_peft_model, PeftModel
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM

from rllm.agents.utils import convert_messages_to_tokens_and_masks
from rllm.parser import ChatTemplateParser

# verl核心算法（不需要Ray，纯PyTorch函数）
try:
    from verl.trainer.ppo.core_algos import (
        compute_policy_loss,
        kl_penalty,
    )
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False
    print("[WARNING] verl not available, using fallback PPO implementation")

from src.agents.instance.language_model_agent import LanguageModelAgent
from src.callbacks.callback import Callback, CallbackArguments
from src.language_models.instance.huggingface_language_model import (
    HuggingfaceLanguageModel,
)
from src.typings import Role, SampleStatus, Session, SessionEvaluationOutcome


@dataclass
class AttemptRecord:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    gen_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    reward: float
    sample_index: str | int
    sampling_logprobs: Optional[torch.Tensor] = None


class GRPOTrainingCallbackRLLM(Callback):
    """
    GRPO (Group Relative Policy Optimization) trainer with rllm/verl framework.

    Framework Integration:
    - Tokenization & Masking: rllm.agents.utils.convert_messages_to_tokens_and_masks
    - Chat Template Parsing: rllm.parser.ChatTemplateParser
    - PPO Loss Computation: verl.trainer.ppo.core_algos.compute_policy_loss
    - KL Penalty: verl.trainer.ppo.core_algos.kl_penalty
    - LoRA Fine-tuning: peft library
    - Monitoring: TensorBoard + detailed TSV logs

    GRPO核心特点:
    - 使用group-relative reward normalization代替GAE
    - 在线采样，每个样本采样group_size次后立即训练
    - Clipped PPO loss + KL penalty防止策略偏移
    """

    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        self.config: Dict[str, Any] = self._load_config(config_path)
        self.group_size: int = int(self.config.get("group_size", 1))
        self.best_metric_strategy: str = self.config.get(
            "best_metric_strategy", "best_reward"
        )
        self.generation_config: Dict[str, Any] = self.config.get("generation", {})
        self.grpo_config: Dict[str, Any] = self.config.get("grpo", {})
        self.lora_config: Dict[str, Any] = self.config.get("lora", {})
        self.optim_config: Dict[str, Any] = self.config.get("optim", {})
        self.save_config: Dict[str, Any] = self.config.get("save", {})
        self.monitor_config: Dict[str, Any] = self.config.get("monitoring", {})

        self.pending_attempts: dict[str | int, List[AttemptRecord]] = {}
        self.trained_steps: int = 0
        self._state_file: Optional[str] = None
        self.log_path: Optional[str] = None

        self.policy_language_model: Optional[HuggingfaceLanguageModel] = None
        self.policy_model = None
        self.ref_model = None
        self.tokenizer = None
        self.optimizer: Optional[AdamW] = None
        self.lora_applied: bool = False
        self.device = None
        self.system_prompt: str = ""
        self.skip_override: bool = False
        self.chat_parser = None

        # TensorBoard writer
        self.writer: Optional[SummaryWriter] = None
        self.enable_tensorboard = self.monitor_config.get("tensorboard", True)

        self._log_header = (
            "global_step\tsample_index\tattempt_idx\tloss_total\tloss_policy\tloss_kl\t"
            "reward\tgroup_acc\tratio_mean\tratio_max\tkl_mean\tgrad_norm\ttrain_started\n"
        )

        # 缓存采样时的logprobs
        self._sampling_logprobs_cache: Dict[str, torch.Tensor] = {}

        # 贪心评估统计
        self._greedy_correct_count: int = 0
        self._greedy_total_count: int = 0

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
            self._state_file = os.path.join(self.get_state_dir(), "state.json")
        if self._state_file and os.path.exists(self._state_file):
            try:
                state = json.load(open(self._state_file, "r"))
                self.trained_steps = state.get("trained_steps", 0)
                self._greedy_correct_count = state.get("greedy_correct_count", 0)
                self._greedy_total_count = state.get("greedy_total_count", 0)
            except Exception:
                pass
        if self.log_path is None:
            self.log_path = os.path.join(self.get_state_dir(), "train_log.tsv")
        self._ensure_log_header()

    def on_session_create(self, callback_args: CallbackArguments) -> None:
        agent = callback_args.session_context.agent
        self._ensure_models(agent)
        self._override_inference_config(agent)

    def on_agent_inference(self, callback_args: CallbackArguments) -> None:
        """在agent推理后立即保存采样时的logprobs"""
        session = callback_args.current_session
        agent = callback_args.session_context.agent

        if getattr(session, 'finish_reason', None) == "GREEDY_EVAL":
            return

        if self.policy_language_model is None:
            return

        messages = self._build_messages(session)
        if len(messages) > 0:
            input_ids, action_mask = self._tokenize_with_mask(messages)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            with torch.no_grad():
                sampling_logps_full = self._token_logprobs(
                    self.policy_language_model.model, input_ids, attention_mask
                )
                action_mask_shift = action_mask[:, 1:]
                sampling_logps = sampling_logps_full[action_mask_shift]

            # 使用稳定的缓存key
            turn_count = len([item for item in range(session.chat_history.get_value_length())
                            if session.chat_history.get_item_deep_copy(item).role == Role.AGENT])
            cache_key = f"{session.sample_index}_turn_{turn_count}"
            self._sampling_logprobs_cache[cache_key] = sampling_logps.detach().cpu()

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        session = callback_args.current_session
        agent = callback_args.session_context.agent

        # 贪心评估：统计正确率并记录到TensorBoard
        if session.finish_reason == "GREEDY_EVAL":
            self._greedy_total_count += 1
            if session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT:
                self._greedy_correct_count += 1

            # 计算累计正确率
            greedy_accuracy = self._greedy_correct_count / self._greedy_total_count

            # 记录到TensorBoard
            if self.writer:
                self.writer.add_scalar("eval/greedy_accuracy", greedy_accuracy, self._greedy_total_count)
                self.writer.add_scalar("eval/greedy_correct_count", self._greedy_correct_count, self._greedy_total_count)

            print(f"[GRPO-RLLM] Greedy eval #{self._greedy_total_count}: "
                  f"accuracy={greedy_accuracy:.2%} ({self._greedy_correct_count}/{self._greedy_total_count})")
            return

        if self.policy_language_model is None:
            self._ensure_models(agent)
        if self.policy_language_model is None or self.tokenizer is None:
            return

        reward = self._calc_reward(session)

        # 获取缓存的采样logprobs
        turn_count = len([item for item in range(session.chat_history.get_value_length())
                        if session.chat_history.get_item_deep_copy(item).role == Role.AGENT])
        cache_key = f"{session.sample_index}_turn_{turn_count}"
        sampling_logprobs = self._sampling_logprobs_cache.get(cache_key)

        attempt = self._build_attempt_record(session, reward, sampling_logprobs)
        if attempt is None:
            return

        key = session.sample_index
        if key not in self.pending_attempts:
            self.pending_attempts[key] = []
        self.pending_attempts[key].append(attempt)

        if len(self.pending_attempts[key]) >= self.group_size:
            self._train_on_group(key, callback_args)
            self.pending_attempts[key] = []
            # 清理缓存
            keys_to_remove = [k for k in self._sampling_logprobs_cache.keys()
                            if k.startswith(f"{session.sample_index}_")]
            for k in keys_to_remove:
                del self._sampling_logprobs_cache[k]

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        if self._state_file is None:
            return
        state = {
            "trained_steps": self.trained_steps,
            "greedy_correct_count": self._greedy_correct_count,
            "greedy_total_count": self._greedy_total_count,
        }
        with open(self._state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _ensure_models(self, agent: Any) -> None:
        if not isinstance(agent, LanguageModelAgent):
            raise TypeError(
                "GRPOTrainingCallbackRLLM requires LanguageModelAgent."
            )
        language_model = getattr(agent, "_language_model", None)
        if not isinstance(language_model, HuggingfaceLanguageModel):
            raise TypeError(
                "GRPOTrainingCallbackRLLM currently supports HuggingfaceLanguageModel only."
            )
        self.policy_language_model = language_model
        self.tokenizer = language_model.tokenizer
        if self.chat_parser is None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer)
        base_model = language_model.model
        self.device = next(base_model.parameters()).device
        save_dir = self.save_config.get("lora_output_dir")
        self._state_file = os.path.join(self.get_state_dir(), "state.json")
        self.system_prompt = getattr(agent, "_system_prompt", "")

        # 初始化TensorBoard
        if self.enable_tensorboard and self.writer is None:
            tb_dir = os.path.join(os.path.dirname(save_dir or "outputs"), "tensorboard")
            self.writer = SummaryWriter(tb_dir)
            print(f"[GRPO-RLLM] TensorBoard logging to: {tb_dir}")

        if not self.lora_applied:
            if isinstance(base_model, PeftModel):
                self.policy_model = base_model
                self.lora_applied = True
            else:
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
                if save_dir and os.path.exists(save_dir):
                    base_model = PeftModel.from_pretrained(
                        base_model, save_dir, is_trainable=True
                    )
                else:
                    base_model = get_peft_model(base_model, lora_cfg)
                self.policy_language_model.model = base_model
                self.policy_model = base_model
                self.lora_applied = True
        else:
            self.policy_model = self.policy_language_model.model

        if self.optimizer is None:
            self.optimizer = AdamW(
                self.policy_language_model.model.parameters(),
                lr=float(self.optim_config.get("learning_rate", 2e-5)),
                weight_decay=float(self.optim_config.get("weight_decay", 0.0)),
            )
        if self.ref_model is None and self.grpo_config.get("reference_model_path"):
            ref_path = self.grpo_config["reference_model_path"]
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                ref_path,
                torch_dtype=self.policy_language_model.model.dtype,
                device_map="auto",
            )

    def _override_inference_config(self, agent: LanguageModelAgent) -> None:
        if not self.generation_config:
            return
        if getattr(agent, "_force_greedy", False):
            return
        if self.skip_override:
            return
        agent._inference_config_dict = self.generation_config

    def _build_messages(self, session: Session) -> List[Dict[str, str]]:
        assert self.policy_language_model is not None
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        role_dict = self.policy_language_model.role_dict
        for i in range(session.chat_history.get_value_length()):
            item = session.chat_history.get_item_deep_copy(i)
            role = role_dict[item.role]
            messages.append({"role": role, "content": item.content})
        return messages

    def _tokenize_with_mask(self, messages: List[Dict[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.tokenizer is not None
        if self.chat_parser is None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer)
        device = self.device
        if len(messages) == 0:
            raise ValueError("Empty messages passed to _tokenize_with_mask.")
        token_list, mask_list = convert_messages_to_tokens_and_masks(
            messages,
            tokenizer=self.tokenizer,
            parser=self.chat_parser,
            contains_first_msg=True,
            contains_generation_msg=True,
        )
        input_ids = torch.tensor(token_list, dtype=torch.long, device=device).unsqueeze(0)
        action_mask = torch.tensor(mask_list, dtype=torch.bool, device=device).unsqueeze(0)
        return input_ids, action_mask

    @staticmethod
    def _token_logprobs(
        model, input_ids: torch.Tensor, attention_mask: torch.Tensor, enable_grad: bool = False
    ) -> torch.Tensor:
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

    def _build_attempt_record(
        self, session: Session, reward: float, sampling_logprobs: Optional[torch.Tensor] = None
    ) -> Optional[AttemptRecord]:
        messages = self._build_messages(session)
        if len(messages) == 0:
            return None
        input_ids, action_mask = self._tokenize_with_mask(messages)
        if action_mask.sum().item() == 0:
            return None
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        gen_logps_full = self._token_logprobs(
            self.policy_language_model.model, input_ids, attention_mask
        )
        action_mask_shift = action_mask[:, 1:]
        gen_logps = gen_logps_full[action_mask_shift]

        if self.ref_model is not None:
            ref_logps_full = self._token_logprobs(
                self.ref_model, input_ids, attention_mask
            )
            ref_logps = ref_logps_full[action_mask_shift]
        else:
            if sampling_logprobs is not None:
                ref_logps = sampling_logprobs
            else:
                ref_logps = gen_logps.detach().clone()

        if sampling_logprobs is None:
            sampling_logprobs = gen_logps.detach().clone()

        return AttemptRecord(
            input_ids=input_ids.detach().cpu(),
            attention_mask=attention_mask.detach().cpu(),
            action_mask=action_mask.detach().cpu(),
            gen_logprobs=gen_logps.detach().cpu(),
            ref_logprobs=ref_logps.detach().cpu(),
            reward=reward,
            sample_index=session.sample_index,
            sampling_logprobs=sampling_logprobs,
        )

    @staticmethod
    def _calc_reward(session: Session) -> float:
        """
        纯正向奖励函数：
        - 结果正确：+1.0
        - 状态是COMPLETED：+0.5
        - 其他情况：0（不扣分）
        """
        reward = 0.0
        outcome = session.evaluation_record.outcome
        if outcome == SessionEvaluationOutcome.CORRECT:
            reward += 1.0
        status = session.sample_status
        if status == SampleStatus.COMPLETED:
            reward += 0.5
        return reward

    def _train_on_group(self, sample_index: str | int, callback_args: CallbackArguments) -> None:
        """使用verl框架的GRPO训练"""
        assert self.policy_model is not None
        assert self.optimizer is not None
        attempts = self.pending_attempts[sample_index]
        if len(attempts) == 0:
            return
        if self.log_path is None:
            self.log_path = os.path.join(self.get_state_dir(), "train_log.tsv")
        self._ensure_log_header()

        device = self.device
        normalize = bool(self.grpo_config.get("normalize_rewards", False))
        raw_rewards = torch.tensor(
            [a.reward for a in attempts], device=device, dtype=torch.float32
        )

        # GRPO样本选择
        use_best_of_n = self.grpo_config.get("use_best_of_n", False)
        if use_best_of_n and len(attempts) > 1:
            best_idx = raw_rewards.argmax().item()
            worst_idx = raw_rewards.argmin().item()
            if raw_rewards[best_idx] != raw_rewards[worst_idx]:
                train_attempts = [attempts[best_idx], attempts[worst_idx]]
                train_rewards = torch.tensor([raw_rewards[best_idx], raw_rewards[worst_idx]], device=device)
            else:
                return
        else:
            train_attempts = attempts
            train_rewards = raw_rewards

        # 奖励归一化（GRPO核心）
        if normalize and len(train_rewards) > 1:
            reward_std = train_rewards.std().item()
            if reward_std > 1e-8:
                advantages = (train_rewards - train_rewards.mean()) / (train_rewards.std() + 1e-8)
            else:
                return
        else:
            # GRPO不用GAE，直接用reward作为advantage
            advantages = train_rewards

        beta = float(self.grpo_config.get("beta", 0.1))
        clip_param = float(self.grpo_config.get("clip_param", 0.2))
        grad_accum = int(self.optim_config.get("gradient_accumulation_steps", 1))
        max_grad_norm = float(self.optim_config.get("max_grad_norm", 0.5))
        num_epochs = int(self.optim_config.get("num_train_epochs", 1))
        save_dir = self.save_config.get("lora_output_dir")
        group_acc = float((raw_rewards > 0).float().mean().item())
        train_started_flag = 1

        self.policy_model.train()
        step_count = 0

        for _ in range(num_epochs):
            backward_calls = 0
            for idx, attempt in enumerate(train_attempts):
                action_mask = attempt.action_mask.to(device)
                if action_mask.sum().item() == 0:
                    continue

                input_ids = attempt.input_ids.to(device)
                attention_mask = attempt.attention_mask.to(device)

                # 计算当前logprobs
                token_logps_full = self._token_logprobs(
                    self.policy_model, input_ids, attention_mask, enable_grad=True
                )
                action_mask_shift = action_mask[:, 1:]
                new_logps = token_logps_full[action_mask_shift]

                old_logps = attempt.sampling_logprobs.to(device) if attempt.sampling_logprobs is not None else attempt.gen_logprobs.to(device)
                ref_logps = attempt.ref_logprobs.to(device)

                advantage = advantages[idx]

                # 使用verl核心算法计算PPO loss和KL penalty
                if VERL_AVAILABLE:
                    # 转换为verl需要的2D格式: (batch_size=1, seq_len)
                    new_logps_2d = new_logps.unsqueeze(0)
                    old_logps_2d = old_logps.unsqueeze(0)
                    ref_logps_2d = ref_logps.unsqueeze(0)
                    advantages_2d = advantage.expand_as(new_logps).unsqueeze(0)
                    response_mask = torch.ones_like(new_logps_2d, dtype=torch.bool)

                    # verl.compute_policy_loss: 返回 (pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower)
                    policy_loss, pg_clipfrac, ppo_kl, _ = compute_policy_loss(
                        old_log_prob=old_logps_2d,
                        log_prob=new_logps_2d,
                        advantages=advantages_2d,
                        response_mask=response_mask,
                        cliprange=clip_param,
                    )
                    # 计算ratio用于日志
                    logp_diff = torch.clamp(new_logps - old_logps, -20.0, 20.0)
                    ratio = torch.exp(logp_diff)
                    clipped_ratio = pg_clipfrac

                    # verl.kl_penalty: 使用k3（低方差KL近似，永远非负）
                    per_token_kl = kl_penalty(
                        logprob=new_logps_2d,
                        ref_logprob=ref_logps_2d,
                        kl_penalty="k3",
                    )
                    kl_loss = beta * per_token_kl.mean()
                else:
                    # Fallback: 手动实现（数值稳定版本）
                    logp_diff = torch.clamp(new_logps - old_logps, -20.0, 20.0)
                    ratio = torch.exp(logp_diff)
                    clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
                    policy_component = torch.min(ratio * advantage, clipped_ratio * advantage)
                    policy_loss = -policy_component.mean()

                    logp_ref_diff = torch.clamp(new_logps - ref_logps, -20.0, 20.0)
                    ratio_ref = torch.exp(logp_ref_diff)
                    per_token_kl = ratio_ref - logp_ref_diff - 1.0
                    kl_loss = beta * per_token_kl.mean()

                # Loss clip保护
                policy_loss = torch.clamp(policy_loss, -100.0, 100.0)
                kl_loss = torch.clamp(kl_loss, -10.0, 10.0)

                loss = (policy_loss + kl_loss) / grad_accum

                # 异常检测
                if torch.isnan(loss) or torch.isinf(loss) or abs(loss.item()) > 50.0:
                    print(f"[WARNING] Abnormal loss detected: {loss.item():.2f}, skipping batch")
                    self.optimizer.zero_grad()
                    continue

                loss.backward()
                backward_calls += 1

                if (idx + 1) % grad_accum == 0:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        self.policy_model.parameters(), max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    step_count += 1
                    global_step = self.trained_steps + step_count

                    # 文件日志
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(
                            f"{global_step}\t{sample_index}\t{idx}\t{float(loss.item()):.6f}\t"
                            f"{float(policy_loss.item()):.6f}\t{float(kl_loss.item()):.6f}\t"
                            f"{float(train_rewards[idx].item() if idx < len(train_rewards) else train_rewards[-1].item()):.6f}\t"
                            f"{group_acc:.4f}\t{float(ratio.mean().item()):.4f}\t{float(ratio.max().item()):.4f}\t"
                            f"{float(per_token_kl.mean().item()):.4f}\t{float(total_norm):.4f}\t{train_started_flag}\n"
                        )

                    # TensorBoard监控
                    if self.writer:
                        self.writer.add_scalar("train/loss_total", loss.item(), global_step)
                        self.writer.add_scalar("train/loss_policy", policy_loss.item(), global_step)
                        self.writer.add_scalar("train/loss_kl", kl_loss.item(), global_step)
                        self.writer.add_scalar("train/group_accuracy", group_acc, global_step)
                        self.writer.add_scalar("train/ratio_mean", ratio.mean().item(), global_step)
                        self.writer.add_scalar("train/ratio_max", ratio.max().item(), global_step)
                        self.writer.add_scalar("train/kl_mean", per_token_kl.mean().item(), global_step)
                        self.writer.add_scalar("train/grad_norm", total_norm, global_step)
                        self.writer.add_histogram("train/rewards", train_rewards, global_step)
                        self.writer.add_scalar("train/reward_mean", train_rewards.mean().item(), global_step)

                    train_started_flag = 0

            # Flush remainder grads
            if backward_calls % grad_accum != 0 and backward_calls > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(), max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                step_count += 1
                global_step = self.trained_steps + step_count
                last_idx = min(backward_calls - 1, len(advantages) - 1)

                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{global_step}\t{sample_index}\t{last_idx}\t{float(loss.item()):.6f}\t"
                        f"{float(policy_loss.item()):.6f}\t{float(kl_loss.item()):.6f}\t"
                        f"{float(train_rewards[last_idx].item() if last_idx < len(train_rewards) else train_rewards[-1].item()):.6f}\t"
                        f"{group_acc:.4f}\t{float(ratio.mean().item()):.4f}\t{float(ratio.max().item()):.4f}\t"
                        f"{float(per_token_kl.mean().item()):.4f}\t{float(total_norm):.4f}\t{train_started_flag}\n"
                    )
                train_started_flag = 0

        self.trained_steps += step_count
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.policy_model.save_pretrained(save_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(save_dir)

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
