import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import yaml
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM
from torch.optim import AdamW

from rllm.parser import ChatTemplateParser
from rllm.agents.utils import convert_messages_to_tokens_and_masks

from src.callbacks.callback import Callback, CallbackArguments
from src.typings import (
    Session,
    SessionEvaluationOutcome,
    SampleStatus,
    Role,
)
from src.agents.instance.language_model_agent import LanguageModelAgent
from src.language_models.instance.huggingface_language_model import (
    HuggingfaceLanguageModel,
)


@dataclass
class AttemptRecord:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    gen_logprobs: torch.Tensor  # 当前policy的logprobs（用于KL计算）
    ref_logprobs: torch.Tensor  # reference model的logprobs
    reward: float
    sample_index: str | int
    # 新增：保存采样时的logprobs用于PPO ratio计算
    sampling_logprobs: Optional[torch.Tensor] = None


class GRPOTrainingCallback(Callback):
    """
    On-policy GRPO + LoRA trainer.

    - Collects `group_size` attempts per sample and triggers an update immediately.
    - Masks user tokens when computing log-probs (only assistant/agent tokens contribute).
    - Keeps policy model updated in-place so later samples use the new LoRA weights.
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
        self._log_header = (
            "global_step\tsample_index\tattempt_idx\tloss_total\tloss_policy\tloss_kl\t"
            "reward\tgroup_acc\ttrain_started\n"
        )

        # 新增：缓存采样时的logprobs
        self._sampling_logprobs_cache: Dict[str, torch.Tensor] = {}

    @classmethod
    def is_unique(cls) -> bool:
        return True

    # --------------------------------------------------------------------- #
    # Lifecycle
    # --------------------------------------------------------------------- #
    def restore_state(self) -> None:
        if self._state_file is None:
            self._state_file = os.path.join(self.get_state_dir(), "state.json")
        if self._state_file and os.path.exists(self._state_file):
            try:
                state = json.load(open(self._state_file, "r"))
                self.trained_steps = state.get("trained_steps", 0)
            except Exception:
                pass
        if self.log_path is None:
            self.log_path = os.path.join(self.get_state_dir(), "train_log.tsv")
        self._ensure_log_header()

    def on_session_create(self, callback_args: CallbackArguments) -> None:
        # Lazily initialize models and optimizer when we know the agent instance.
        agent = callback_args.session_context.agent
        self._ensure_models(agent)
        self._override_inference_config(agent)

    def on_agent_inference(self, callback_args: CallbackArguments) -> None:
        """在agent推理后立即保存采样时的logprobs"""
        session = callback_args.current_session
        agent = callback_args.session_context.agent

        # 跳过greedy evaluation
        if getattr(session, 'finish_reason', None) == "GREEDY_EVAL":
            return

        if self.policy_language_model is None:
            return

        # 保存当前policy的logprobs用于后续PPO训练
        messages = self._build_messages(session)
        if len(messages) > 0:
            input_ids, action_mask = self._tokenize_with_mask(messages)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            # 获取采样时的policy logprobs（重要！）
            with torch.no_grad():
                sampling_logps_full = self._token_logprobs(
                    self.policy_language_model.model, input_ids, attention_mask
                )
                action_mask_shift = action_mask[:, 1:]
                sampling_logps = sampling_logps_full[action_mask_shift]

            # 使用更稳定的缓存key：sample_index + 当前turn数（agent推理的轮次）
            turn_count = len([item for item in range(session.chat_history.get_value_length())
                            if session.chat_history.get_item_deep_copy(item).role == Role.AGENT])
            cache_key = f"{session.sample_index}_turn_{turn_count}"
            self._sampling_logprobs_cache[cache_key] = sampling_logps.detach().cpu()

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        session = callback_args.current_session
        agent = callback_args.session_context.agent
        # Skip greedy-only evaluation sessions
        if session.finish_reason == "GREEDY_EVAL":
            return  # 贪心评估不参与训练

        if self.policy_language_model is None:
            self._ensure_models(agent)
        if self.policy_language_model is None or self.tokenizer is None:
            return  # Safety: nothing to do.

        reward = self._calc_reward(session)

        # 获取缓存的采样logprobs - 使用相同的turn计数方式
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
            keys_to_remove = [k for k in self._sampling_logprobs_cache.keys() if k.startswith(f"{session.sample_index}_")]
            for k in keys_to_remove:
                del self._sampling_logprobs_cache[k]

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        if self._state_file is None:
            return
        state = {"trained_steps": self.trained_steps}
        with open(self._state_file, "w") as f:
            json.dump(state, f, indent=2)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
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

    def _ensure_models(self, agent: Any) -> None:
        if not isinstance(agent, LanguageModelAgent):
            raise TypeError(
                "GRPOTrainingCallback requires LanguageModelAgent to access the underlying HuggingFace model."
            )
        language_model = getattr(agent, "_language_model", None)
        if not isinstance(language_model, HuggingfaceLanguageModel):
            raise TypeError(
                "GRPOTrainingCallback currently supports HuggingfaceLanguageModel only."
            )
        self.policy_language_model = language_model
        self.tokenizer = language_model.tokenizer
        if self.chat_parser is None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer)
        base_model = language_model.model  # could already be a PeftModel (from SFT)
        self.device = next(base_model.parameters()).device
        save_dir = self.save_config.get("lora_output_dir")
        self._state_file = os.path.join(self.get_state_dir(), "state.json")
        self.system_prompt = getattr(agent, "_system_prompt", "")
        # Apply or load LoRA only once
        if not self.lora_applied:
            # If base_model is already a PeftModel (from SFT), reuse it directly.
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
            # Always load a clean base model as reference (no LoRA)
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                ref_path,
                torch_dtype=self.policy_language_model.model.dtype,
                device_map="auto",
            )

    def _override_inference_config(self, agent: LanguageModelAgent) -> None:
        if not self.generation_config:
            return
        # Skip overriding when agent is forced to run greedy eval
        if getattr(agent, "_force_greedy", False):
            return
        if self.skip_override:
            return
        # Override agent inference config to force sampling (non-greedy).
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
        """
        Returns:
            input_ids: (1, L)
            action_mask: (1, L) bool, True where tokens belong to agent/assistant.
        """
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
            self.policy_language_model.model, input_ids, attention_mask  # type: ignore[arg-type]
        )
        action_mask_shift = action_mask[:, 1:]
        gen_logps = gen_logps_full[action_mask_shift]

        # 处理reference logprobs
        if self.ref_model is not None:
            # 有独立的ref model，使用它计算ref_logps
            ref_logps_full = self._token_logprobs(
                self.ref_model, input_ids, attention_mask
            )
            ref_logps = ref_logps_full[action_mask_shift]
        else:
            # 没有ref model时，使用采样时的logprobs作为参考（更合理）
            # 这样KL惩罚会约束模型不要偏离采样时的策略太远
            if sampling_logprobs is not None:
                ref_logps = sampling_logprobs
            else:
                # fallback：使用当前的gen_logps（这种情况KL=0，但至少不会出错）
                ref_logps = gen_logps.detach().clone()

        # 如果没有采样时的logprobs，使用当前的（这种情况不理想但作为fallback）
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
            sampling_logprobs=sampling_logprobs,  # 保存采样时的logprobs
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

        # 正确性奖励
        outcome = session.evaluation_record.outcome
        if outcome == SessionEvaluationOutcome.CORRECT:
            reward += 1.0

        # 完成状态奖励
        status = session.sample_status
        if status == SampleStatus.COMPLETED:
            reward += 0.5

        return reward

    def _train_on_group(self, sample_index: str | int, callback_args: CallbackArguments) -> None:
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
        # GRPO样本选择策略
        use_best_of_n = self.grpo_config.get("use_best_of_n", False)
        if use_best_of_n and len(attempts) > 1:
            # Best-of-N模式：选择最佳和最差进行对比学习
            # 注意：这种模式下只有2个样本，归一化会强制变成[+1, -1]
            best_idx = raw_rewards.argmax().item()
            worst_idx = raw_rewards.argmin().item()
            # 使用最佳和最差形成对比（只要有差异就训练）
            if raw_rewards[best_idx] != raw_rewards[worst_idx]:
                train_attempts = [attempts[best_idx], attempts[worst_idx]]
                train_rewards = torch.tensor([raw_rewards[best_idx], raw_rewards[worst_idx]], device=device)
            else:
                # 全部样本奖励相同，跳过训练（没有相对差异）
                return
        else:
            # 标准GRPO模式：使用全部样本进行相对比较
            train_attempts = attempts
            train_rewards = raw_rewards

        # GRPO核心：奖励归一化（实现Group Relative）
        if normalize and len(train_rewards) > 1:
            reward_std = train_rewards.std().item()
            if reward_std > 1e-8:  # 有方差才归一化
                rewards = (train_rewards - train_rewards.mean()) / (train_rewards.std() + 1e-8)
            else:
                # 完全相同的奖励，跳过训练（没有学习信号）
                return
        else:
            rewards = train_rewards
        beta = float(self.grpo_config.get("beta", 0.04))
        clip_param = float(self.grpo_config.get("clip_param", 0.2))
        grad_accum = int(self.optim_config.get("gradient_accumulation_steps", 1))
        max_grad_norm = float(self.optim_config.get("max_grad_norm", 1.0))
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
                token_logps_full = self._token_logprobs(
                    self.policy_model, input_ids, attention_mask, enable_grad=True
                )
                action_mask_shift = action_mask[:, 1:]
                new_logps = token_logps_full[action_mask_shift]
                # 使用采样时的logprobs计算PPO ratio（关键修复！）
                old_logps = attempt.sampling_logprobs.to(device) if attempt.sampling_logprobs is not None else attempt.gen_logprobs.to(device)
                ref_logps = attempt.ref_logprobs.to(device)

                # 计算PPO ratio（数值稳定版本）
                logp_diff = new_logps - old_logps
                # 限制logp差异范围，避免exp爆炸
                logp_diff = torch.clamp(logp_diff, -20.0, 20.0)
                ratio = torch.exp(logp_diff)
                clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)

                # 获取对应的advantage
                advantage = rewards[idx] if idx < len(rewards) else rewards[-1]
                policy_component = torch.min(ratio * advantage, clipped_ratio * advantage)

                # KL散度惩罚（数值稳定版本）
                # KL(π_ref || π_new) ≈ (ratio_ref - 1) - log(ratio_ref)
                logp_ref_diff = new_logps - ref_logps
                logp_ref_diff = torch.clamp(logp_ref_diff, -20.0, 20.0)  # 限制范围
                ratio_ref = torch.exp(logp_ref_diff)
                # 使用数值稳定的KL计算
                per_token_kl = ratio_ref - logp_ref_diff - 1.0

                policy_loss = -policy_component.mean()
                kl_loss = beta * per_token_kl.mean()

                # 额外的loss clip保护
                policy_loss = torch.clamp(policy_loss, -100.0, 100.0)
                kl_loss = torch.clamp(kl_loss, -10.0, 10.0)

                loss = (policy_loss + kl_loss) / grad_accum

                # 检测异常loss，跳过这个batch
                if torch.isnan(loss) or torch.isinf(loss) or abs(loss.item()) > 50.0:
                    print(f"[WARNING] Abnormal loss detected: {loss.item():.2f}, skipping this batch")
                    self.optimizer.zero_grad()  # 清空梯度
                    continue

                loss.backward()
                backward_calls += 1
                if (idx + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_model.parameters(), max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    step_count += 1
                    global_step = self.trained_steps + step_count
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(
                            f"{global_step}\t{sample_index}\t{idx}\t{float(loss.item()):.6f}\t"
                            f"{float(policy_loss.item()):.6f}\t{float(kl_loss.item()):.6f}\t"
                            f"{float(train_rewards[idx].item() if idx < len(train_rewards) else train_rewards[-1].item()):.6f}\t{group_acc:.4f}\t{train_started_flag}\n"
                        )
                    train_started_flag = 0
            # Flush remainder grads if any
            if backward_calls % grad_accum != 0 and backward_calls > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(), max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                step_count += 1
                global_step = self.trained_steps + step_count
                last_idx = min(backward_calls - 1, len(rewards) - 1)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{global_step}\t{sample_index}\t{last_idx}\t{float(loss.item()):.6f}\t"
                        f"{float(policy_loss.item()):.6f}\t{float(kl_loss.item()):.6f}\t"
                        f"{float(train_rewards[last_idx].item() if last_idx < len(train_rewards) else train_rewards[-1].item()):.6f}\t{group_acc:.4f}\t{train_started_flag}\n"
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
        # Write header if file missing or empty
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
