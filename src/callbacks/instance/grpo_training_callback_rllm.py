import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import yaml
import niuload  # 均匀分配显存
from peft import LoraConfig, get_peft_model, PeftModel
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM

from rllm.agents.utils import convert_messages_to_tokens_and_masks
from rllm.parser import ChatTemplateParser

# verl 核心算法（完全对齐 verl 的 loss 计算）
from verl.trainer.ppo.core_algos import (
    compute_policy_loss,
    kl_penalty,
    agg_loss,
)
import verl.utils.torch_functional as verl_F

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
    is_correct: bool = False  # 直接记录是否正确
    sampling_logprobs: Optional[torch.Tensor] = None


class GRPOTrainingCallbackRLLM(Callback):
    """
    GRPO/DAPO (Group Relative Policy Optimization / Decoupled Clip and Dynamic Sampling Policy Optimization) 
    trainer with rllm/verl framework.

    Framework Integration:
    - Tokenization & Masking: rllm.agents.utils.convert_messages_to_tokens_and_masks
    - Chat Template Parsing: rllm.parser.ChatTemplateParser
    - PPO Loss Computation: verl.trainer.ppo.core_algos.compute_policy_loss
    - KL Penalty: verl.trainer.ppo.core_algos.kl_penalty
    - LoRA Fine-tuning: peft library
    - Monitoring: TensorBoard + detailed TSV logs

    支持两种算法：
    - GRPO: 使用 grpo 配置块
    - DAPO: 使用 dapo 配置块，包含以下特性：
      - Clip-Higher: 非对称 clip (clip_low, clip_high)
      - 移除 KL penalty (beta=0)
      - Token-level loss aggregation
      - Dynamic Sampling: 过滤 accuracy=0 或 1 的 group
    """

    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        self.config: Dict[str, Any] = self._load_config(config_path)
        self.group_size: int = int(self.config.get("group_size", 1))
        self.accumulate_samples: int = int(self.config.get("accumulate_samples", 1))  # 累积多少个样本后训练
        self.best_metric_strategy: str = self.config.get(
            "best_metric_strategy", "best_reward"
        )
        self.generation_config: Dict[str, Any] = self.config.get("generation", {})
        self.grpo_config: Dict[str, Any] = self.config.get("grpo", {})
        self.dapo_config: Dict[str, Any] = self.config.get("dapo", {})
        
        # 判断使用哪种算法
        self.use_dapo: bool = len(self.dapo_config) > 0
        self.algo_config: Dict[str, Any] = self.dapo_config if self.use_dapo else self.grpo_config
        
        self.lora_config: Dict[str, Any] = self.config.get("lora", {})
        self.optim_config: Dict[str, Any] = self.config.get("optim", {})
        self.save_config: Dict[str, Any] = self.config.get("save", {})
        self.monitor_config: Dict[str, Any] = self.config.get("monitoring", {})

        # Format reward 配置
        self.use_format_reward: bool = self.config.get("use_format_reward", True)

        self.pending_attempts: dict[str | int, List[AttemptRecord]] = {}
        self.accumulated_groups: List[List[AttemptRecord]] = []  # 累积的已完成的 group
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

        # 日志头部（对齐 verl 格式）
        self._log_header = (
            "global_step\tsample_index\tepoch\tloss_total\tpg_loss\tkl_loss\t"
            "ppo_kl\tclipfrac\tclipfrac_lower\tmean_reward\tmean_adv\tgroup_acc\tgrad_norm\tentropy\n"
        )

        # 缓存采样时的logprobs
        self._sampling_logprobs_cache: Dict[str, torch.Tensor] = {}

        # 贪心评估统计
        self._greedy_correct_count: int = 0
        self._greedy_total_count: int = 0

        # 轨迹保存相关
        self._rollout_log_dir: Optional[str] = None
        self._trajectory_cache: Dict[str | int, List[Dict]] = {}  # sample_index -> list of trajectory records

        # 打印算法类型
        algo_name = "DAPO" if self.use_dapo else "GRPO"
        print(f"[{algo_name}-RLLM] Initialized with algorithm={algo_name}")

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

    def _resolve_output_dir(self, relative_path: str) -> str:
        """将相对路径解析为基于主 output 目录的绝对路径"""
        try:
            # 从 callback state_dir 获取主 output 目录
            # state_dir 格式: outputs/reflection_grpo_k1_4_k2_4/{TIMESTAMP}/callback_state/callback_xxx
            state_dir = self.get_state_dir()
            # 向上两级获取主 output 目录
            main_output_dir = os.path.dirname(os.path.dirname(state_dir))
            # 解析相对路径
            if relative_path.startswith("outputs/"):
                # 移除 "outputs/" 前缀
                relative_path = relative_path[8:]
            # 返回绝对路径
            return os.path.join(main_output_dir, relative_path)
        except Exception:
            # 如果获取失败，返回原路径
            return relative_path

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
        # 初始化轨迹保存目录
        if self._rollout_log_dir is None:
            state_dir = self.get_state_dir()
            main_output_dir = os.path.dirname(os.path.dirname(state_dir))
            self._rollout_log_dir = os.path.join(main_output_dir, "base_model_rollouts")
            os.makedirs(self._rollout_log_dir, exist_ok=True)

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

            # 设置 eval 模式避免 dropout 影响（确保 sampling_logprobs 和 gen_logprobs 一致）
            was_training = self.policy_language_model.model.training
            self.policy_language_model.model.eval()
            with torch.no_grad():
                sampling_logps_full = self._token_logprobs(
                    self.policy_language_model.model, input_ids, attention_mask
                )
                action_mask_shift = action_mask[:, 1:]
                sampling_logps = sampling_logps_full[action_mask_shift]
            if was_training:
                self.policy_language_model.model.train()

            # 使用稳定的缓存key
            turn_count = len([item for item in range(session.chat_history.get_value_length())
                            if session.chat_history.get_item_deep_copy(item).role == Role.AGENT])
            cache_key = f"{session.sample_index}_turn_{turn_count}"
            self._sampling_logprobs_cache[cache_key] = sampling_logps.detach().cpu()

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        session = callback_args.current_session
        agent = callback_args.session_context.agent

        # 贪心评估：统计正确率并记录到TensorBoard
        if session.finish_reason == "GREEDY_EVAL" or session.finish_reason == "GREEDY_EVAL_RETRY":
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

        reward = self._calc_reward(session, use_format_reward=self.use_format_reward)

        # 如果使用 format reward，记录格式是否正确
        if self.use_format_reward:
            format_reward = self._calc_format_reward(session)
            if format_reward < 0:
                print(f"[GRPO-RLLM] Format violation detected for sample {session.sample_index}, "
                      f"format_reward={format_reward:.2f}, total_reward={reward:.2f}")

        # 获取缓存的采样logprobs
        turn_count = len([item for item in range(session.chat_history.get_value_length())
                        if session.chat_history.get_item_deep_copy(item).role == Role.AGENT])
        cache_key = f"{session.sample_index}_turn_{turn_count}"
        sampling_logprobs = self._sampling_logprobs_cache.get(cache_key)

        attempt = self._build_attempt_record(session, reward, sampling_logprobs)
        if attempt is None:
            return

        # Debug: 检查 sampling_logprobs 和 gen_logprobs 是否匹配
        if sampling_logprobs is not None and attempt.gen_logprobs.shape != sampling_logprobs.shape:
            print(f"[WARNING] Logprobs shape mismatch! "
                  f"sampling={sampling_logprobs.shape}, gen={attempt.gen_logprobs.shape}")
            print(f"  → This may cause ratio ≠ 1 at epoch_0. Consider single-turn tasks or fix multi-turn caching.")

        key = session.sample_index
        if key not in self.pending_attempts:
            self.pending_attempts[key] = []
        self.pending_attempts[key].append(attempt)

        # 缓存轨迹用于保存
        self._cache_trajectory(session, reward)

        if len(self.pending_attempts[key]) >= self.group_size:
            group = self.pending_attempts[key]
            self.pending_attempts[key] = []

            # 清理缓存
            keys_to_remove = [k for k in self._sampling_logprobs_cache.keys()
                            if k.startswith(f"{session.sample_index}_")]
            for k in keys_to_remove:
                del self._sampling_logprobs_cache[k]

            # 检查 group 方差，方差太小则丢弃
            group_rewards = torch.tensor([a.reward for a in group], dtype=torch.float32)
            reward_std = group_rewards.std().item()
            epsilon = 1e-6
            algo_name = "DAPO" if self.use_dapo else "GRPO"

            # DAPO Dynamic Sampling: 过滤 accuracy=0 或 1 的 group
            # 直接使用 is_correct 字段判断
            if self.use_dapo and self.algo_config.get("enable_dynamic_sampling", True):
                correct_count = sum(1 for a in group if a.is_correct)
                total_count = len(group)
                if correct_count == 0 or correct_count == total_count:
                    print(f"[{algo_name}-RLLM] Dynamic Sampling: Discarding group (sample={key}) with accuracy={correct_count}/{total_count} "
                          f"(all correct or all wrong)")
                    # 保存轨迹（即使被丢弃也保存）
                    self._save_trajectory_rollouts(key, group, group_rewards)
                    return

            if reward_std < epsilon:
                # 方差太小，丢弃这个 group，不计入累积
                print(f"[{algo_name}-RLLM] Discarding group (sample={key}) with zero variance "
                      f"(all rewards = {group_rewards[0].item():.2f})")
            else:
                # 方差足够，加入累积队列
                self.accumulated_groups.append(group)
                print(f"[{algo_name}-RLLM] Accumulated group {len(self.accumulated_groups)}/{self.accumulate_samples} "
                      f"(sample={key}, reward_std={reward_std:.4f})")

            # 保存轨迹（无论是否丢弃都保存）
            self._save_trajectory_rollouts(key, group, group_rewards)

            # 达到累积数量后训练
            if len(self.accumulated_groups) >= self.accumulate_samples:
                self._train_on_accumulated_groups(callback_args)
                self.accumulated_groups = []

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
        save_dir_raw = self.save_config.get("lora_output_dir")
        save_dir = self._resolve_output_dir(save_dir_raw) if save_dir_raw else None
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
            # 使用 niuload 均匀分配显存（与 policy 模型一致）
            ref_device_map = niuload.balanced_load(ref_path, return_device_map_only=True)
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                ref_path,
                torch_dtype=self.policy_language_model.model.dtype,
                device_map=ref_device_map,
            )
            self.ref_model.eval()  # ref_model 不训练，设为 eval 模式

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

    def _tokenize_with_mask(self, messages: List[Dict[str, str]], debug_print: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
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

        # Debug: 将 tokenization 和 mask 结果写入日志文件
        if debug_print:
            try:
                import os
                state_dir = self.get_state_dir()
                debug_log_path = os.path.join(state_dir, "tokenize_mask_debug.log")
                with open(debug_log_path, "w", encoding="utf-8") as f:
                    f.write("="*80 + "\n")
                    f.write("[DEBUG] _tokenize_with_mask 验证\n")
                    f.write("="*80 + "\n\n")

                    # 完整序列
                    full_text = self.tokenizer.decode(token_list)
                    f.write(f"[完整序列] (共 {len(token_list)} tokens):\n")
                    f.write(full_text + "\n")
                    f.write("\n" + "-"*40 + "\n")

                    # 逐 token 显示
                    f.write("\n[逐 Token 分析] (mask=1 表示参与训练):\n")
                    for i, (tok_id, mask) in enumerate(zip(token_list, mask_list)):
                        tok_str = self.tokenizer.decode([tok_id])
                        # 用 *** 标记 mask=1
                        marker = "***" if mask == 1 else "   "
                        f.write(f"  {i:4d}: {marker} mask={mask}  id={tok_id:6d}  |{repr(tok_str)}|\n")

                    # 统计
                    mask_1_count = sum(mask_list)
                    mask_0_count = len(mask_list) - mask_1_count
                    f.write(f"\n[统计] mask=0 (不训练): {mask_0_count}, mask=1 (训练): {mask_1_count}\n")

                    # 只打印 mask=1 的部分
                    masked_tokens = [tok_id for tok_id, mask in zip(token_list, mask_list) if mask == 1]
                    if masked_tokens:
                        masked_text = self.tokenizer.decode(masked_tokens)
                        f.write(f"\n[仅 mask=1 部分]:\n")
                        f.write(masked_text + "\n")

                    f.write("\n" + "="*80 + "\n")

                print(f"[DEBUG] Tokenize mask 调试信息已写入: {debug_log_path}")
            except Exception as e:
                print(f"[DEBUG] 写入调试日志失败: {e}")

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

    @staticmethod
    def _token_logprobs_and_entropy(
        model, input_ids: torch.Tensor, attention_mask: torch.Tensor, action_mask: torch.Tensor, enable_grad: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算 token log probabilities 和 entropy

        Returns:
            token_log_probs: shape (1, seq_len-1)
            entropy: 标量，action tokens 的平均 entropy
        """
        ctx = torch.enable_grad if enable_grad else torch.no_grad
        with ctx():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]  # (1, seq_len-1, vocab_size)
            target_ids = input_ids[:, 1:]
            log_probs = logits.log_softmax(dim=-1)
            token_log_probs = torch.gather(
                log_probs, dim=-1, index=target_ids.unsqueeze(-1)
            ).squeeze(-1)

            # 计算真正的 entropy: H = -sum(p * log(p))
            # 只对 action tokens 计算
            action_mask_shift = action_mask[:, 1:]  # 对齐 logits
            probs = torch.softmax(logits, dim=-1)  # (1, seq_len-1, vocab_size)
            # 避免 log(0)
            token_entropy = -(probs * log_probs).sum(dim=-1)  # (1, seq_len-1)
            # 只取 action tokens 的 entropy
            action_entropy = token_entropy[action_mask_shift]
            mean_entropy = action_entropy.mean() if action_entropy.numel() > 0 else torch.tensor(0.0)

        return token_log_probs, mean_entropy

    def _build_attempt_record(
        self, session: Session, reward: float, sampling_logprobs: Optional[torch.Tensor] = None
    ) -> Optional[AttemptRecord]:
        messages = self._build_messages(session)
        if len(messages) == 0:
            return None

        # Debug: 只在第一次调用时打印
        debug_print = not getattr(self, '_debug_printed', False)
        if debug_print:
            self._debug_printed = True

        input_ids, action_mask = self._tokenize_with_mask(messages, debug_print=debug_print)
        if action_mask.sum().item() == 0:
            return None
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # 设置 eval 模式避免 dropout 影响
        was_training = self.policy_language_model.model.training
        self.policy_language_model.model.eval()
        gen_logps_full = self._token_logprobs(
            self.policy_language_model.model, input_ids, attention_mask
        )
        action_mask_shift = action_mask[:, 1:]
        gen_logps = gen_logps_full[action_mask_shift]
        if was_training:
            self.policy_language_model.model.train()

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

        # 直接记录是否正确
        is_correct = session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT

        return AttemptRecord(
            input_ids=input_ids.detach().cpu(),
            attention_mask=attention_mask.detach().cpu(),
            action_mask=action_mask.detach().cpu(),
            gen_logprobs=gen_logps.detach().cpu(),
            ref_logprobs=ref_logps.detach().cpu(),
            reward=reward,
            sample_index=session.sample_index,
            is_correct=is_correct,
            sampling_logprobs=sampling_logprobs,
        )

    @staticmethod
    def _check_format(agent_response: str) -> bool:
        """
        检查 agent 输出是否符合格式要求：
        1. Action: Operation + SQL代码块
        2. Action: Answer + Final Answer

        Returns:
            True 如果格式正确，False 如果格式错误
        """
        import re

        # 去除首尾空白
        response = agent_response.strip()

        # 检查 Operation 格式: Action: Operation 后跟 SQL 代码块
        operation_pattern = r'Action:\s*Operation\s*```sql\s*.+?\s*```'
        if re.search(operation_pattern, response, re.DOTALL | re.IGNORECASE):
            return True

        # 检查 Answer 格式: Action: Answer 后跟 Final Answer
        answer_pattern = r'Action:\s*Answer\s*Final Answer:\s*.+'
        if re.search(answer_pattern, response, re.DOTALL | re.IGNORECASE):
            return True

        return False

    @staticmethod
    def _calc_format_reward(session: Session) -> float:
        """
        计算格式奖励：检查最后一个 agent 回复是否符合格式

        Returns:
            1.0 如果格式正确，-0.5 如果格式错误
        """
        from src.typings import Role

        # 获取最后一个 agent 回复
        last_agent_response = ""
        for i in range(session.chat_history.get_value_length() - 1, -1, -1):
            item = session.chat_history.get_item_deep_copy(i)
            if item.role == Role.AGENT:
                last_agent_response = item.content
                break

        if not last_agent_response:
            return -0.5  # 没有 agent 回复，格式错误

        if GRPOTrainingCallbackRLLM._check_format(last_agent_response):
            return 0.0  # 格式正确，不加额外奖励（保持中性）
        else:
            return -0.5  # 格式错误，惩罚

    @staticmethod
    def _calc_reward(session: Session, use_format_reward: bool = True) -> float:
        """
        奖励函数：
        - 结果正确：+1.0
        - 状态是COMPLETED：+0.5
        - 格式正确：+0.0（中性）
        - 格式错误：-0.5（惩罚）
        """
        reward = 0.0
        outcome = session.evaluation_record.outcome
        if outcome == SessionEvaluationOutcome.CORRECT:
            reward += 1.0
        status = session.sample_status
        if status == SampleStatus.COMPLETED:
            reward += 0.5

        # 添加格式奖励
        if use_format_reward:
            format_reward = GRPOTrainingCallbackRLLM._calc_format_reward(session)
            reward += format_reward

        return reward

    def _train_on_group(self, sample_index: str | int, callback_args: CallbackArguments) -> None:
        """
        完全对齐 verl 的 GRPO 训练：
        1. 批处理所有 attempts（padding + response_mask）
        2. 使用 verl 的 compute_policy_loss（dual-clip PPO）
        3. 使用 verl 的 kl_penalty（k3 模式）
        4. 使用 agg_loss 进行 token-mean 聚合
        """
        assert self.policy_model is not None
        assert self.optimizer is not None
        attempts = self.pending_attempts[sample_index]
        if len(attempts) == 0:
            return
        if self.log_path is None:
            self.log_path = os.path.join(self.get_state_dir(), "train_log.tsv")
        self._ensure_log_header()
        device = self.device

        # ==================== GRPO Advantage 计算（对齐 verl） ====================
        raw_rewards = torch.tensor(
            [a.reward for a in attempts], device=device, dtype=torch.float32
        )

        # GRPO 组内归一化（对齐 verl compute_grpo_outcome_advantage）
        epsilon = 1e-6
        if len(raw_rewards) == 1:
            # 单样本：advantage = 0，无学习信号
            advantages = torch.zeros_like(raw_rewards)
        else:
            reward_mean = raw_rewards.mean()
            reward_std = raw_rewards.std()
            if reward_std < epsilon:
                # 方差过小，跳过训练
                print(f"[GRPO-RLLM] Skipping group {sample_index}: reward std too small ({reward_std:.6f})")
                return
            advantages = (raw_rewards - reward_mean) / (reward_std + epsilon)

        # 超参数（对齐 verl）
        beta = float(self.grpo_config.get("beta", 0.04))
        clip_ratio = float(self.grpo_config.get("clip_param", 0.2))
        clip_ratio_c = float(self.grpo_config.get("clip_ratio_c", 3.0))  # dual-clip
        kl_penalty_mode = str(self.grpo_config.get("kl_penalty_mode", "k3"))
        loss_agg_mode = str(self.grpo_config.get("loss_agg_mode", "token-mean"))
        max_grad_norm = float(self.optim_config.get("max_grad_norm", 1.0))
        num_epochs = int(self.optim_config.get("num_train_epochs", 1))
        save_dir_raw = self.save_config.get("lora_output_dir")
        save_dir = self._resolve_output_dir(save_dir_raw) if save_dir_raw else None

        # 统计信息
        group_acc = sum(1 for a in attempts if a.is_correct) / len(attempts)  # 直接使用 is_correct

        # ==================== 构建批处理张量（对齐 verl batch 格式） ====================
        batch_size = len(attempts)
        # 获取每个 response 的长度（只计算 action tokens）
        response_lengths = [a.gen_logprobs.shape[0] for a in attempts]
        max_response_len = max(response_lengths)

        # 初始化 batched tensors: (batch_size, max_response_len)
        old_log_prob = torch.zeros(batch_size, max_response_len, device=device)
        ref_log_prob = torch.zeros(batch_size, max_response_len, device=device)
        response_mask = torch.zeros(batch_size, max_response_len, device=device)
        # 优势广播到 token 级别（对齐 verl: scores.unsqueeze(-1) * response_mask）
        token_level_advantages = torch.zeros(batch_size, max_response_len, device=device)

        for i, attempt in enumerate(attempts):
            seq_len = response_lengths[i]
            # old_log_prob: 采样时的 logprobs（用于 PPO ratio）
            if attempt.sampling_logprobs is not None:
                old_log_prob[i, :seq_len] = attempt.sampling_logprobs.to(device)
            else:
                old_log_prob[i, :seq_len] = attempt.gen_logprobs.to(device)
            # ref_log_prob: 参考模型的 logprobs（用于 KL penalty）
            ref_log_prob[i, :seq_len] = attempt.ref_logprobs.to(device)
            # response_mask: 标记有效 token（只有 assistant/agent tokens）
            response_mask[i, :seq_len] = 1.0
            # 优势广播到 token 级别
            token_level_advantages[i, :seq_len] = advantages[i]

        # 保存原始数据用于 forward
        input_ids_list = [a.input_ids.to(device) for a in attempts]
        attention_mask_list = [a.attention_mask.to(device) for a in attempts]
        action_mask_list = [a.action_mask.to(device) for a in attempts]

        self.policy_model.train()
        step_count = 0

        for epoch in range(num_epochs):
            # ==================== 梯度累积模式：逐个序列计算并累积梯度 ====================
            self.optimizer.zero_grad()

            # 用于记录统计量
            total_pg_loss = 0.0
            total_kl_loss = 0.0
            total_ppo_kl = 0.0
            total_clipfrac = 0.0
            total_clipfrac_lower = 0.0  # 下界 clip 比例
            total_entropy = 0.0
            valid_samples = 0
            valid_tokens = 0

            for i in range(batch_size):
                input_ids = input_ids_list[i]
                attention_mask = attention_mask_list[i]
                action_mask = action_mask_list[i]

                # 单个序列的前向传播，同时计算 entropy
                token_logps_full, entropy_i = self._token_logprobs_and_entropy(
                    self.policy_model, input_ids, attention_mask, action_mask, enable_grad=True
                )
                action_mask_shift = action_mask[:, 1:]
                seq_logps = token_logps_full[action_mask_shift]
                seq_len = seq_logps.shape[0]

                # 构建单个样本的张量
                single_new_log_prob = torch.zeros(1, max_response_len, device=device)
                single_new_log_prob[0, :seq_len] = seq_logps

                single_old_log_prob = old_log_prob[i:i+1]
                single_ref_log_prob = ref_log_prob[i:i+1]
                single_response_mask = response_mask[i:i+1]
                single_advantages = token_level_advantages[i:i+1]

                # 计算单个样本的 loss
                pg_loss_i, pg_clipfrac_i, ppo_kl_i, pg_clipfrac_lower_i = compute_policy_loss(
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

                # 单个样本的总 loss（除以 batch_size 进行平均）
                sample_loss = (pg_loss_i + beta * kl_loss_i) / batch_size

                # 累积梯度
                if not (torch.isnan(sample_loss) or torch.isinf(sample_loss)):
                    sample_loss.backward()
                    total_pg_loss += pg_loss_i.item()
                    total_kl_loss += kl_loss_i.item()
                    total_ppo_kl += ppo_kl_i.item()
                    total_clipfrac += pg_clipfrac_i.item()
                    if pg_clipfrac_lower_i is not None:
                        total_clipfrac_lower += pg_clipfrac_lower_i.item() if hasattr(pg_clipfrac_lower_i, 'item') else pg_clipfrac_lower_i
                    # 累积真正的 entropy（已在 _token_logprobs_and_entropy 中计算）
                    total_entropy += entropy_i.item() if hasattr(entropy_i, 'item') else entropy_i
                    valid_samples += 1
                    valid_tokens += seq_len

            if valid_samples == 0:
                print(f"[WARNING] No valid samples in this batch, skipping")
                continue

            # 平均统计量
            pg_loss = total_pg_loss / valid_samples
            kl_loss = total_kl_loss / valid_samples
            ppo_kl = total_ppo_kl / valid_samples
            pg_clipfrac = total_clipfrac / valid_samples
            pg_clipfrac_lower = total_clipfrac_lower / valid_samples
            mean_entropy = total_entropy / valid_samples
            total_loss = pg_loss + beta * kl_loss

            # 梯度裁剪和优化器更新
            total_norm = torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_grad_norm)
            self.optimizer.step()
            step_count += 1

            # ==================== 日志记录（对齐 verl 格式） ====================
            global_step = self.trained_steps + step_count
            # 计算额外统计量
            mean_reward = raw_rewards.mean().item()
            mean_adv = advantages.mean().item()

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{global_step}\t{sample_index}\tepoch_{epoch}\t"
                    f"{total_loss:.6f}\t"
                    f"{pg_loss:.6f}\t"
                    f"{kl_loss:.6f}\t"
                    f"{ppo_kl:.6f}\t"
                    f"{pg_clipfrac:.4f}\t"
                    f"{pg_clipfrac_lower:.4f}\t"
                    f"{mean_reward:.4f}\t"
                    f"{mean_adv:.4f}\t"
                    f"{group_acc:.4f}\t"
                    f"{float(total_norm):.4f}\t"
                    f"{mean_entropy:.4f}\n"
                )

            # TensorBoard 监控（对齐 verl 格式）
            if self.writer:
                self.writer.add_scalar("train/loss_total", total_loss, global_step)
                self.writer.add_scalar("train/pg_loss", pg_loss, global_step)
                self.writer.add_scalar("train/kl_loss", kl_loss, global_step)
                self.writer.add_scalar("train/ppo_kl", ppo_kl, global_step)
                self.writer.add_scalar("train/pg_clipfrac", pg_clipfrac, global_step)
                self.writer.add_scalar("train/pg_clipfrac_lower", pg_clipfrac_lower, global_step)
                self.writer.add_scalar("train/group_accuracy", group_acc, global_step)
                self.writer.add_scalar("train/reward_mean", mean_reward, global_step)
                self.writer.add_scalar("train/advantage_mean", mean_adv, global_step)
                self.writer.add_scalar("train/grad_norm", total_norm, global_step)
                self.writer.add_scalar("train/entropy", mean_entropy, global_step)
                self.writer.add_histogram("train/rewards", raw_rewards, global_step)
                self.writer.add_histogram("train/advantages", advantages, global_step)

            # 打印训练信息（对齐 verl 格式）
            print(f"[GRPO-RLLM] step={global_step} | sample={sample_index} | epoch={epoch} | "
                  f"loss={total_loss:.4f} | pg_loss={pg_loss:.4f} | "
                  f"kl_loss={kl_loss:.4f} | ppo_kl={ppo_kl:.4f} | "
                  f"clipfrac={pg_clipfrac:.4f} | reward={mean_reward:.4f} | acc={group_acc:.4f}")

        self.trained_steps += step_count

        # 保存模型
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.policy_model.save_pretrained(save_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(save_dir)

    def _train_on_accumulated_groups(self, callback_args: CallbackArguments) -> None:
        """
        在累积的多个 group 上进行批训练。
        支持 GRPO 和 DAPO 两种算法：
        - GRPO: 对称 clip，KL penalty，sample-level loss
        - DAPO: 非对称 clip (Clip-Higher)，无 KL penalty，token-level loss
        """
        assert self.policy_model is not None
        assert self.optimizer is not None

        if len(self.accumulated_groups) == 0:
            return

        if self.log_path is None:
            self.log_path = os.path.join(self.get_state_dir(), "train_log.tsv")
        self._ensure_log_header()
        device = self.device
        algo_name = "DAPO" if self.use_dapo else "GRPO"

        # 超参数（支持 GRPO 和 DAPO）
        if self.use_dapo:
            # DAPO 配置
            beta = 0.0  # DAPO 移除 KL penalty
            clip_low = float(self.algo_config.get("clip_low", 0.2))
            clip_high = float(self.algo_config.get("clip_high", 0.28))  # Clip-Higher
            loss_agg_mode = self.algo_config.get("loss_agg_mode", "token-level")
            kl_penalty_mode = "k3"  # 不使用，但需要传参
        else:
            # GRPO 配置
            beta = float(self.algo_config.get("beta", 0.04))
            clip_low = float(self.algo_config.get("clip_param", 0.2))
            clip_high = clip_low  # GRPO 使用对称 clip
            loss_agg_mode = self.algo_config.get("loss_agg_mode", "token-mean")
            kl_penalty_mode = str(self.algo_config.get("kl_penalty_mode", "k3"))
        
        clip_ratio_c = float(self.algo_config.get("clip_ratio_c", 3.0))
        max_grad_norm = float(self.optim_config.get("max_grad_norm", 1.0))
        num_epochs = int(self.optim_config.get("num_train_epochs", 1))
        save_dir_raw = self.save_config.get("lora_output_dir")
        save_dir = self._resolve_output_dir(save_dir_raw) if save_dir_raw else None
        epsilon = 1e-6

        # 收集所有 attempts 和计算 per-group 归一化的 advantages
        all_attempts: List[AttemptRecord] = []
        all_advantages: List[float] = []
        all_sample_indices: List[str | int] = []
        total_reward = 0.0
        total_correct = 0

        for group in self.accumulated_groups:
            if len(group) == 0:
                continue

            # 计算组内奖励
            group_rewards = torch.tensor([a.reward for a in group], dtype=torch.float32)
            total_reward += group_rewards.sum().item()
            total_correct += sum(1 for a in group if a.is_correct)  # 直接使用 is_correct 字段

            # GRPO/DAPO 组内归一化
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
                all_sample_indices.append(attempt.sample_index)

        if len(all_attempts) == 0:
            return

        # 转换为张量
        advantages = torch.tensor(all_advantages, device=device, dtype=torch.float32)

        # 统计信息
        num_groups = len(self.accumulated_groups)
        total_attempts = len(all_attempts)
        mean_reward = total_reward / total_attempts
        group_acc = total_correct / total_attempts

        print(f"[{algo_name}-RLLM] Training on {num_groups} accumulated groups, {total_attempts} total attempts")

        # 构建批处理张量
        batch_size = len(all_attempts)
        response_lengths = [a.gen_logprobs.shape[0] for a in all_attempts]
        max_response_len = max(response_lengths)
        total_tokens = sum(response_lengths)  # 用于 DAPO token-level loss

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

        self.policy_model.train()
        step_count = 0

        for epoch in range(num_epochs):
            # ==================== 梯度累积模式 ====================
            self.optimizer.zero_grad()

            total_pg_loss = 0.0
            total_kl_loss = 0.0
            total_ppo_kl = 0.0
            total_clipfrac = 0.0
            total_clipfrac_lower = 0.0  # 下界 clip 比例（verl 返回的是 lower）
            total_entropy = 0.0
            valid_samples = 0
            valid_tokens = 0

            for i in range(batch_size):
                input_ids = input_ids_list[i]
                attention_mask = attention_mask_list[i]
                action_mask = action_mask_list[i]

                # 单个序列的前向传播，同时计算 entropy
                token_logps_full, entropy_i = self._token_logprobs_and_entropy(
                    self.policy_model, input_ids, attention_mask, action_mask, enable_grad=True
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

                # 使用 verl 的 compute_policy_loss（支持非对称 clip）
                pg_loss_i, pg_clipfrac_i, ppo_kl_i, pg_clipfrac_lower_i = compute_policy_loss(
                    old_log_prob=single_old_log_prob,
                    log_prob=single_new_log_prob,
                    advantages=single_advantages,
                    response_mask=single_response_mask,
                    cliprange=clip_low,  # 兼容性参数
                    cliprange_low=clip_low,  # DAPO: 下界 clip
                    cliprange_high=clip_high,  # DAPO: 上界 clip (Clip-Higher)
                    clip_ratio_c=clip_ratio_c,
                    loss_agg_mode="token-mean",  # 内部使用 token-mean，外部再处理
                )

                # KL penalty（DAPO 时 beta=0，不影响）
                if beta > 0:
                    per_token_kl_i = kl_penalty(
                        logprob=single_new_log_prob,
                        ref_logprob=single_ref_log_prob,
                        kl_penalty=kl_penalty_mode,
                    )
                    kl_loss_i = agg_loss(
                        loss_mat=per_token_kl_i,
                        loss_mask=single_response_mask,
                        loss_agg_mode="token-mean",
                    )
                else:
                    kl_loss_i = torch.tensor(0.0, device=device)

                # Loss 聚合方式
                # DAPO Token-level: 所有 token 贡献相等
                # 由于 pg_loss_i 已经是 token-mean（单样本内平均），
                # 我们需要按 token 数量加权后求和，再除以总 token 数
                # 即: sum(pg_loss_i * seq_len_i) / total_tokens
                # 简化后: sum(pg_loss_i * seq_len_i / total_tokens)
                if loss_agg_mode == "token-level" and self.use_dapo:
                    # token-level: 每个样本按 token 数量加权
                    # pg_loss_i 是 token-mean，乘以 seq_len 还原为 token-sum
                    # 再除以 total_tokens 得到全局 token-mean
                    sample_loss = pg_loss_i * seq_len / total_tokens
                else:
                    # GRPO Sample-level: 平均每个样本
                    sample_loss = (pg_loss_i + beta * kl_loss_i) / batch_size

                if not (torch.isnan(sample_loss) or torch.isinf(sample_loss)):
                    sample_loss.backward()
                    total_pg_loss += pg_loss_i.item()
                    total_kl_loss += kl_loss_i.item() if isinstance(kl_loss_i, torch.Tensor) else kl_loss_i
                    total_ppo_kl += ppo_kl_i.item()
                    total_clipfrac += pg_clipfrac_i.item()
                    if pg_clipfrac_lower_i is not None:
                        total_clipfrac_lower += pg_clipfrac_lower_i.item() if hasattr(pg_clipfrac_lower_i, 'item') else pg_clipfrac_lower_i
                    # 累积真正的 entropy（已在 _token_logprobs_and_entropy 中计算）
                    total_entropy += entropy_i.item() if hasattr(entropy_i, 'item') else entropy_i
                    valid_samples += 1
                    valid_tokens += seq_len

            if valid_samples == 0:
                print(f"[{algo_name}-RLLM] Warning: No valid samples in accumulated batch, skipping")
                continue

            pg_loss = total_pg_loss / valid_samples
            kl_loss = total_kl_loss / valid_samples
            ppo_kl = total_ppo_kl / valid_samples
            pg_clipfrac = total_clipfrac / valid_samples
            pg_clipfrac_lower = total_clipfrac_lower / valid_samples
            mean_entropy = total_entropy / valid_samples
            total_loss = pg_loss + beta * kl_loss

            total_norm = torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_grad_norm)
            self.optimizer.step()
            step_count += 1

            global_step = self.trained_steps + step_count
            mean_adv = advantages.mean().item()

            # 日志记录（对齐日志头部格式）
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{global_step}\tbatch_{num_groups}groups\tepoch_{epoch}\t"
                    f"{total_loss:.6f}\t"
                    f"{pg_loss:.6f}\t"
                    f"{kl_loss:.6f}\t"
                    f"{ppo_kl:.6f}\t"
                    f"{pg_clipfrac:.4f}\t"
                    f"{pg_clipfrac_lower:.4f}\t"
                    f"{mean_reward:.4f}\t"
                    f"{mean_adv:.4f}\t"
                    f"{group_acc:.4f}\t"
                    f"{float(total_norm):.4f}\t"
                    f"{mean_entropy:.4f}\n"
                )

            # TensorBoard
            if self.writer:
                self.writer.add_scalar("train/loss_total", total_loss, global_step)
                self.writer.add_scalar("train/pg_loss", pg_loss, global_step)
                self.writer.add_scalar("train/kl_loss", kl_loss, global_step)
                self.writer.add_scalar("train/ppo_kl", ppo_kl, global_step)
                self.writer.add_scalar("train/pg_clipfrac", pg_clipfrac, global_step)
                self.writer.add_scalar("train/pg_clipfrac_lower", pg_clipfrac_lower, global_step)
                self.writer.add_scalar("train/batch_size", total_attempts, global_step)
                self.writer.add_scalar("train/num_groups", num_groups, global_step)
                self.writer.add_scalar("train/group_accuracy", group_acc, global_step)
                self.writer.add_scalar("train/reward_mean", mean_reward, global_step)
                self.writer.add_scalar("train/grad_norm", total_norm, global_step)
                self.writer.add_scalar("train/entropy", mean_entropy, global_step)

            print(f"[{algo_name}-RLLM] step={global_step} | groups={num_groups} | attempts={total_attempts} | epoch={epoch} | "
                  f"loss={total_loss:.4f} | pg_loss={pg_loss:.4f} | "
                  f"kl_loss={kl_loss:.4f} | clipfrac={pg_clipfrac:.4f} | clipfrac_lower={pg_clipfrac_lower:.4f} | "
                  f"reward={mean_reward:.4f} | acc={group_acc:.4f}")

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

    def _cache_trajectory(self, session: Session, reward: float) -> None:
        """缓存轨迹用于后续保存"""
        sample_index = session.sample_index
        if sample_index not in self._trajectory_cache:
            self._trajectory_cache[sample_index] = []

        # 提取轨迹文本
        trajectory_text = ""
        try:
            if self.policy_language_model is not None:
                agent_role_dict = self.policy_language_model.role_dict
                trajectory_text = session.chat_history.get_value_str(
                    agent_role_dict, start_index=0, end_index=None
                )
        except Exception:
            pass

        # 提取注入的 reflection（如果有）
        reflection_text = getattr(session, '_reflection_text', '')
        reflection_id = getattr(session, '_reflection_id', None)
        rollout_id = getattr(session, '_rollout_id', None)

        # 获取评估结果（只看 evaluation outcome，不需要检查 sample_status）
        is_correct = session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT

        self._trajectory_cache[sample_index].append({
            "rollout_id": rollout_id,
            "reflection_id": reflection_id,
            "reflection_text": reflection_text,
            "trajectory": trajectory_text,
            "reward": reward,
            "is_correct": is_correct,
            "sample_status": session.sample_status.value if session.sample_status else None,
            "evaluation_outcome": session.evaluation_record.outcome.value if session.evaluation_record.outcome else None,
        })

    def _save_trajectory_rollouts(
        self,
        sample_index: str | int,
        group: List[AttemptRecord],
        group_rewards: torch.Tensor
    ) -> None:
        """
        保存 base model 轨迹到日志目录

        保存格式：每个样本一个 JSON 文件，包含：
        - sample_index
        - group_size
        - rollouts: list of {
            rollout_id, reflection_id, reflection_text,
            trajectory, reward, reward_normalized,
            is_correct, sample_status, evaluation_outcome
          }
        """
        # 懒初始化轨迹保存目录
        if self._rollout_log_dir is None:
            try:
                state_dir = self.get_state_dir()
                main_output_dir = os.path.dirname(os.path.dirname(state_dir))
                self._rollout_log_dir = os.path.join(main_output_dir, "base_model_rollouts")
                os.makedirs(self._rollout_log_dir, exist_ok=True)
            except Exception as e:
                print(f"[DAPO-RLLM] Failed to create rollout log dir: {e}")
                return

        # 计算归一化后的 advantages
        epsilon = 1e-6
        if len(group_rewards) == 1:
            advantages = [0.0]
        else:
            reward_mean = group_rewards.mean()
            reward_std = group_rewards.std()
            if reward_std < epsilon:
                advantages = [0.0] * len(group)
            else:
                normalized = (group_rewards - reward_mean) / (reward_std + epsilon)
                advantages = normalized.tolist()

        # 从缓存中获取轨迹
        cached_trajectories = self._trajectory_cache.get(sample_index, [])

        # 构建保存数据
        rollouts_data = []
        for i, attempt in enumerate(group):
            # 尝试匹配缓存的轨迹
            cached = cached_trajectories[i] if i < len(cached_trajectories) else {}

            rollout_record = {
                "rollout_id": cached.get("rollout_id"),
                "reflection_id": cached.get("reflection_id"),
                "reflection_text": cached.get("reflection_text", ""),
                "trajectory": cached.get("trajectory", ""),
                "reward": attempt.reward,  # 归一化前
                "reward_normalized": advantages[i],  # 归一化后 (advantage)
                "is_correct": cached.get("is_correct", False),
                "sample_status": cached.get("sample_status"),
                "evaluation_outcome": cached.get("evaluation_outcome"),
            }
            rollouts_data.append(rollout_record)

        save_data = {
            "sample_index": sample_index,
            "group_size": self.group_size,
            "trained_step": self.trained_steps,
            "rollouts": rollouts_data,
        }

        # 保存到文件
        output_path = os.path.join(
            self._rollout_log_dir, f"sample_{sample_index}.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        algo_name = "DAPO" if self.use_dapo else "GRPO"
        print(f"[{algo_name}-RLLM] Saved trajectory rollouts to {output_path}")

        # 清理缓存
        if sample_index in self._trajectory_cache:
            del self._trajectory_cache[sample_index]
