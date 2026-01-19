"""
Reflection GRPO/DAPO Training Callback - 对齐 verl 实现

训练 reflection 模型，使用 group accuracy 作为奖励：
- 每个 reflection 对应 k2 个 base model rollouts
- Reflection reward = group_acc (k2 个 rollouts 中正确的比例)
- 额外功能：如果 greedy decode 失败但 k1*k2 rollouts 中有成功的，存储一条成功轨迹到 memory

支持两种算法：
- GRPO: 使用 grpo 配置块
- DAPO: 使用 dapo 配置块，包含以下特性：
  - Clip-Higher: 非对称 clip (clip_low, clip_high)
  - 移除 KL penalty (beta=0)
  - Token-level loss aggregation
  - Dynamic Sampling: 过滤 accuracy=0 或 1 的 group
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
_REFLECTION_GRPO_INSTANCE: Optional["ReflectionGRPOTrainingCallback"] = None


def get_reflection_grpo_instance() -> Optional["ReflectionGRPOTrainingCallback"]:
    """获取全局 ReflectionGRPOTrainingCallback 实例"""
    return _REFLECTION_GRPO_INSTANCE


@dataclass
class ReflectionAttemptRecord:
    """Reflection 的一次尝试记录"""
    input_ids: torch.Tensor  # reflection 完整输入 (prompt + output)
    attention_mask: torch.Tensor
    action_mask: torch.Tensor  # 哪些 token 是 reflection 生成的
    gen_logprobs: torch.Tensor  # 当前策略的 action logprobs
    ref_logprobs: torch.Tensor  # 参考策略的 action logprobs
    reward: float  # group accuracy reward
    sample_index: str | int
    reflection_id: int  # 第几个 reflection (0 到 k1-1)
    group_correct_count: int  # k2 个 rollouts 中正确的数量
    group_size: int  # k2
    sampling_logprobs: Optional[torch.Tensor] = None  # 采样时的 logprobs


class ReflectionGRPOTrainingCallback(Callback):
    """
    Reflection GRPO 训练回调 - 完全对齐 verl 实现

    核心流程：
    1. 每个样本先跑一遍 greedy decode（用于 metric）
    2. 对每个样本，生成 k1 个 reflection
    3. 每个 reflection 对应 k2 个 base model rollouts
    4. Reflection 的 reward = 其 k2 个 rollouts 的 group_acc
    5. 累积足够样本后用 GRPO 训练 reflection model
    6. 如果 greedy 失败但 k1*k2 中有成功的，存储一条成功轨迹到 memory
    """

    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        self.config: Dict[str, Any] = self._load_config(config_path)

        # GRPO 配置
        self.k1: int = int(self.config.get("k1", 4))  # 每个样本的 reflection 数量
        self.k2: int = int(self.config.get("k2", 4))  # 每个 reflection 的 base model rollout 数量
        self.accumulate_samples: int = int(self.config.get("accumulate_samples", 1))
        self.warmup_samples: int = int(self.config.get("warmup_samples", 0))
        self.store_rollout_success_to_memory: bool = bool(
            self.config.get("store_rollout_success_to_memory", True)
        )  # 是否存储 rollout 成功轨迹到 memory
        
        # 支持 GRPO 和 DAPO 两种配置
        self.grpo_config: Dict[str, Any] = self.config.get("grpo", {})
        self.dapo_config: Dict[str, Any] = self.config.get("dapo", {})
        
        # 判断使用哪种算法
        self.use_dapo: bool = len(self.dapo_config) > 0
        self.algo_config: Dict[str, Any] = self.dapo_config if self.use_dapo else self.grpo_config
        
        self.lora_config: Dict[str, Any] = self.config.get("lora", {})
        self.optim_config: Dict[str, Any] = self.config.get("optim", {})
        self.save_config: Dict[str, Any] = self.config.get("save", {})
        self.monitor_config: Dict[str, Any] = self.config.get("monitoring", {})

        # Reflection 模型配置
        self.reflection_model_path: str = self.config.get("reflection_model_path", "")
        self.reflection_device: str = self.config.get("reflection_device", "auto")

        # Generation 配置（用于生成 reflection）
        self.generation_config: Dict[str, Any] = self.config.get("generation", {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        })

        # 状态
        self.pending_reflections: Dict[str | int, List[ReflectionAttemptRecord]] = {}
        self.accumulated_groups: List[List[ReflectionAttemptRecord]] = []
        self.trained_steps: int = 0
        self._state_file: Optional[str] = None
        self.log_path: Optional[str] = None

        # 模型相关
        self.reflection_model = None
        self.ref_model = None
        self.tokenizer = None
        self.chat_parser = None
        self.optimizer: Optional[AdamW] = None
        self.lora_applied: bool = False
        self.device = None

        # Greedy 结果缓存: sample_index -> is_correct
        self._greedy_results: Dict[str | int, bool] = {}

        # Reflection 生成缓存: (sample_index, reflection_id) -> {input_ids, action_mask, logprobs, ...}
        self._reflection_generation_cache: Dict[tuple[str | int, int], Dict] = {}

        # 轨迹保存相关
        self._rollout_log_dir: Optional[str] = None
        self._reflection_rollouts: Dict[str | int, List[Dict]] = {}  # sample_index -> list of reflection records

        # Base model rollout 结果: (sample_index, reflection_id) -> [is_correct_1, is_correct_2, ..., is_correct_k2]
        self._rollout_results: Dict[tuple[str | int, int], List[bool]] = {}

        # 成功轨迹候选（用于存储到 memory）: sample_index -> Session
        # 每个 sample 最多存储一条成功轨迹（第一条成功的）
        self._first_success_candidate: Dict[str | int, Optional[Session]] = {}

        # Warmup 计数器
        self._processed_samples: int = 0
        self._seen_sample_indices: set = set()

        # TensorBoard
        self.writer: Optional[SummaryWriter] = None
        self.enable_tensorboard = self.monitor_config.get("tensorboard", True)

        # 日志头部（对齐 verl 格式）
        self._log_header = (
            "global_step\tsample_index\tepoch\tloss_total\tpg_loss\tkl_loss\t"
            "ppo_kl\tclipfrac\tclipfrac_high\tmean_reward\tmean_group_acc\tmean_adv\tgrad_norm\tentropy\n"
        )

        # 注册全局实例
        global _REFLECTION_GRPO_INSTANCE
        _REFLECTION_GRPO_INSTANCE = self
        algo_name = "DAPO" if self.use_dapo else "GRPO"
        print(f"[Reflection{algo_name}] Registered global instance (algorithm={algo_name})")

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
            self._state_file = os.path.join(self.get_state_dir(), "reflection_state.json")
        if self._state_file and os.path.exists(self._state_file):
            try:
                state = json.load(open(self._state_file, "r"))
                self.trained_steps = state.get("trained_steps", 0)
                self._processed_samples = state.get("processed_samples", 0)
                self._seen_sample_indices = set(state.get("seen_sample_indices", []))
            except Exception:
                pass
        if self.log_path is None:
            self.log_path = os.path.join(self.get_state_dir(), "reflection_train_log.tsv")
        self._ensure_log_header()
        # 初始化轨迹保存目录
        if self._rollout_log_dir is None:
            state_dir = self.get_state_dir()
            main_output_dir = os.path.dirname(os.path.dirname(state_dir))
            self._rollout_log_dir = os.path.join(main_output_dir, "reflection_rollouts")
            os.makedirs(self._rollout_log_dir, exist_ok=True)

    def _ensure_reflection_model(self) -> bool:
        """确保 reflection 模型已加载"""
        if self.reflection_model is not None:
            return True

        if not self.reflection_model_path:
            print("[ReflectionGRPO] Error: reflection_model_path not configured")
            return False

        print(f"[ReflectionGRPO] Loading reflection model from {self.reflection_model_path}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.reflection_model_path, trust_remote_code=True
        )
        self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer)

        # 使用 niuload 均匀分配显存
        device_map = niuload.balanced_load(self.reflection_model_path, return_device_map_only=True)

        # 加载 policy 模型
        self.reflection_model = AutoModelForCausalLM.from_pretrained(
            self.reflection_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.device = next(self.reflection_model.parameters()).device

        # 加载 ref model (frozen) - 仅当 beta > 0 时需要
        beta = float(self.grpo_config.get("beta", 0.0))
        if beta > 0:
            ref_path = self.grpo_config.get("reference_model_path", self.reflection_model_path)
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
            print(f"[ReflectionGRPO] Loaded ref model from {ref_path}")
        else:
            self.ref_model = None
            print("[ReflectionGRPO] beta=0, skipping ref model loading (no KL loss)")

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
            save_dir_raw = self.save_config.get("lora_output_dir")
            save_dir = self._resolve_output_dir(save_dir_raw) if save_dir_raw else None
            if save_dir and os.path.exists(save_dir):
                self.reflection_model = PeftModel.from_pretrained(
                    self.reflection_model, save_dir, is_trainable=True
                )
            else:
                self.reflection_model = get_peft_model(self.reflection_model, lora_cfg)
            self.reflection_model.print_trainable_parameters()
            self.lora_applied = True

        # 初始化优化器
        if self.optimizer is None:
            self.optimizer = AdamW(
                self.reflection_model.parameters(),
                lr=float(self.optim_config.get("learning_rate", 2e-5)),
                weight_decay=float(self.optim_config.get("weight_decay", 0.0)),
            )

        # 初始化 TensorBoard
        if self.enable_tensorboard and self.writer is None:
            save_dir_raw = self.save_config.get("lora_output_dir", "outputs/reflection")
            save_dir = self._resolve_output_dir(save_dir_raw)
            tb_dir = os.path.join(os.path.dirname(save_dir), "tensorboard_reflection")
            os.makedirs(tb_dir, exist_ok=True)
            self.writer = SummaryWriter(tb_dir)
            print(f"[ReflectionGRPO] TensorBoard logging to: {tb_dir}")

        print(f"[ReflectionGRPO] Reflection model initialized on {self.device}")
        return True

    def register_greedy_result(self, sample_index: str | int, is_correct: bool) -> None:
        """注册 greedy decode 的结果"""
        self._greedy_results[sample_index] = is_correct
        # 初始化成功轨迹候选
        self._first_success_candidate[sample_index] = None
        print(f"[ReflectionGRPO] Registered greedy result for {sample_index}: {'correct' if is_correct else 'wrong'}")

    def generate_reflection(
        self,
        current_query: str,
        greedy_trajectory: str,
        greedy_correct: bool,
        sample_index: str | int,
        reflection_id: int
    ) -> tuple[str, str, List[Dict[str, str]]]:
        """
        基于 greedy decode 的轨迹生成 reflection

        Args:
            current_query: 当前任务的 query
            greedy_trajectory: greedy decode 的完整轨迹（agent 的推理过程）
            greedy_correct: greedy decode 是否正确
            sample_index: 样本索引
            reflection_id: 第几个 reflection (0 到 k1-1)

        Returns:
            tuple[str, str, List[Dict]]: (reflection_text, insight_text, messages)
                - reflection_text: 完整的 JSON 格式 reflection（用于存储和训练）
                - insight_text: 提取的 insight 字段（用于注入到 prompt）
                - messages: 生成时使用的 messages
        """
        if not self._ensure_reflection_model():
            return "", "", []

        # 使用与 previous_sample_embedding_callback 一致的 prompt
        system_prompt = """You are an expert AI Agent Analyst and Prompt Engineer specialized in optimizing Large Language Model agents for complex reasoning tasks (e.g., Text-to-SQL, Coding, Planning).
Your goal is to analyze the execution trajectory of an agent and distill **generalizable, actionable insights** that can serve as "rules of thumb" to improve future performance on similar (but not identical) tasks."""

        outcome_status = "Success" if greedy_correct else "Failure"
        error_message = "" if greedy_correct else "The agent's output was incorrect."

        user_prompt = f"""### Input Context
**1. User Query:**
{current_query}

**2. Agent Trajectory:**
{greedy_trajectory}

**3. Evaluation Outcome:**
{outcome_status}
{error_message}

---

### Analysis Instructions (Chain of Thought)
Please analyze the trajectory deeply and output a structured analysis. Follow these reasoning steps:

**Step 1: Diagnosis (Root Cause Analysis)**
* **If Failure:** pinpoint the *exact* turn where the logic diverged. Was it a syntax error? A hallucination of a column name? A logical gap? Did the agent misunderstand the schema?
* **If Success:** Identify the *critical decision* or "Aha!" moment that made this solution work. Why was this path effective compared to potential pitfalls?

**Step 2: Abstraction (Generalization)**
* **DO NOT** just summarize "The agent wrote a SQL query." (This is useless).
* **DO NOT** mention specific variable names (like `user_id = 5`) unless necessary for the rule pattern.
* **DO** formulate a general heuristic.
    * *Bad Insight:* "The agent forgot to join table A and B."
    * *SOTA Insight:* "When querying metric X, always perform an INNER JOIN between A and B on 'id' to filter out incomplete records, as relying on implicit joins leads to ambiguous column errors."

**Step 3: Refinement (Actionability)**
* Condense the insight into a single, high-impact "Tip" (under 50 words) that can be injected into a future system prompt.
* The insight must be self-contained.

### Output Format
You must output a valid JSON object strictly matching this schema:

```json
{{
  "diagnosis_reasoning": "Your step-by-step analysis of the failure or success factors...",
  "error_type": "Syntax Error | Logic Error | Schema Misunderstanding | Optimal Path | ...",
  "insight": "The refined, generalizable rule or tip.",
  "tags": ["relevant_tool_name", "relevant_concept"]
}}
```"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 使用本地模型生成
        import torch
        try:
            # 构建 chat messages
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            # 生成参数
            # reflection_id=-1 表示 retry 模式，使用 greedy decode 保证确定性
            # 其他情况使用配置的采样参数（用于训练时生成多样化的 reflections）
            if reflection_id == -1:
                # Retry 模式：greedy decode
                gen_kwargs = {
                    "max_new_tokens": self.generation_config.get("max_new_tokens", 512),
                    "do_sample": False,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }
            else:
                # 训练模式：采样生成
                gen_kwargs = {
                    "max_new_tokens": self.generation_config.get("max_new_tokens", 512),
                    "do_sample": self.generation_config.get("do_sample", True),
                    "temperature": self.generation_config.get("temperature", 0.7),
                    "top_p": self.generation_config.get("top_p", 0.9),
                    "pad_token_id": self.tokenizer.eos_token_id,
                }

            # 生成
            was_training = self.reflection_model.training
            self.reflection_model.eval()
            with torch.no_grad():
                outputs = self.reflection_model.generate(**inputs, **gen_kwargs)
            if was_training:
                self.reflection_model.train()

            # 解码，只取新生成的部分
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            reflection_json_str = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # 解析 JSON 输出，提取 insight 字段
            import json
            import re

            # 尝试解析 JSON
            insight_text = ""
            try:
                # 提取 JSON block（可能在 ```json 标记中）
                json_match = reflection_json_str
                if "```json" in reflection_json_str:
                    json_match = reflection_json_str.split("```json")[1].split("```")[0]
                elif "```" in reflection_json_str:
                    json_match = reflection_json_str.split("```")[1].split("```")[0]

                parsed = json.loads(json_match.strip())
                insight_text = parsed.get("insight", "")

                # 完整的 reflection 保留 JSON 格式（用于存储）
                reflection_text = reflection_json_str

            except Exception as parse_err:
                print(f"[ReflectionGRPO] Failed to parse JSON, using raw output: {parse_err}")
                # 如果解析失败，直接用原始输出
                reflection_text = reflection_json_str
                insight_text = reflection_json_str  # fallback

            print(f"[ReflectionGRPO] Generated reflection for {sample_index}, id={reflection_id}")
            print(f"  Insight: {insight_text[:80]}...")

            # 返回完整的 JSON、insight 和 messages
            return reflection_text, insight_text, messages

        except Exception as exc:
            print(f"[ReflectionGRPO] Failed to generate reflection: {exc}")
            return "", "", []

    def register_reflection_generation(
        self,
        sample_index: str | int,
        reflection_id: int,
        messages: List[Dict[str, str]],
        output_text: str,
    ) -> None:
        """
        注册 reflection 的生成结果

        Args:
            sample_index: 样本索引
            reflection_id: 第几个 reflection (0 到 k1-1)
            messages: reflection 的输入 messages (system + user)
            output_text: reflection 生成的文本
        """
        if not self._ensure_reflection_model():
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
            print(f"[ReflectionGRPO] Failed to tokenize reflection generation: {e}")
            return

        input_ids = torch.tensor(token_list, dtype=torch.long, device=self.device).unsqueeze(0)
        action_mask = torch.tensor(mask_list, dtype=torch.bool, device=self.device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # 计算采样时的 logprobs（eval 模式避免 dropout 影响）
        was_training = self.reflection_model.training
        self.reflection_model.eval()
        with torch.no_grad():
            sampling_logps_full = self._token_logprobs(self.reflection_model, input_ids, attention_mask)
            action_mask_shift = action_mask[:, 1:]
            sampling_logps = sampling_logps_full[action_mask_shift]
        if was_training:
            self.reflection_model.train()

        key = (sample_index, reflection_id)
        self._reflection_generation_cache[key] = {
            "input_ids": input_ids.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "action_mask": action_mask.detach().cpu(),
            "sampling_logprobs": sampling_logps.detach().cpu(),
            "messages": messages,
            "output_text": output_text,
        }
        # 初始化 rollout 结果列表
        self._rollout_results[key] = []

        # 记录 reflection 用于轨迹保存
        if sample_index not in self._reflection_rollouts:
            self._reflection_rollouts[sample_index] = []
        self._reflection_rollouts[sample_index].append({
            "reflection_id": reflection_id,
            "reflection_text": output_text,
            "greedy_correct": self._greedy_results.get(sample_index, False),
        })

        print(f"[ReflectionGRPO] Registered reflection generation for {sample_index}, reflection_id={reflection_id}, "
              f"action_tokens={action_mask.sum().item()}")

    def register_rollout_result(
        self,
        sample_index: str | int,
        reflection_id: int,
        is_correct: bool,
        session: Optional[Session] = None,
    ) -> None:
        """
        注册 base model rollout 的结果

        Args:
            sample_index: 样本索引
            reflection_id: 对应的 reflection id
            is_correct: 该 rollout 是否正确
            session: 该 rollout 的 session（用于可能的 memory 存储）
        """
        key = (sample_index, reflection_id)
        if key not in self._rollout_results:
            print(f"[ReflectionGRPO] Warning: No reflection generation for {key}, skipping rollout result")
            return

        self._rollout_results[key].append(is_correct)
        current_count = len(self._rollout_results[key])
        print(f"[ReflectionGRPO] Rollout result for {sample_index}, reflection_id={reflection_id}: "
              f"{'correct' if is_correct else 'wrong'} ({current_count}/{self.k2})")

        # 如果启用了成功轨迹存储，且是成功的 rollout，且 greedy 失败，且还没有存储成功轨迹，记录候选
        if self.store_rollout_success_to_memory:
            greedy_correct = self._greedy_results.get(sample_index, True)  # 默认 True 避免误存
            if is_correct and not greedy_correct and self._first_success_candidate.get(sample_index) is None:
                if session is not None:
                    self._first_success_candidate[sample_index] = session.model_copy(deep=True)
                    print(f"[ReflectionGRPO] Captured first success candidate for {sample_index} "
                          f"(greedy failed, rollout succeeded)")

        # 检查是否该 reflection 的所有 k2 个 rollouts 都完成了
        if current_count >= self.k2:
            self._maybe_complete_reflection(sample_index, reflection_id)

    def _maybe_complete_reflection(self, sample_index: str | int, reflection_id: int) -> None:
        """当一个 reflection 的所有 k2 个 rollouts 完成时，计算 reward 并可能触发训练"""
        key = (sample_index, reflection_id)
        if key not in self._reflection_generation_cache:
            return
        if key not in self._rollout_results:
            return

        rollout_results = self._rollout_results[key]
        if len(rollout_results) < self.k2:
            return  # 还没完成

        if not self._ensure_reflection_model():
            return

        # 计算 group accuracy reward
        correct_count = sum(rollout_results)
        group_acc = correct_count / self.k2
        reward = group_acc  # 直接使用 group accuracy 作为 reward

        # 构建 attempt record
        cache = self._reflection_generation_cache[key]
        attempt = self._build_attempt_record(
            cache, reward, sample_index, reflection_id, correct_count, self.k2
        )

        if attempt is None:
            print(f"[ReflectionGRPO] Failed to build attempt record for {key}")
            return

        # 添加到 pending reflections
        if sample_index not in self.pending_reflections:
            self.pending_reflections[sample_index] = []
        self.pending_reflections[sample_index].append(attempt)

        print(f"[ReflectionGRPO] Reflection {sample_index}, id={reflection_id}: "
              f"group_acc={group_acc:.2f} ({correct_count}/{self.k2}), reward={reward:.2f}")

        # 检查是否完成一个 group (k1 个 reflections)
        if len(self.pending_reflections[sample_index]) >= self.k1:
            group = self.pending_reflections.pop(sample_index)

            # 更新样本计数器（用于 warmup）
            if sample_index not in self._seen_sample_indices:
                self._seen_sample_indices.add(sample_index)
                self._processed_samples += 1

            self._maybe_accumulate_group(group)

        # 清理缓存
        del self._reflection_generation_cache[key]
        del self._rollout_results[key]

    def _build_attempt_record(
        self,
        cache: Dict,
        reward: float,
        sample_index: str | int,
        reflection_id: int,
        group_correct_count: int,
        group_size: int,
    ) -> Optional[ReflectionAttemptRecord]:
        """构建 attempt record"""
        input_ids = cache["input_ids"].to(self.device)
        attention_mask = cache["attention_mask"].to(self.device)
        action_mask = cache["action_mask"].to(self.device)
        sampling_logprobs = cache["sampling_logprobs"]

        if action_mask.sum().item() == 0:
            return None

        # 计算当前策略的 logprobs（eval 模式）
        was_training = self.reflection_model.training
        self.reflection_model.eval()
        gen_logps_full = self._token_logprobs(self.reflection_model, input_ids, attention_mask)
        action_mask_shift = action_mask[:, 1:]
        gen_logps = gen_logps_full[action_mask_shift]
        if was_training:
            self.reflection_model.train()

        # 计算参考模型的 logprobs（仅当 ref_model 存在时）
        if self.ref_model is not None:
            with torch.no_grad():
                ref_logps_full = self._token_logprobs(self.ref_model, input_ids, attention_mask)
                ref_logps = ref_logps_full[action_mask_shift]
        else:
            # 没有 ref_model 时，使用 sampling_logprobs 或 gen_logps 作为占位
            ref_logps = sampling_logprobs if sampling_logprobs is not None else gen_logps.detach()

        return ReflectionAttemptRecord(
            input_ids=input_ids.detach().cpu(),
            attention_mask=attention_mask.detach().cpu(),
            action_mask=action_mask.detach().cpu(),
            gen_logprobs=gen_logps.detach().cpu(),
            ref_logprobs=ref_logps.detach().cpu() if isinstance(ref_logps, torch.Tensor) else ref_logps,
            reward=reward,
            sample_index=sample_index,
            reflection_id=reflection_id,
            group_correct_count=group_correct_count,
            group_size=group_size,
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
            token_entropy = -(probs * log_probs).sum(dim=-1)  # (1, seq_len-1)
            # 只取 action tokens 的 entropy
            action_entropy = token_entropy[action_mask_shift]
            mean_entropy = action_entropy.mean() if action_entropy.numel() > 0 else torch.tensor(0.0)

        return token_log_probs, mean_entropy

    def _maybe_accumulate_group(self, group: List[ReflectionAttemptRecord]) -> None:
        """可能累积 group 并触发训练"""
        algo_name = "DAPO" if self.use_dapo else "GRPO"

        # 检查 reward 方差
        group_rewards = torch.tensor([a.reward for a in group], dtype=torch.float32)
        reward_std = group_rewards.std().item()
        epsilon = 1e-6

        # 无论是否训练，都先保存 reflection 轨迹
        self._save_reflection_rollouts(group, group_rewards)

        # Warmup 期间跳过训练
        if self._processed_samples <= self.warmup_samples:
            print(f"[Reflection{algo_name}] Warmup phase: {self._processed_samples}/{self.warmup_samples} samples, "
                  f"skipping training")
            return

        # 对于 Reflection model，reward 是 group_acc (0~1 的连续值)
        # Dynamic Sampling 的逻辑应该是：如果所有 k1 个 reflections 的 group_acc 完全相同（方差为 0），则丢弃
        # 这与下面的 variance 检查等价，所以不需要单独的 Dynamic Sampling 逻辑

        if reward_std < epsilon:
            print(f"[Reflection{algo_name}] Discarding group with zero variance "
                  f"(all rewards = {group_rewards[0].item():.2f})")
            return

        self.accumulated_groups.append(group)
        print(f"[Reflection{algo_name}] Accumulated group {len(self.accumulated_groups)}/{self.accumulate_samples} "
              f"(reward_std={reward_std:.4f})")

        # 达到累积数量后训练
        if len(self.accumulated_groups) >= self.accumulate_samples:
            self._train_on_accumulated_groups()
            self.accumulated_groups = []

    def _train_on_accumulated_groups(self) -> None:
        """
        在累积的多个 group 上进行批训练
        支持 GRPO 和 DAPO 两种算法
        """
        assert self.reflection_model is not None
        assert self.optimizer is not None

        if len(self.accumulated_groups) == 0:
            return

        if self.log_path is None:
            self.log_path = os.path.join(self.get_state_dir(), "reflection_train_log.tsv")
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
        all_attempts: List[ReflectionAttemptRecord] = []
        all_advantages: List[float] = []
        total_reward = 0.0
        total_group_acc = 0.0

        for group in self.accumulated_groups:
            if len(group) == 0:
                continue

            group_rewards = torch.tensor([a.reward for a in group], dtype=torch.float32)
            total_reward += group_rewards.sum().item()
            total_group_acc += sum(a.reward for a in group)  # reward 就是 group_acc

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

        if len(all_attempts) == 0:
            return

        advantages = torch.tensor(all_advantages, device=device, dtype=torch.float32)

        # 统计信息
        num_groups = len(self.accumulated_groups)
        total_attempts = len(all_attempts)
        mean_reward = total_reward / total_attempts
        mean_group_acc = total_group_acc / total_attempts

        print(f"[Reflection{algo_name}] Training on {num_groups} groups, {total_attempts} attempts")
        print(f"[Reflection{algo_name}] Mean group_acc: {mean_group_acc:.2%}")

        # 构建批处理张量
        batch_size = len(all_attempts)
        response_lengths = [a.gen_logprobs.shape[0] for a in all_attempts]
        max_response_len = max(response_lengths)
        total_tokens = sum(response_lengths)  # 用于 token-level loss

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

        self.reflection_model.train()
        step_count = 0

        for epoch in range(num_epochs):
            # 梯度累积模式
            self.optimizer.zero_grad()

            total_pg_loss = 0.0
            total_kl_loss = 0.0
            total_ppo_kl = 0.0
            total_clipfrac = 0.0
            total_clipfrac_high = 0.0  # DAPO: 追踪上界 clip
            total_entropy = 0.0  # DAPO: 追踪 entropy
            valid_samples = 0
            valid_tokens = 0

            for i in range(batch_size):
                input_ids = input_ids_list[i]
                attention_mask = attention_mask_list[i]
                action_mask = action_mask_list[i]

                # 单个序列的前向传播，同时计算 entropy
                token_logps_full, entropy_i = self._token_logprobs_and_entropy(
                    self.reflection_model, input_ids, attention_mask, action_mask, enable_grad=True
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
                if loss_agg_mode == "token-level" and self.use_dapo:
                    # token-level: pg_loss_i 是 token-mean，乘以 seq_len 还原为 token-sum
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
                        total_clipfrac_high += pg_clipfrac_lower_i.item() if hasattr(pg_clipfrac_lower_i, 'item') else pg_clipfrac_lower_i
                    # 累积真正的 entropy（已在 _token_logprobs_and_entropy 中计算）
                    total_entropy += entropy_i.item() if hasattr(entropy_i, 'item') else entropy_i
                    valid_samples += 1
                    valid_tokens += seq_len

            if valid_samples == 0:
                print(f"[Reflection{algo_name}] Warning: No valid samples, skipping")
                continue

            pg_loss = total_pg_loss / valid_samples
            kl_loss = total_kl_loss / valid_samples
            ppo_kl = total_ppo_kl / valid_samples
            pg_clipfrac = total_clipfrac / valid_samples
            pg_clipfrac_high = total_clipfrac_high / valid_samples
            mean_entropy = total_entropy / valid_samples
            total_loss = pg_loss + beta * kl_loss

            total_norm = torch.nn.utils.clip_grad_norm_(self.reflection_model.parameters(), max_grad_norm)
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
                    f"{pg_clipfrac_high:.4f}\t"
                    f"{mean_reward:.4f}\t"
                    f"{mean_group_acc:.4f}\t"
                    f"{mean_adv:.4f}\t"
                    f"{float(total_norm):.4f}\t"
                    f"{mean_entropy:.4f}\n"
                )

            # TensorBoard
            if self.writer:
                self.writer.add_scalar("reflection/loss_total", total_loss, global_step)
                self.writer.add_scalar("reflection/pg_loss", pg_loss, global_step)
                self.writer.add_scalar("reflection/kl_loss", kl_loss, global_step)
                self.writer.add_scalar("reflection/ppo_kl", ppo_kl, global_step)
                self.writer.add_scalar("reflection/pg_clipfrac", pg_clipfrac, global_step)
                self.writer.add_scalar("reflection/pg_clipfrac_high", pg_clipfrac_high, global_step)
                self.writer.add_scalar("reflection/group_accuracy", mean_group_acc, global_step)
                self.writer.add_scalar("reflection/reward_mean", mean_reward, global_step)
                self.writer.add_scalar("reflection/grad_norm", total_norm, global_step)
                self.writer.add_scalar("reflection/entropy", mean_entropy, global_step)

            print(f"[ReflectionGRPO] step={global_step} | groups={num_groups} | epoch={epoch} | "
                  f"loss={total_loss:.4f} | pg_loss={pg_loss:.4f} | kl_loss={kl_loss:.4f} | "
                  f"mean_group_acc={mean_group_acc:.2%}")

        self.trained_steps += step_count

        # 保存模型
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.reflection_model.save_pretrained(save_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(save_dir)
            print(f"[ReflectionGRPO] Saved checkpoint to {save_dir}")

    def get_success_candidate_for_memory(self, sample_index: str | int) -> Optional[Session]:
        """
        获取成功轨迹候选（用于存储到 memory）

        Returns:
            如果 greedy 失败但有成功的 rollout，返回第一条成功的 session，否则返回 None
        """
        return self._first_success_candidate.get(sample_index)

    def clear_success_candidate(self, sample_index: str | int) -> None:
        """清理成功轨迹候选"""
        if sample_index in self._first_success_candidate:
            del self._first_success_candidate[sample_index]

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

    def _save_reflection_rollouts(
        self, group: List[ReflectionAttemptRecord], group_rewards: torch.Tensor
    ) -> None:
        """
        保存 reflection 轨迹到日志目录

        保存格式：每个样本一个 JSON 文件，包含：
        - sample_index
        - greedy_correct
        - reflections: list of {
            reflection_id, reflection_text,
            reward (group_acc), reward_normalized (advantage),
            group_correct_count, group_size
          }
        """
        if len(group) == 0:
            return

        # 懒初始化轨迹保存目录
        if self._rollout_log_dir is None:
            try:
                state_dir = self.get_state_dir()
                main_output_dir = os.path.dirname(os.path.dirname(state_dir))
                self._rollout_log_dir = os.path.join(main_output_dir, "reflection_rollouts")
                os.makedirs(self._rollout_log_dir, exist_ok=True)
            except Exception as e:
                print(f"[ReflectionDAPO] Failed to create rollout log dir: {e}")
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

        # 获取 sample_index（所有 attempt 应该是同一个 sample）
        sample_index = group[0].sample_index
        greedy_correct = self._greedy_results.get(sample_index, False)

        # 构建保存数据
        reflections_data = []
        for i, attempt in enumerate(group):
            # 从缓存中获取 reflection 文本
            cached_reflection = None
            for cached in self._reflection_rollouts.get(sample_index, []):
                if cached["reflection_id"] == attempt.reflection_id:
                    cached_reflection = cached
                    break

            reflection_record = {
                "reflection_id": attempt.reflection_id,
                "reflection_text": cached_reflection["reflection_text"] if cached_reflection else "",
                "reward": attempt.reward,  # group_acc (归一化前)
                "reward_normalized": advantages[i],  # advantage (归一化后)
                "group_correct_count": attempt.group_correct_count,
                "group_size": attempt.group_size,
            }
            reflections_data.append(reflection_record)

        save_data = {
            "sample_index": sample_index,
            "greedy_correct": greedy_correct,
            "k1": self.k1,
            "k2": self.k2,
            "trained_step": self.trained_steps,
            "reflections": reflections_data,
        }

        # 保存到文件
        output_path = os.path.join(
            self._rollout_log_dir, f"sample_{sample_index}.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"[ReflectionGRPO] Saved reflection rollouts to {output_path}")

        # 清理缓存
        if sample_index in self._reflection_rollouts:
            del self._reflection_rollouts[sample_index]
