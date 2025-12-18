import json
import logging
import os
import re
from datetime import datetime
from typing import List, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

from src.callbacks.callback import Callback, CallbackArguments
from src.language_models.instance.huggingface_language_model import (
    HuggingfaceLanguageModel,
)
from src.typings import Role, Session, SampleStatus, SessionEvaluationOutcome

logger = logging.getLogger(__name__)

RECALL_SECTION_PATTERN = re.compile(
    r"\nBelow are prior trajectories related to the current query; use them as guidance before planning SQL:\n",
    re.IGNORECASE,
)


def _strip_recall(content: str) -> str:
    parts = RECALL_SECTION_PATTERN.split(content, maxsplit=1)
    return parts[0] if parts else content


class AssistantOnlyNoRecallDataset(Dataset):
    """只训练 assistant 回复，并剔除 user prompt 中的召回部分。"""

    def __init__(
        self,
        sessions: List[Session],
        tokenizer,
        *,
        max_length: int,
        start_from_user_turn: int,
    ) -> None:
        self.sessions = sessions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.start_from_user_turn = max(1, start_from_user_turn)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prepared_samples = self._prepare()

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.prepared_samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        sample = self.prepared_samples[idx]
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "labels": sample["labels"],
        }

    def export_serializable(self) -> List[dict[str, object]]:
        payload: List[dict[str, object]] = []
        for sample in self.prepared_samples:
            payload.append(
                {
                    "sample_index": sample["sample_index"],
                    "messages": sample["messages"],
                    "input_ids": sample["input_ids"].tolist(),
                    "attention_mask": sample["attention_mask"].tolist(),
                    "labels": sample["labels"].tolist(),
                    "truncated": sample["truncated"],
                }
            )
        return payload

    def _prepare(self) -> List[dict[str, object]]:
        prepared: List[dict[str, object]] = []
        for session in self.sessions:
            messages = self._build_messages(session)
            if not messages:
                continue
            stripped_messages = self._strip_recall(messages)
            if not stripped_messages:
                continue
            tokenized = self._tokenize(stripped_messages)
            assistant_mask = self._build_assistant_mask(stripped_messages, tokenized.shape[-1])
            input_ids, attention_mask, truncated = self._pad_or_truncate(tokenized)
            assistant_mask = self._align_mask(assistant_mask, input_ids.shape[0])
            labels = input_ids.clone()
            labels[~assistant_mask] = -100
            labels[attention_mask == 0] = -100
            prepared.append(
                {
                    "sample_index": session.sample_index,
                    "messages": stripped_messages,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "truncated": truncated,
                }
            )
        return prepared

    def _build_messages(self, session: Session) -> List[dict[str, str]]:
        total = session.chat_history.get_value_length()
        user_seen = 0
        start_index = 0
        for idx in range(total):
            item = session.chat_history.get_item_deep_copy(idx)
            if item.role == Role.USER:
                user_seen += 1
                if user_seen == self.start_from_user_turn:
                    start_index = idx
                    break
        if user_seen < self.start_from_user_turn:
            start_index = 0
        messages: List[dict[str, str]] = []
        for idx in range(start_index, total):
            item = session.chat_history.get_item_deep_copy(idx)
            role = "assistant" if item.role == Role.AGENT else item.role.value
            messages.append({"role": role, "content": item.content})
        return messages

    def _strip_recall(self, messages: List[dict[str, str]]) -> List[dict[str, str]]:
        stripped: List[dict[str, str]] = []
        for msg in messages:
            content = msg["content"]
            if msg["role"] == "user":
                content = _strip_recall(content)
                if not content.strip():
                    continue
            stripped.append({"role": msg["role"], "content": content})
        return stripped

    def _tokenize(self, messages: List[dict[str, str]]) -> torch.Tensor:
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        return tokenized[0]

    def _build_assistant_mask(self, messages: List[dict[str, str]], length: int) -> torch.Tensor:
        mask = torch.zeros(length, dtype=torch.bool)
        prev_len = 0
        for idx in range(len(messages)):
            partial = self.tokenizer.apply_chat_template(
                messages[: idx + 1],
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            )[0]
            current_len = partial.shape[-1]
            if messages[idx]["role"] == "assistant":
                mask[prev_len:current_len] = True
            prev_len = current_len
        return mask

    def _pad_or_truncate(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        if input_ids.shape[-1] > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = torch.ones(self.max_length, dtype=torch.long)
            return input_ids, attention_mask, True
        pad_len = self.max_length - input_ids.shape[-1]
        if pad_len == 0:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            return input_ids, attention_mask, False
        pad_tensor = torch.full((pad_len,), pad_token_id, dtype=torch.long)
        attention_mask = torch.cat(
            [torch.ones_like(input_ids, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]
        )
        input_ids = torch.cat([input_ids, pad_tensor], dim=0)
        return input_ids, attention_mask, False

    @staticmethod
    def _align_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
        if mask.shape[0] > target_len:
            return mask[:target_len]
        if mask.shape[0] < target_len:
            pad = torch.zeros(target_len - mask.shape[0], dtype=torch.bool)
            return torch.cat([mask, pad], dim=0)
        return mask


class NoRecallLossLogger(TrainerCallback):
    def __init__(self, loss_log_path: str, batch_id: int) -> None:
        self.loss_log_path = loss_log_path
        self.batch_id = batch_id
        if not os.path.exists(self.loss_log_path):
            os.makedirs(os.path.dirname(self.loss_log_path), exist_ok=True)
            with open(self.loss_log_path, "w", encoding="utf-8") as f:
                f.write("batch_id,step,loss,learning_rate,timestamp\n")

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if logs and "loss" in logs:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            loss = logs["loss"]
            lr = logs.get("learning_rate", "N/A")
            with open(self.loss_log_path, "a", encoding="utf-8") as f:
                f.write(f"{self.batch_id},{state.global_step},{loss},{lr},{timestamp}\n")


class TestTimeTrainingAssistantOnlyRecallFreeSFTCallback(Callback):
    def __init__(
        self,
        batch_size: int = 8,
        sft_data_dir: str = "outputs/{TIMESTAMP}/sft_data",
        loss_log_path: str = "outputs/{TIMESTAMP}/loss_log.csv",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        learning_rate: float = 2e-4,
        num_train_epochs: int = 1,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        max_seq_length: int = 2048,
        start_from_user_turn: int = 1,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.sft_data_dir_template = sft_data_dir
        self.loss_log_path_template = loss_log_path
        self.max_seq_length = max_seq_length
        self.start_from_user_turn = start_from_user_turn

        self.sft_data_dir: Optional[str] = None
        self.loss_log_path: Optional[str] = None
        self.trainer_output_dir: Optional[str] = None
        self.paths_initialized = False

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.successful_sessions: List[Session] = []
        self.model_ref = None
        self.tokenizer_ref = None
        self.language_model_ref: Optional[HuggingfaceLanguageModel] = None
        self.is_lora_applied = False
        self.training_batch_count = 0

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        session = callback_args.current_session
        if (
            session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT
            and session.sample_status == SampleStatus.COMPLETED
        ):
            self.successful_sessions.append(session.model_copy(deep=True))
            if len(self.successful_sessions) >= self.batch_size:
                self._run_training(callback_args)

    def _run_training(self, callback_args: CallbackArguments) -> None:
        self._initialize_paths()
        if not self._initialize_model_refs(callback_args):
            return
        if not self.is_lora_applied:
            self._apply_lora()

        dataset = AssistantOnlyNoRecallDataset(
            self.successful_sessions,
            self.tokenizer_ref,
            max_length=self.max_seq_length,
            start_from_user_turn=self.start_from_user_turn,
        )
        if len(dataset) == 0:
            logger.warning("[TTT-NoRecall] 当前批次在裁剪后为空，跳过训练。")
            self.successful_sessions = []
            return

        exported = dataset.export_serializable()
        self._save_batch(exported)

        training_args = TrainingArguments(
            output_dir=self.trainer_output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            overwrite_output_dir=True,
        )
        trainer = Trainer(
            model=self.model_ref,
            args=training_args,
            train_dataset=dataset,
            callbacks=[NoRecallLossLogger(self.loss_log_path, self.training_batch_count)],
        )
        trainer.train()

        self.successful_sessions = []
        self.training_batch_count += 1

    def _initialize_paths(self) -> None:
        if self.paths_initialized:
            return
        state_dir = self.get_state_dir()
        output_dir = os.path.dirname(os.path.dirname(state_dir))
        self.sft_data_dir = os.path.join(output_dir, "sft_data_no_recall")
        self.loss_log_path = os.path.join(output_dir, "loss_log_no_recall.csv")
        self.trainer_output_dir = os.path.join(output_dir, ".ttt_no_recall_trainer")
        os.makedirs(self.sft_data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.loss_log_path), exist_ok=True)
        os.makedirs(self.trainer_output_dir, exist_ok=True)
        self.paths_initialized = True

    def _initialize_model_refs(self, callback_args: CallbackArguments) -> bool:
        if self.model_ref is not None:
            return True
        agent = callback_args.session_context.agent
        if not hasattr(agent, "_language_model"):
            logger.warning("[TTT-NoRecall] Agent 缺少 _language_model，跳过训练。")
            return False
        language_model = agent._language_model
        if not isinstance(language_model, HuggingfaceLanguageModel):
            logger.warning("[TTT-NoRecall] 仅支持 HuggingfaceLanguageModel，跳过训练。")
            return False
        self.language_model_ref = language_model
        self.model_ref = language_model.model
        self.tokenizer_ref = language_model.tokenizer
        return True

    def _apply_lora(self) -> None:
        if self.language_model_ref is None or self.model_ref is None:
            return
        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model_ref = get_peft_model(self.model_ref, config)
        self.language_model_ref.model = self.model_ref
        self.is_lora_applied = True

    def _save_batch(self, samples: List[dict[str, object]]) -> None:
        batch_file = os.path.join(
            self.sft_data_dir,
            f"batch_no_recall_{self.training_batch_count:03d}.json",
        )
        payload = {
            "batch_id": self.training_batch_count,
            "sample_count": len(samples),
            "sample_indices": [sample["sample_index"] for sample in samples],
            "prepared_samples": samples,
        }
        with open(batch_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        state = {
            "training_batch_count": self.training_batch_count,
            "is_lora_applied": self.is_lora_applied,
            "successful_sessions": [s.model_dump() for s in self.successful_sessions],
        }
        path = os.path.join(self.get_state_dir(), "no_recall_state.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def restore_state(self) -> None:
        self._initialize_paths()
        path = os.path.join(self.get_state_dir(), "no_recall_state.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.training_batch_count = state.get("training_batch_count", 0)
        self.is_lora_applied = state.get("is_lora_applied", False)
        self.successful_sessions = [
            Session.model_validate(payload)
            for payload in state.get("successful_sessions", [])
        ]
