import json
import logging
import os
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
from src.typings import Role, Session, SessionEvaluationOutcome

logger = logging.getLogger(__name__)


class UserObservationDataset(Dataset):
    """只训练环境 observation（user role）的位置。"""

    def __init__(
        self,
        sessions: List[Session],
        tokenizer,
        *,
        max_length: int = 2048,
        learn_from_user_turn: int = 3,
    ) -> None:
        self.sessions = sessions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.learn_from_user_turn = max(1, learn_from_user_turn)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._prepared_samples = self._prepare_samples()

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._prepared_samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        sample = self._prepared_samples[idx]
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "labels": sample["labels"],
        }

    def export_observation_metadata(self) -> List[dict[str, object]]:
        return [
            {
                "sample_index": sample["sample_index"],
                "observation_messages": sample["observation_messages"],
            }
            for sample in self._prepared_samples
        ]

    def _prepare_samples(self) -> List[dict[str, object]]:
        prepared: List[dict[str, object]] = []
        for session in self.sessions:
            messages_simple: List[dict[str, str]] = []
            trainable_flags: List[bool] = []
            observation_messages: List[dict[str, object]] = []
            user_turn = 0
            total = session.chat_history.get_value_length()
            for idx in range(total):
                item = session.chat_history.get_item_deep_copy(idx)
                if item.role == Role.USER:
                    user_turn += 1
                    learnable = user_turn >= self.learn_from_user_turn
                    if learnable:
                        observation_messages.append(
                            {
                                "user_turn": user_turn,
                                "content": item.content,
                            }
                        )
                else:
                    learnable = False
                role_str = "assistant" if item.role == Role.AGENT else item.role.value
                messages_simple.append({"role": role_str, "content": item.content})
                trainable_flags.append(learnable and role_str == "user")

            if not observation_messages:
                continue

            full_input_ids = self._tokenize_messages(messages_simple)
            mask = self._build_trainable_mask(
                messages_simple, trainable_flags, full_input_ids.shape[-1]
            )
            input_ids, attention_mask, truncated = self._pad_or_truncate(full_input_ids)
            mask = self._align_mask(mask, input_ids.shape[0])

            labels = input_ids.clone()
            labels[~mask] = -100
            labels[attention_mask == 0] = -100

            prepared.append(
                {
                    "sample_index": session.sample_index,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "truncated": truncated,
                    "observation_messages": observation_messages,
                }
            )
        return prepared

    def _tokenize_messages(self, messages: List[dict[str, str]]) -> torch.Tensor:
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        return tokenized[0]

    def _build_trainable_mask(
        self,
        messages: List[dict[str, str]],
        trainable_flags: List[bool],
        length: int,
    ) -> torch.Tensor:
        mask = torch.zeros(length, dtype=torch.bool)
        prev_len = 0
        for msg_idx in range(len(messages)):
            partial_ids = self.tokenizer.apply_chat_template(
                messages[: msg_idx + 1],
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            )[0]
            current_len = partial_ids.shape[-1]
            if trainable_flags[msg_idx]:
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


class WorldModelLossLogger(TrainerCallback):
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


class TestTimeTrainingUserObservationCallback(Callback):
    """失败样本世界模型：只训练环境 observation。"""

    def __init__(
        self,
        batch_size: int = 8,
        sft_data_dir: str = "outputs/{TIMESTAMP}/world_model_sft",
        loss_log_path: str = "outputs/{TIMESTAMP}/world_model_loss.csv",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        learning_rate: float = 2e-4,
        num_train_epochs: int = 1,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        max_seq_length: int = 2048,
        learn_from_user_turn: int = 3,
        collect_from_session_rank: int = 1,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.sft_data_dir_template = sft_data_dir
        self.loss_log_path_template = loss_log_path
        self.max_seq_length = max_seq_length
        self.learn_from_user_turn = learn_from_user_turn
        self.collect_from_session_rank = max(1, collect_from_session_rank)

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

        self.failed_sessions: List[Session] = []
        self.model_ref = None
        self.tokenizer_ref = None
        self.language_model_ref: Optional[HuggingfaceLanguageModel] = None
        self.is_lora_applied = False
        self.training_batch_count = 0
        self.completed_session_count = 0

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        session = callback_args.current_session
        self.completed_session_count += 1
        if self.completed_session_count < self.collect_from_session_rank:
            return
        if session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT:
            return
        logger.info(
            "[TTT-UserObs] 收集失败样本 sample_index=%s", session.sample_index
        )
        self.failed_sessions.append(session.model_copy(deep=True))
        if len(self.failed_sessions) >= self.batch_size:
            self._run_training(callback_args)

    def _run_training(self, callback_args: CallbackArguments) -> None:
        self._initialize_paths()
        if not self._initialize_model_refs(callback_args):
            return
        if not self.is_lora_applied:
            self._apply_lora()

        dataset = UserObservationDataset(
            self.failed_sessions,
            self.tokenizer_ref,
            max_length=self.max_seq_length,
            learn_from_user_turn=self.learn_from_user_turn,
        )
        if len(dataset) == 0:
            logger.warning(
                "[TTT-UserObs] 本批失败样本中没有符合条件的 Observation，跳过训练。"
            )
            self.failed_sessions = []
            return

        observation_meta = dataset.export_observation_metadata()
        self._save_world_model_batch(observation_meta)

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
        loss_logger = WorldModelLossLogger(
            self.loss_log_path, self.training_batch_count
        )
        trainer = Trainer(
            model=self.model_ref,
            args=training_args,
            train_dataset=dataset,
            callbacks=[loss_logger],
        )
        trainer.train()
        self.failed_sessions = []
        self.training_batch_count += 1

    def _initialize_paths(self) -> None:
        if self.paths_initialized:
            return
        state_dir = self.get_state_dir()
        output_dir = os.path.dirname(os.path.dirname(state_dir))
        self.sft_data_dir = os.path.join(output_dir, "world_model_sft")
        self.loss_log_path = os.path.join(output_dir, "world_model_loss.csv")
        self.trainer_output_dir = os.path.join(output_dir, ".world_model_trainer_temp")
        os.makedirs(self.sft_data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.loss_log_path), exist_ok=True)
        os.makedirs(self.trainer_output_dir, exist_ok=True)
        self.paths_initialized = True

    def _initialize_model_refs(self, callback_args: CallbackArguments) -> bool:
        if self.model_ref is not None:
            return True
        agent = callback_args.session_context.agent
        if not hasattr(agent, "_language_model"):
            logger.warning("[TTT-UserObs] Agent 缺少 _language_model，跳过训练。")
            return False
        language_model = agent._language_model
        if not isinstance(language_model, HuggingfaceLanguageModel):
            logger.warning("[TTT-UserObs] 仅支持 HuggingfaceLanguageModel，跳过训练。")
            return False
        self.language_model_ref = language_model
        self.model_ref = language_model.model
        self.tokenizer_ref = language_model.tokenizer
        return True

    def _apply_lora(self) -> None:
        if self.language_model_ref is None or self.model_ref is None:
            return
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model_ref = get_peft_model(self.model_ref, lora_config)
        self.language_model_ref.model = self.model_ref
        self.is_lora_applied = True

    def _save_world_model_batch(
        self, observation_meta: List[dict[str, object]]
    ) -> None:
        batch_path = os.path.join(
            self.sft_data_dir,
            f"world_batch_{self.training_batch_count:03d}.json",
        )
        payload = {
            "batch_id": self.training_batch_count,
            "sample_count": len(observation_meta),
            "sample_indices": [meta["sample_index"] for meta in observation_meta],
            "world_model_samples": observation_meta,
        }
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        state = {
            "training_batch_count": self.training_batch_count,
            "is_lora_applied": self.is_lora_applied,
            "failed_sessions": [s.model_dump() for s in self.failed_sessions],
            "completed_session_count": self.completed_session_count,
        }
        state_path = os.path.join(self.get_state_dir(), "user_obs_state.json")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def restore_state(self) -> None:
        self._initialize_paths()
        state_path = os.path.join(self.get_state_dir(), "user_obs_state.json")
        if not os.path.exists(state_path):
            return
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.training_batch_count = state.get("training_batch_count", 0)
        self.is_lora_applied = state.get("is_lora_applied", False)
        self.failed_sessions = [
            Session.model_validate(payload)
            for payload in state.get("failed_sessions", [])
        ]
        self.completed_session_count = state.get("completed_session_count", 0)
