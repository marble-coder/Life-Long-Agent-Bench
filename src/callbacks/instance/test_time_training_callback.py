import os
import json
import torch
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

# 确保已安装: pip install peft transformers accelerate
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import TrainingArguments, Trainer, TrainerCallback
from torch.utils.data import Dataset

from src.callbacks.callback import Callback, CallbackArguments
from src.typings import Session, SessionEvaluationOutcome, SampleStatus
from src.language_models.instance.huggingface_language_model import HuggingfaceLanguageModel

logger = logging.getLogger(__name__)


class SftTrajectoryDataset(Dataset):
    """
    用于 SFT 的轨迹数据集
    兼容所有支持 apply_chat_template 的模型（Llama, Qwen, Mistral 等）
    """
    def __init__(self, trajectories: List[Session], tokenizer, max_length=2048):
        self.trajectories = trajectories
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.trajectories)

    # def __getitem__(self, idx):
    #     session = self.trajectories[idx]
        
    #     # 将 ChatHistory 转换为标准的 messages 格式
    #     messages = []
    #     for item in session.chat_history.value:
    #         messages.append({
    #             "role": item.role.value,  # "user" 或 "agent"
    #             "content": item.content
    #         })
        
    #     # 使用 tokenizer 的聊天模板（自动适配 Llama/Qwen/Mistral 等）
    #     # 注意：不同模型的 tokenizer 会自动使用各自的特殊 token
    #     text = self.tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=False
    #     )
        
    #     # Tokenize
    #     encoding = self.tokenizer(
    #         text,
    #         truncation=True,
    #         max_length=self.max_length,
    #         padding="max_length",
    #         return_tensors="pt"
    #     )
        
    #     input_ids = encoding["input_ids"].squeeze(0)
    #     attention_mask = encoding["attention_mask"].squeeze(0)
        
    #     # 对于 Causal LM，labels 就是 input_ids
    #     # 但我们需要将 padding 部分的 label 设为 -100（忽略 loss）
    #     labels = input_ids.clone()
    #     labels[attention_mask == 0] = -100
        
    #     return {
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask,
    #         "labels": labels
    #     }
    # ✅ 正确的访问方式
    def __getitem__(self, idx):
        session = self.trajectories[idx]
        
        # 将 ChatHistory 转换为标准的 messages 格式
        messages = []
        for item_index in range(session.chat_history.get_value_length()):
            item = session.chat_history.get_item_deep_copy(item_index)
            # 统一角色名称：agent -> assistant
            role = "assistant" if item.role.value == "agent" else item.role.value
            messages.append({
                "role": role,
                "content": item.content
            })
        
        # 使用 tokenizer 的聊天模板（自动适配 Llama/Qwen 等）
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception as e:
            # 如果 apply_chat_template 失败，使用简单拼接
            logger.warning(f"apply_chat_template 失败，使用简单格式: {e}")
            text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        # ... 后续代码保持不变
        # Tokenize
        # ✅ 关键修复：确保 tokenizer 有 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # 对于 Causal LM，labels 就是 input_ids
        # 但我们需要将 padding 部分的 label 设为 -100（忽略 loss）
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    


class LossLoggingCallback(TrainerCallback):
    """记录训练 loss 到 CSV 文件"""
    def __init__(self, loss_log_path: str, batch_id: int):
        self.loss_log_path = loss_log_path
        self.batch_id = batch_id
        
        # 如果是第一个 batch，写入 header
        if not os.path.exists(self.loss_log_path):
            os.makedirs(os.path.dirname(self.loss_log_path), exist_ok=True)
            with open(self.loss_log_path, "w") as f:
                f.write("batch_id,step,loss,learning_rate,timestamp\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            with open(self.loss_log_path, "a") as f:
                step = state.global_step
                loss = logs['loss']
                lr = logs.get('learning_rate', 'N/A')
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{self.batch_id},{step},{loss},{lr},{timestamp}\n")


class TestTimeTrainingCallback(Callback):
    """
    Test-Time Training with LoRA
    
    特性：
    1. 模型无关：兼容 Llama, Qwen, Mistral 等所有 HuggingFace 模型
    2. 渐进式学习：LoRA 权重持续累积更新
    3. 完整可追溯：每个 batch 的数据和 loss 都单独保存
    4. 状态可恢复：支持实验中断后继续
    """
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
    ):
        super().__init__()
        self.batch_size = batch_size
        # 保存原始路径模板（带占位符）
        self.sft_data_dir_template = sft_data_dir
        self.loss_log_path_template = loss_log_path
        # 实际路径将在首次运行时初始化
        self.sft_data_dir = None
        self.loss_log_path = None
        self.trainer_output_dir = None  # 新增：Trainer临时目录
        self.max_seq_length = max_seq_length
        
        # LoRA 配置
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # 如果未指定，使用通用的 target_modules（适配大多数模型）
        self.lora_target_modules = lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"  # 适配 Llama/Qwen 的 MLP
        ]
        
        # 训练配置
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # 内部状态
        self.successful_trajectories: List[Session] = []
        self.model_ref = None
        self.tokenizer_ref = None
        self.is_lora_applied = False
        self.training_batch_count = 0
        self.paths_initialized = False  # ✅ 添加这一行！

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        """在每个样本完成后，检查是否成功并收集轨迹"""
        session = callback_args.current_session
        
        # 只收集成功的轨迹
        if (
            session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT and
            session.sample_status == SampleStatus.COMPLETED
        ):
            print(f"✅ [TTT] 成功轨迹收集: sample_index={session.sample_index}")
            self.successful_trajectories.append(session.model_copy(deep=True))
            
            # 检查是否达到 batch_size
            if len(self.successful_trajectories) >= self.batch_size:
                print(f"\n{'='*60}")
                print(f"🎯 [TTT] 已收集 {len(self.successful_trajectories)} 条成功轨迹")
                print(f"{'='*60}\n")
                self._run_training(callback_args)

    # def _run_training(self, callback_args: CallbackArguments):
    #     """执行一次 LoRA 微调"""
    #     print(f"🚀 [TTT] 开始第 {self.training_batch_count + 1} 轮 LoRA 训练...")
        
    #     # 1. 获取模型和 tokenizer 引用
    #     if not self._initialize_model_refs(callback_args):
    #         return
        
    #     # 2. 首次应用 LoRA（只在第一次训练时）
    #     if not self.is_lora_applied:
    #         self._apply_lora_to_model()
        
    #     # 3. 保存当前 batch 的训练数据
    #     self._save_sft_batch_data()
        
    #     # 4. 准备数据集
    #     train_dataset = SftTrajectoryDataset(
    #         self.successful_trajectories,
    #         self.tokenizer_ref,
    #         max_length=self.max_seq_length
    #     )
        
    #     # 5. 配置训练参数
    #     training_args = TrainingArguments(
    #         output_dir=os.path.join(
    #             os.path.dirname(self.loss_log_path),
    #             f"ttt_trainer_batch_{self.training_batch_count}"
    #         ),
    #         num_train_epochs=self.num_train_epochs,
    #         per_device_train_batch_size=self.per_device_train_batch_size,
    #         gradient_accumulation_steps=self.gradient_accumulation_steps,
    #         learning_rate=self.learning_rate,
    #         logging_steps=1,
    #         save_strategy="no",
    #         report_to="none",
    #         remove_unused_columns=False,
    #     )
        
    #     # 6. 创建 Trainer 并注入 loss 日志回调
    #     loss_logger = LossLoggingCallback(self.loss_log_path, self.training_batch_count)
    #     trainer = Trainer(
    #         model=self.model_ref,
    #         args=training_args,
    #         train_dataset=train_dataset,
    #         callbacks=[loss_logger]
    #     )
        
    #     # 7. 开始训练
    #     print(f"🏋️  [TTT] 训练中...")
    #     train_result = trainer.train()
    #     print(f"✅ [TTT] 训练完成! Loss: {train_result.training_loss:.4f}")
    #     print(f"📊 [TTT] Loss 已记录至: {self.loss_log_path}")
        
    #     # 8. 清理当前 batch，准备下一轮
    #     self.successful_trajectories = []
    #     self.training_batch_count += 1
    #     print(f"🔄 [TTT] 清理完成，继续收集下一批轨迹\n")
    def _run_training(self, callback_args: CallbackArguments):
        """执行一次 LoRA 微调"""
        # 首先初始化路径
        self._initialize_paths()
        
        print(f"🚀 [TTT] 开始第 {self.training_batch_count + 1} 轮 LoRA 训练...")
        
        # 1. 获取模型和 tokenizer 引用
        if not self._initialize_model_refs(callback_args):
            return
        
        # 2. 首次应用 LoRA（只在第一次训练时）
        if not self.is_lora_applied:
            self._apply_lora_to_model()
        
        # 3. 保存当前 batch 的训练数据
        self._save_sft_batch_data()
        
        # 4. 准备数据集
        train_dataset = SftTrajectoryDataset(
            self.successful_trajectories,
            self.tokenizer_ref,
            max_length=self.max_seq_length
        )
        
        # 5. 配置训练参数
        training_args = TrainingArguments(
            output_dir=self.trainer_output_dir,  # ✅ 所有batch共用一个目录
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            overwrite_output_dir=True,  # ✅ 允许覆盖（因为共用目录）
        )
        
        # 6. 创建 Trainer 并注入 loss 日志回调
        loss_logger = LossLoggingCallback(self.loss_log_path, self.training_batch_count)
        trainer = Trainer(
            model=self.model_ref,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=[loss_logger]
        )
        
        # 7. 开始训练
        print(f"🏋️  [TTT] 训练中...")
        train_result = trainer.train()
        print(f"✅ [TTT] 训练完成! Loss: {train_result.training_loss:.4f}")
        print(f"📊 [TTT] Loss 已记录至: {self.loss_log_path}")
        
        # 8. 清理当前 batch，准备下一轮
        self.successful_trajectories = []
        self.training_batch_count += 1
        print(f"🔄 [TTT] 清理完成，继续收集下一批轨迹\n")
        
    def _initialize_paths(self):
        """初始化实际的输出路径（从state_dir提取）"""
        if self.paths_initialized:
            return
        
        # 从 state_dir 提取实际的 output_dir
        # state_dir 格式: outputs/2025-10-14-13-24-48/callback_state/callback_3
        state_dir = self.get_state_dir()
        output_dir = os.path.dirname(os.path.dirname(state_dir))  # 向上两级
        
        # 替换占位符，使用实际的时间戳目录
        self.sft_data_dir = os.path.join(output_dir, "sft_data")
        self.loss_log_path = os.path.join(output_dir, "loss_log.csv")
        # 所有batch共用一个临时目录（避免产生大量空目录）
        self.trainer_output_dir = os.path.join(output_dir, ".ttt_trainer_temp")
        
        # 创建必要的目录
        os.makedirs(self.sft_data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.loss_log_path), exist_ok=True)
        os.makedirs(self.trainer_output_dir, exist_ok=True)
        
        print(f"📁 [TTT] 输出目录已初始化:")
        print(f"   - SFT数据: {self.sft_data_dir}")
        print(f"   - Loss日志: {self.loss_log_path}")
        print(f"   - Trainer临时目录: {self.trainer_output_dir}")
        
        self.paths_initialized = True

    def _initialize_model_refs(self, callback_args: CallbackArguments) -> bool:
        """初始化模型和 tokenizer 的引用"""
        if self.model_ref is not None:
            return True
        
        agent = callback_args.session_context.agent
        if not hasattr(agent, "_language_model"):
            print("⚠️  [TTT] Agent 没有 _language_model 属性，跳过训练")
            return False
        
        if not isinstance(agent._language_model, HuggingfaceLanguageModel):
            print("⚠️  [TTT] 只支持 HuggingfaceLanguageModel，跳过训练")
            return False
        
        self.model_ref = agent._language_model.model
        self.tokenizer_ref = agent._language_model.tokenizer
        
        # 检测模型类型（用于日志）
        model_type = self.model_ref.config.model_type
        print(f"🔍 [TTT] 检测到模型类型: {model_type}")
        
        return True

    def _apply_lora_to_model(self):
        """首次应用 LoRA 适配器到模型"""
        print(f"\n{'='*60}")
        print(f"🔧 [TTT] 首次应用 LoRA 适配器")
        print(f"{'='*60}")
        
        # 创建 LoRA 配置
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 应用 LoRA
        self.model_ref = get_peft_model(self.model_ref, lora_config)
        
        # 打印可训练参数统计
        self.model_ref.print_trainable_parameters()
        
        # 关键：更新 agent 中的模型引用
        # 这样后续的推理会使用带 LoRA 的模型
        agent = self.model_ref  # 已经是 PeftModel 了
        
        self.is_lora_applied = True
        print(f"✅ [TTT] LoRA 适配器已成功应用\n")

    def _save_sft_batch_data(self):
        """保存当前 batch 的 SFT 数据"""
        batch_file = os.path.join(
            self.sft_data_dir,
            f"batch_{self.training_batch_count:03d}.json"
        )
        os.makedirs(os.path.dirname(batch_file), exist_ok=True)
        
        # 保存完整的 Session 数据
        batch_data = {
            "batch_id": self.training_batch_count,
            "sample_count": len(self.successful_trajectories),
            "sample_indices": [s.sample_index for s in self.successful_trajectories],
            "trajectories": [s.model_dump() for s in self.successful_trajectories]
        }
        
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 [TTT] Batch {self.training_batch_count} 数据已保存: {batch_file}")

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        """保存回调状态（支持断点恢复）"""
        state = {
            "training_batch_count": self.training_batch_count,
            "is_lora_applied": self.is_lora_applied,
            "successful_trajectories": [s.model_dump() for s in self.successful_trajectories]
        }
        state_path = os.path.join(self.get_state_dir(), "ttt_state.json")
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

    # def restore_state(self) -> None:
    #     """恢复回调状态"""
    #     state_path = os.path.join(self.get_state_dir(), "ttt_state.json")
    #     if os.path.exists(state_path):
    #         with open(state_path, 'r') as f:
    #             state = json.load(f)
    #         self.training_batch_count = state.get("training_batch_count", 0)
    #         self.is_lora_applied = state.get("is_lora_applied", False)
    #         self.successful_trajectories = [
    #             Session.model_validate(s)
    #             for s in state.get("successful_trajectories", [])
    #         ]
    #         print(f"🔄 [TTT] 状态已恢复: 已完成 {self.training_batch_count} 轮训练，"
    #               f"当前收集 {len(self.successful_trajectories)} 条轨迹")
    def restore_state(self) -> None:
        """恢复回调状态"""
        # 先初始化路径
        self._initialize_paths()
        
        state_path = os.path.join(self.get_state_dir(), "ttt_state.json")
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
            self.training_batch_count = state.get("training_batch_count", 0)
            self.is_lora_applied = state.get("is_lora_applied", False)
            self.successful_trajectories = [
                Session.model_validate(s)
                for s in state.get("successful_trajectories", [])
            ]
            print(f"🔄 [TTT] 状态已恢复: 已完成 {self.training_batch_count} 轮训练，"
                f"当前收集 {len(self.successful_trajectories)} 条轨迹")