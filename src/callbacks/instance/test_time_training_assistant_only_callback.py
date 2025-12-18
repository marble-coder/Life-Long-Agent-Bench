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
    用于 SFT 的轨迹数据集（只训练 Assistant 输出）
    兼容所有支持 apply_chat_template 的模型（Llama, Qwen, Mistral 等）
    """
    def __init__(self, trajectories: List[Session], tokenizer, max_length=2048):
        self.trajectories = trajectories
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.trajectories)

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
        
        # ✅ 关键：只训练 assistant 的输出，mask 掉 user input
        labels = input_ids.clone()
        
        # 识别哪些 token 属于 assistant 消息
        assistant_mask = self._identify_assistant_tokens(messages, input_ids)
        
        # 设置 labels：只在 assistant 部分计算 loss
        # -100 表示忽略该位置的 loss
        labels[~assistant_mask] = -100  # 非 assistant 部分设为 -100
        labels[attention_mask == 0] = -100    # padding 部分设为 -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def _identify_assistant_tokens(self, messages, full_input_ids):
        """
        识别完整序列中哪些 token 属于 assistant 消息
        返回一个与 input_ids 长度相同的布尔 mask
        """
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 初始化 mask：全为 False（表示都不是 assistant）
        assistant_mask = torch.zeros(len(full_input_ids), dtype=torch.bool)
        
        # 获取实际序列长度（去除 padding）
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        actual_length = (full_input_ids != pad_token_id).sum().item()
        if actual_length == 0:
            return assistant_mask
        
        # 方法：通过逐步构建对话来识别每个消息的token范围
        try:
            # 首先获取完整序列的token ids（用于匹配）
            full_tokens = full_input_ids[:actual_length].tolist()
            
            # 逐步构建，找到每个消息的边界
            prev_tokens = []
            
            for msg_idx in range(len(messages)):
                # 构建到当前消息为止的对话
                partial_messages = messages[:msg_idx + 1]
                
                try:
                    partial_text = self.tokenizer.apply_chat_template(
                        partial_messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                except:
                    partial_text = "\n".join([f"{m['role']}: {m['content']}" for m in partial_messages])
                
                # Tokenize 部分对话（不添加special tokens，因为完整序列已经包含了）
                partial_encoding = self.tokenizer(
                    partial_text,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=False,
                    return_tensors="pt"
                )
                partial_tokens = partial_encoding["input_ids"].squeeze(0).tolist()
                
                # 在当前消息之前的所有token
                if msg_idx == 0:
                    msg_start = 0
                else:
                    # 在完整序列中找到prev_tokens的位置
                    msg_start = self._find_token_sequence(full_tokens, prev_tokens, 0)
                    if msg_start == -1:
                        # 如果找不到，使用前一个消息的结束位置估计
                        msg_start = len(prev_tokens)
                
                # 在完整序列中找到partial_tokens的位置（从msg_start开始）
                actual_start = self._find_token_sequence(full_tokens, partial_tokens, max(0, msg_start - 10))
                if actual_start == -1:
                    # 如果找不到精确匹配，使用增量方式
                    actual_start = len(prev_tokens)
                
                msg_end = min(actual_start + len(partial_tokens), actual_length)
                msg_start = actual_start
                
                # 如果当前消息是 assistant，标记对应的 token
                if messages[msg_idx]["role"] == "assistant":
                    if msg_start < actual_length:
                        assistant_mask[msg_start:msg_end] = True
                
                # 保存当前token序列用于下一次迭代
                prev_tokens = partial_tokens[:]
                
                # 如果已经达到序列长度，提前退出
                if len(partial_tokens) >= actual_length:
                    break
                    
        except Exception as e:
            # 如果识别失败，回退到简单策略
            logger.warning(f"识别 assistant tokens 失败，使用回退策略: {e}")
            # 回退策略：通过查找assistant消息的文本模式来识别
            # 这取决于具体的tokenizer格式，但至少能处理一部分情况
            try:
                # 尝试通过直接匹配assistant消息内容来识别
                for msg in messages:
                    if msg["role"] == "assistant":
                        # Tokenize assistant消息内容
                        content_tokens = self.tokenizer.encode(
                            msg["content"],
                            add_special_tokens=False
                        )
                        # 在完整序列中查找（简单匹配）
                        # 注意：这个方法不够精确，但作为回退方案
                        for i in range(len(full_tokens) - len(content_tokens) + 1):
                            if full_tokens[i:i+len(content_tokens)] == content_tokens:
                                assistant_mask[i:i+len(content_tokens)] = True
                                break
            except:
                # 最后的回退：假设后50%是assistant
                mid_point = actual_length // 2
                assistant_mask[mid_point:actual_length] = True
        
        return assistant_mask
    
    def _find_token_sequence(self, full_tokens, sequence, start_pos=0):
        """在完整序列中查找子序列的位置"""
        if not sequence:
            return start_pos
        for i in range(start_pos, len(full_tokens) - len(sequence) + 1):
            if full_tokens[i:i+len(sequence)] == sequence:
                return i
        return -1


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


class TestTimeTrainingAssistantOnlyCallback(Callback):
    """
    Test-Time Training with LoRA (只训练 Assistant 输出)
    
    特性：
    1. 模型无关：兼容 Llama, Qwen, Mistral 等所有 HuggingFace 模型
    2. 渐进式学习：LoRA 权重持续累积更新
    3. 完整可追溯：每个 batch 的数据和 loss 都单独保存
    4. 状态可恢复：支持实验中断后继续
    5. 只训练 Assistant 输出：User input 部分不参与 loss 计算
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
        # 仅在前 100 条样本上进行 SFT
        try:
            if int(session.sample_index) >= 100:
                return
        except Exception:
            pass
        
        # 只收集成功的轨迹
        if (
            session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT and
            session.sample_status == SampleStatus.COMPLETED
        ):
            print(f"✅ [TTT-AssistantOnly] 成功轨迹收集: sample_index={session.sample_index}")
            self.successful_trajectories.append(session.model_copy(deep=True))
            
            # 检查是否达到 batch_size
            if len(self.successful_trajectories) >= self.batch_size:
                print(f"\n{'='*60}")
                print(f"🎯 [TTT-AssistantOnly] 已收集 {len(self.successful_trajectories)} 条成功轨迹")
                print(f"{'='*60}\n")
                self._run_training(callback_args)

    def _run_training(self, callback_args: CallbackArguments):
        """执行一次 LoRA 微调"""
        # 首先初始化路径
        self._initialize_paths()
        
        print(f"🚀 [TTT-AssistantOnly] 开始第 {self.training_batch_count + 1} 轮 LoRA 训练...")
        
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
        print(f"🏋️  [TTT-AssistantOnly] 训练中...")
        train_result = trainer.train()
        print(f"✅ [TTT-AssistantOnly] 训练完成! Loss: {train_result.training_loss:.4f}")
        print(f"📊 [TTT-AssistantOnly] Loss 已记录至: {self.loss_log_path}")
        
        # 8. 清理当前 batch，准备下一轮
        self.successful_trajectories = []
        self.training_batch_count += 1
        print(f"🔄 [TTT-AssistantOnly] 清理完成，继续收集下一批轨迹\n")
        
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
        
        print(f"📁 [TTT-AssistantOnly] 输出目录已初始化:")
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
            print("⚠️  [TTT-AssistantOnly] Agent 没有 _language_model 属性，跳过训练")
            return False
        
        if not isinstance(agent._language_model, HuggingfaceLanguageModel):
            print("⚠️  [TTT-AssistantOnly] 只支持 HuggingfaceLanguageModel，跳过训练")
            return False
        
        self.model_ref = agent._language_model.model
        self.tokenizer_ref = agent._language_model.tokenizer
        
        # 检测模型类型（用于日志）
        model_type = self.model_ref.config.model_type
        print(f"🔍 [TTT-AssistantOnly] 检测到模型类型: {model_type}")
        
        return True

    def _apply_lora_to_model(self):
        """首次应用 LoRA 适配器到模型"""
        print(f"\n{'='*60}")
        print(f"🔧 [TTT-AssistantOnly] 首次应用 LoRA 适配器")
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
        print(f"✅ [TTT-AssistantOnly] LoRA 适配器已成功应用\n")

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
        
        print(f"💾 [TTT-AssistantOnly] Batch {self.training_batch_count} 数据已保存: {batch_file}")

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
            print(f"🔄 [TTT-AssistantOnly] 状态已恢复: 已完成 {self.training_batch_count} 轮训练，"
                f"当前收集 {len(self.successful_trajectories)} 条轨迹")

