"""
Workflow Memory Callback - 基于 AWM Online 方法的工作流记忆机制
参考: Agent Workflow Memory (https://arxiv.org/abs/2409.07429)

核心特性：
- 使用 Agent 当前的 LanguageModel 进行 workflow 归纳（自归纳）
- 自动收集成功样本并定期归纳
- 将归纳的 workflow 注入到后续任务的 prompt 中
- 实时打印归纳的 workflows 到控制台
"""

from typing import Optional, List
from abc import ABC, abstractmethod
import os
import json

from src.callbacks import Callback, CallbackArguments
from src.typings import (
    Session, 
    SampleStatus, 
    SessionEvaluationOutcome, 
    ChatHistoryItem, 
    Role,
    ChatHistory,
)
from src.language_models import LanguageModel
from src.utils import SafeLogger


class WorkflowMemoryCallback(Callback, ABC):
    """
    抽象基类：实现 AWM Online 的工作流归纳和利用机制
    
    核心流程 (与 AWM 原始实现一致):
    1. 收集成功的执行轨迹
    2. 每 N 个样本后使用 Agent 的模型归纳 workflow
    3. 将 workflow 注入到后续任务的 prompt 中
    """
    
    def __init__(
        self,
        induction_frequency: int = 5,      # 每 N 个样本归纳一次
        max_workflows: int = 10,            # 最多保留的 workflow 数量
        min_success_samples: int = 2,       # 最少需要的成功样本数
        max_examples_for_induction: int = 10,  # 用于归纳的最大示例数
        workflow_file_name: str = "workflows.txt",
        temperature: float = 0.0,           # 向后兼容，仍然支持单独的 temperature
        use_previous_workflows: bool = False,  # AWM 不使用之前的 workflows
        instruction_file: Optional[str] = None,  # 归纳指令文件路径
        one_shot_file: Optional[str] = None,     # One-shot 示例文件路径
        inference_config_dict: Optional[dict] = None,  # 完整的推理配置（优先级更高）
    ):
        super().__init__()
        self.induction_frequency = induction_frequency
        self.max_workflows = max_workflows
        self.min_success_samples = min_success_samples
        self.max_examples_for_induction = max_examples_for_induction
        self.workflow_file_name = workflow_file_name
        self.temperature = temperature
        self.use_previous_workflows = use_previous_workflows
        
        # 如果提供了完整的 inference_config_dict，使用它；否则根据 temperature 构建
        if inference_config_dict is not None:
            self.inference_config_dict = inference_config_dict
        else:
            # 向后兼容：根据 temperature 自动构建配置
            if temperature == 0.0:
                self.inference_config_dict = {
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": 2048,
                }
            else:
                self.inference_config_dict = {
                    "do_sample": True,
                    "temperature": temperature,
                    "max_new_tokens": 2048,
                }
        self.instruction_file = instruction_file
        self.one_shot_file = one_shot_file
        
        # 状态变量
        self.successful_sessions: List[Session] = []
        self.workflows: List[str] = []
        self.processed_count = 0
        self.induction_count = 0
        
        # LanguageModel 将在运行时由 Agent 提供
        self._language_model: Optional[LanguageModel] = None
        
    @classmethod
    def is_unique(cls) -> bool:
        return True
    
    def restore_state(self) -> None:
        """从保存的状态中恢复"""
        state_file = os.path.join(self.get_state_dir(), "workflow_memory_state.json")
        if os.path.exists(state_file):
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                self.processed_count = state.get("processed_count", 0)
                self.induction_count = state.get("induction_count", 0)
                self.workflows = state.get("workflows", [])
                SafeLogger.info(
                    f"[WorkflowMemory] 恢复状态: {self.processed_count} 个已处理样本, "
                    f"{len(self.workflows)} 个 workflows"
                )
        
        workflow_file = self._get_workflow_file_path()
        if os.path.exists(workflow_file):
            with open(workflow_file, 'r', encoding='utf-8') as f:
                content = f.read()
                self.workflows = [wf.strip() for wf in content.split('\n\n') if wf.strip()]
    
    def on_task_reset(self, callback_args: CallbackArguments) -> None:
        """
        任务重置时：
        1. 从 Agent 获取 LanguageModel
        2. 将 workflows 注入到第一条 USER 消息（系统指令）中
        """
        agent = callback_args.session_context.agent
        
        # 1. 获取 LanguageModel
        if hasattr(agent, '_language_model'):
            self._language_model = agent._language_model
            SafeLogger.debug(
                f"[WorkflowMemory] 获取到 Agent 的模型引用: "
                f"{type(self._language_model).__name__}"
            )
        elif hasattr(agent, 'language_model'):
            self._language_model = agent.language_model
            SafeLogger.debug(
                f"[WorkflowMemory] 获取到 Agent 的模型引用: "
                f"{type(self._language_model).__name__}"
            )
        else:
            SafeLogger.warning("[WorkflowMemory] Agent 没有 language_model 或 _language_model 属性")
        
        # 2. 注入 workflows 到第一条 USER 消息末尾
        if len(self.workflows) > 0:
            task = callback_args.session_context.task
            
            # 获取当前的第一条 USER 消息
            try:
                current_first_prompt = task.chat_history_item_factory.construct(0, Role.USER).content
            except Exception as e:
                SafeLogger.warning(f"[WorkflowMemory] 无法获取第一条 USER 消息: {e}")
                return
            
            # 定义 workflow section 的标记
            workflow_marker = "\n\n" + "=" * 80 + "\n" + "Here are some useful skills abstracted from previous successful trajectories:"
            
            # 移除旧的 workflow section（如果存在）
            if workflow_marker in current_first_prompt:
                original_first_prompt = current_first_prompt.split(workflow_marker)[0]
                SafeLogger.debug(f"[WorkflowMemory] 检测到旧的 workflow section，已移除")
            else:
                original_first_prompt = current_first_prompt
            
            # 构建新的 workflow 提示（添加到系统指令末尾）
            workflow_section = "\n\n" + "=" * 80 + "\n"
            workflow_section += "Here are some useful skills abstracted from previous successful trajectories:\n"
            workflow_section += "You can refer to these patterns when solving similar problems.\n\n"
            
            # 添加每个 workflow
            for workflow in self.workflows:
                workflow_section += workflow + "\n\n"
            
            workflow_section += "=" * 80
            
            # 构建增强的 prompt（workflows 在末尾）
            enhanced_prompt = original_first_prompt + workflow_section
            
            # 更新第一条 USER 消息
            task.chat_history_item_factory.set(0, Role.USER, enhanced_prompt)
            
            SafeLogger.info(
                f"[WorkflowMemory] ✅ 更新系统指令中的 workflows (共 {len(self.workflows)} 个) "
                f"(样本 {callback_args.current_session.sample_index}), "
                f"原始: {len(original_first_prompt)} 字符 → 增强后: {len(enhanced_prompt)} 字符"
            )
    
    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        """收集成功样本并检查是否需要归纳"""
        session = callback_args.current_session
        
        # 只收集成功的样本 (与 AWM 一致)
        if (session.sample_status == SampleStatus.COMPLETED and 
            session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT):
            self.successful_sessions.append(session.model_copy(deep=True))
            SafeLogger.info(
                f"[WorkflowMemory] 收集成功样本 {session.sample_index}, "
                f"成功样本数: {len(self.successful_sessions)}"
            )
        
        self.processed_count += 1
        
        # 检查是否需要归纳
        should_induce = (
            self.processed_count % self.induction_frequency == 0 and 
            len(self.successful_sessions) >= self.min_success_samples and
            self._language_model is not None
        )
        
        if should_induce:
            SafeLogger.info(f"[WorkflowMemory] 开始第 {self.induction_count + 1} 次归纳...")
            self._induce_workflows()
    
    def on_agent_inference(self, callback_args: CallbackArguments) -> None:
        """
        注入到第一条 USER 消息后，这个方法就不需要了
        workflows 已经在 on_task_reset 中注入到系统指令了
        """
        pass
    
    def on_state_save(self, callback_args: CallbackArguments) -> None:
        """保存状态"""
        state_file = os.path.join(self.get_state_dir(), "workflow_memory_state.json")
        state = {
            "processed_count": self.processed_count,
            "induction_count": self.induction_count,
            "workflows": self.workflows,
            "successful_count": len(self.successful_sessions),
        }
        
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        if len(self.workflows) > 0:
            workflow_file = self._get_workflow_file_path()
            os.makedirs(os.path.dirname(workflow_file), exist_ok=True)
            with open(workflow_file, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(self.workflows))
    
    def _induce_workflows(self) -> None:
        """调用 LLM 归纳 workflows (遵循 AWM 的方式)"""
        if self._language_model is None:
            SafeLogger.error("[WorkflowMemory] 模型未设置")
            return
        
        try:
            # 选择示例
            examples = self.successful_sessions[-self.max_examples_for_induction:]
            SafeLogger.info(f"[WorkflowMemory] 使用 {len(examples)} 个样本归纳")
            
            # 格式化示例 (子类实现)
            formatted = self._format_successful_sessions(examples)
            
            # 构造 prompt (遵循 AWM 的结构)
            prompt = self._build_awm_style_prompt(formatted)
            
            # 调用 LLM
            chat_history = ChatHistory()
            chat_history.inject(ChatHistoryItem(role=Role.USER, content=prompt))
            
            SafeLogger.info(
                f"[WorkflowMemory] 调用 LLM 归纳 "
                f"(config={self.inference_config_dict})..."
            )
            
            response = self._language_model.inference(
                batch_chat_history=[chat_history],
                inference_config_dict=self.inference_config_dict,
                system_prompt="You are an expert at extracting common patterns from task executions."
            )[0]
            
            # 解析 workflows
            new_workflows = self._parse_workflows(response.content)
            
            if len(new_workflows) > 0:
                self.workflows.extend(new_workflows)
                if len(self.workflows) > self.max_workflows:
                    self.workflows = self.workflows[-self.max_workflows:]
                
                SafeLogger.info(
                    f"[WorkflowMemory] 归纳得到 {len(new_workflows)} 个新 workflows, "
                    f"总数: {len(self.workflows)}"
                )
                
                # 🎯 打印新归纳的 workflows 到控制台
                self._print_workflows_to_console(new_workflows)
                
            else:
                SafeLogger.warning("[WorkflowMemory] LLM 没有生成任何 workflow")
            
            self.induction_count += 1
            
            # 保留部分样本
            keep = min(self.induction_frequency, len(self.successful_sessions))
            self.successful_sessions = self.successful_sessions[-keep:]
            
        except Exception as e:
            SafeLogger.error(f"[WorkflowMemory] 归纳失败: {e}", exc_info=True)
    
    def _print_workflows_to_console(self, workflows: List[str]) -> None:
        """
        打印新归纳的 workflows 到控制台
        使用漂亮的格式方便阅读
        """
        separator = "=" * 80
        SafeLogger.info(f"\n{separator}")
        SafeLogger.info(f"🎯 第 {self.induction_count + 1} 次归纳 - 新生成的 Workflows:")
        SafeLogger.info(separator)
        
        for idx, workflow in enumerate(workflows, 1):
            SafeLogger.info(f"\n📋 Workflow {idx}:")
            SafeLogger.info("-" * 80)
            # 逐行打印 workflow，保持格式
            for line in workflow.split('\n'):
                SafeLogger.info(f"  {line}")
            SafeLogger.info("")
        
        SafeLogger.info(separator)
        SafeLogger.info(f"✅ 本次归纳完成！当前共有 {len(self.workflows)} 个 workflows")
        SafeLogger.info(f"{separator}\n")
    
    def _build_awm_style_prompt(self, formatted_examples: str) -> str:
        """
        构造 AWM 风格的 prompt
        结构: INSTRUCTION + ONE_SHOT + formatted_examples + "# Summary Workflows"
        """
        components = []
        
        # 1. INSTRUCTION
        instruction = self._load_instruction()
        components.append(instruction)
        
        # 2. ONE_SHOT (可选)
        one_shot = self._load_one_shot()
        if one_shot:
            components.append(one_shot)
        
        # 3. Formatted Examples
        components.append(formatted_examples)
        
        # 4. Summary marker
        components.append("# Summary Workflows")
        
        return '\n\n'.join(components)
    
    def _load_instruction(self) -> str:
        """加载归纳指令"""
        if self.instruction_file and os.path.exists(self.instruction_file):
            with open(self.instruction_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        return self._get_default_instruction()
    
    def _load_one_shot(self) -> str:
        """加载 one-shot 示例"""
        if self.one_shot_file and os.path.exists(self.one_shot_file):
            with open(self.one_shot_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        return ""
    
    # ========== 子类必须实现的抽象方法 ==========
    
    @abstractmethod
    def _format_successful_sessions(self, sessions: List[Session]) -> str:
        """格式化成功的会话为 AWM 风格的示例"""
        raise NotImplementedError()
    
    @abstractmethod
    def _parse_workflows(self, llm_response: str) -> List[str]:
        """解析 LLM 响应中的 workflows"""
        raise NotImplementedError()
    
    @abstractmethod
    def _format_workflows_for_prompt(self) -> str:
        """格式化 workflows 用于注入到 Agent prompt"""
        raise NotImplementedError()
    
    @abstractmethod
    def _get_default_instruction(self) -> str:
        """获取默认的归纳指令 (如果文件不存在)"""
        raise NotImplementedError()
    
    # ========== 辅助方法 ==========
    
    def _get_workflow_file_path(self) -> str:
        return os.path.join(self.get_state_dir(), self.workflow_file_name)

