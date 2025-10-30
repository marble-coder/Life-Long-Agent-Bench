"""
DB Bench Workflow Memory Callback - DB Bench 任务的工作流记忆实现
完全对齐 AWM (Agent Workflow Memory) 的格式和风格
"""

from typing import List
import re

from src.callbacks.instance.workflow_memory_callback import WorkflowMemoryCallback
from src.typings import Session, Role
from src.utils import SafeLogger


class DBBenchWorkflowMemoryCallback(WorkflowMemoryCallback):
    """
    DB Bench 任务的 Workflow Memory 实现
    
    遵循 AWM 的设计原则：
    1. 从成功的执行轨迹中提取通用 SQL 模式
    2. 使用变量名替换具体的表名、列名、值
    3. 每个 workflow 至少包含 2 个步骤
    4. 避免生成重复或重叠的 workflows
    
    示例输出：
    # Workflow 1: Filter records by single condition
    1. SELECT * FROM <table_name> WHERE <column_name> = '<value>'
    2. Verify the result contains expected records
    """
    
    def _format_successful_sessions(self, sessions: List[Session]) -> str:
        """
        格式化成功的 DB Bench 会话为 AWM 风格的示例
        
        输出格式 (遵循 AWM):
        ## Concrete Examples
        
        ## Example 1
        Task: <natural language task description>
        Trajectory:
        Step 1: <SQL query>
        Observation: <execution result>
        Step 2: <SQL query>
        Observation: <execution result>
        Final Answer: <final correct SQL>
        
        Args:
            sessions: 成功的会话列表
            
        Returns:
            格式化的示例字符串
        """
        formatted_parts = ["## Concrete Examples"]
        
        for idx, session in enumerate(sessions, 1):
            example_parts = [f"\n## Example {idx}"]
            
            # 1. 提取任务描述
            task_desc = self._extract_task_description(session)
            if task_desc:
                example_parts.append(f"Task: {task_desc}")
            else:
                example_parts.append(f"Task: Database query task {idx}")
            
            # 2. 提取执行轨迹
            example_parts.append("Trajectory:")
            trajectory_steps = self._extract_trajectory(session)
            
            if trajectory_steps:
                example_parts.extend(trajectory_steps)
            else:
                # 如果没有轨迹，至少添加最终答案
                SafeLogger.warning(f"[DBBenchWorkflow] Example {idx} 没有提取到轨迹")
            
            # 3. 最终答案（正确的 SQL）
            final_sql = self._extract_final_answer(session)
            if final_sql:
                example_parts.append(f"Final Answer: {final_sql}")
            
            formatted_parts.append('\n'.join(example_parts))
        
        result = '\n'.join(formatted_parts)
        SafeLogger.debug(f"[DBBenchWorkflow] 格式化了 {len(sessions)} 个成功会话")
        return result
    
    def _extract_task_description(self, session: Session) -> str:
        """
        从会话的第一条 USER 消息中提取任务描述
        
        Args:
            session: 会话对象
            
        Returns:
            任务描述字符串
        """
        if session.chat_history.get_value_length() == 0:
            return ""
        
        first_message = session.chat_history.get_item_deep_copy(0)
        if first_message.role != Role.USER:
            return ""
        
        content = first_message.content
        
        # 尝试多种模式提取任务描述
        patterns = [
            # 明确的标记
            r'(?:Task|Question|Query|Instruction):\s*(.+?)(?:\n|$)',
            # 常见的动词开头
            r'(?:Please|Write|Generate|Find|List|Get|Show|Count|Select|Create|Update|Delete)\s+(.+?)(?:\n|$)',
            # SQL 相关的问题
            r'(?:What|Which|How many|Who|When|Where)\s+(.+?)(?:\?|\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                desc = match.group(1).strip()
                # 清理：压缩空白，移除多余字符
                desc = ' '.join(desc.split())
                desc = desc.rstrip('?!.,;')
                # 限制长度
                if len(desc) > 200:
                    desc = desc[:197] + "..."
                return desc
        
        # 如果没有匹配到，返回第一行
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        if lines:
            desc = lines[0]
            desc = desc.rstrip('?!.,;')
            if len(desc) > 200:
                desc = desc[:197] + "..."
            return desc
        
        return ""
    
    def _extract_trajectory(self, session: Session) -> List[str]:
        """
        提取执行轨迹（Agent 的 SQL 查询和环境的观察）
        
        返回格式:
        [
            "Step 1: SELECT * FROM ...",
            "Observation: 5 rows returned",
            "Step 2: UPDATE ...",
            "Observation: Success",
        ]
        
        Args:
            session: 会话对象
            
        Returns:
            轨迹步骤列表
        """
        trajectory_steps = []
        step_num = 1
        
        chat_history = session.chat_history
        
        # 从第二条消息开始（第一条是任务描述）
        for i in range(1, chat_history.get_value_length()):
            item = chat_history.get_item_deep_copy(i)
            
            if item.role == Role.AGENT:
                # Agent 的动作（SQL 查询）
                sql_query = self._extract_sql_from_agent_response(item.content)
                if sql_query:
                    trajectory_steps.append(f"Step {step_num}: {sql_query}")
                else:
                    # 如果没有提取到 SQL，可能是其他类型的响应
                    SafeLogger.debug(f"[DBBenchWorkflow] 未从 Agent 响应中提取到 SQL")
                    
            elif item.role == Role.USER:
                # 环境的观察（执行结果）
                observation = self._extract_observation_from_env(item.content)
                if observation:
                    # 只在有对应 Step 的情况下添加 Observation
                    if len(trajectory_steps) > 0 and trajectory_steps[-1].startswith(f"Step {step_num}"):
                        trajectory_steps.append(f"Observation: {observation}")
                        step_num += 1
        
        return trajectory_steps
    
    def _extract_sql_from_agent_response(self, content: str) -> str:
        """
        从 Agent 响应中提取 SQL 查询
        
        支持的格式:
        - ```sql ... ```
        - ``` ... ```
        - SQL: ...
        - 直接的 SQL 语句（SELECT, INSERT, UPDATE, DELETE, ...）
        
        Args:
            content: Agent 响应内容
            
        Returns:
            提取的 SQL 语句
        """
        # 尝试多种模式提取 SQL
        patterns = [
            # Markdown 代码块（优先级最高）
            (r'```sql\s*\n(.+?)\n```', re.DOTALL | re.IGNORECASE),
            (r'```\s*\n(.+?)\n```', re.DOTALL | re.IGNORECASE),
            # 明确的 SQL 标记
            (r'SQL:\s*(.+?)(?:\n\n|\n(?=[A-Z][a-z]+:)|$)', re.DOTALL | re.IGNORECASE),
            (r'Query:\s*(.+?)(?:\n\n|\n(?=[A-Z][a-z]+:)|$)', re.DOTALL | re.IGNORECASE),
            # 直接的 SQL 语句（按常见程度排序）
            (r'(SELECT\s+.+?)(?:\n\n|\n(?=[A-Z][a-z]+:)|;(?:\n|$)|$)', re.DOTALL | re.IGNORECASE),
            (r'(INSERT\s+INTO\s+.+?)(?:\n\n|\n(?=[A-Z][a-z]+:)|;(?:\n|$)|$)', re.DOTALL | re.IGNORECASE),
            (r'(UPDATE\s+.+?\s+SET\s+.+?)(?:\n\n|\n(?=[A-Z][a-z]+:)|;(?:\n|$)|$)', re.DOTALL | re.IGNORECASE),
            (r'(DELETE\s+FROM\s+.+?)(?:\n\n|\n(?=[A-Z][a-z]+:)|;(?:\n|$)|$)', re.DOTALL | re.IGNORECASE),
            (r'(CREATE\s+TABLE\s+.+?)(?:\n\n|\n(?=[A-Z][a-z]+:)|;(?:\n|$)|$)', re.DOTALL | re.IGNORECASE),
            (r'(DROP\s+TABLE\s+.+?)(?:\n\n|\n(?=[A-Z][a-z]+:)|;(?:\n|$)|$)', re.DOTALL | re.IGNORECASE),
        ]
        
        for pattern, flags in patterns:
            match = re.search(pattern, content, flags)
            if match:
                sql = match.group(1).strip()
                # 清理 SQL
                sql = self._clean_sql(sql)
                return sql
        
        return ""
    
    def _clean_sql(self, sql: str) -> str:
        """
        清理和规范化 SQL 语句
        
        Args:
            sql: 原始 SQL 语句
            
        Returns:
            清理后的 SQL 语句
        """
        # 去除多余的空白（保留单个空格）
        sql = ' '.join(sql.split())
        
        # 去除末尾的分号
        sql = sql.rstrip(';').strip()
        
        # 限制长度
        if len(sql) > 500:
            sql = sql[:497] + "..."
        
        return sql
    
    def _extract_observation_from_env(self, content: str) -> str:
        """
        从环境消息（USER）中提取观察结果
        
        常见情况:
        - 错误消息
        - 空结果
        - 返回的行数
        - 成功消息
        
        Args:
            content: 环境响应内容
            
        Returns:
            简化的观察结果
        """
        content_lower = content.lower()
        
        # 1. 检查错误
        error_keywords = ['error', 'fail', 'exception', 'invalid', 'syntax error', 'not found']
        if any(keyword in content_lower for keyword in error_keywords):
            # 尝试提取具体的错误类型
            if 'syntax error' in content_lower:
                return "Syntax error"
            elif 'not found' in content_lower:
                return "Table/column not found"
            else:
                return "Error occurred"
        
        # 2. 检查空结果
        empty_keywords = ['empty', 'no rows', '0 rows', 'no results', 'no data']
        if any(keyword in content_lower for keyword in empty_keywords):
            return "Empty result"
        
        # 3. 提取行数信息（最常见）
        row_patterns = [
            r'(\d+)\s+rows?\s+(?:returned|affected|found)',
            r'returned\s+(\d+)\s+rows?',
            r'(\d+)\s+rows?',
        ]
        for pattern in row_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                num_rows = match.group(1)
                return f"{num_rows} rows returned"
        
        # 4. 检查成功消息
        success_keywords = ['success', 'completed', 'done', 'ok', 'successfully']
        if any(keyword in content_lower for keyword in success_keywords):
            return "Success"
        
        # 5. 默认：简化长的观察结果
        simplified = content.strip().replace('\n', ' ')
        if len(simplified) > 100:
            simplified = simplified[:97] + "..."
        return simplified
    
    def _extract_final_answer(self, session: Session) -> str:
        """
        提取最终答案（正确的 SQL 查询）
        
        Args:
            session: 会话对象
            
        Returns:
            最终的 SQL 语句
        """
        # 优先从 task_output 获取
        if session.task_output and 'sql' in session.task_output:
            sql = session.task_output['sql']
            if sql:
                return self._clean_sql(sql)
        
        # 如果 task_output 没有，尝试从最后一个 Agent 响应获取
        chat_history = session.chat_history
        for i in range(chat_history.get_value_length() - 1, -1, -1):
            item = chat_history.get_item_deep_copy(i)
            if item.role == Role.AGENT:
                sql = self._extract_sql_from_agent_response(item.content)
                if sql:
                    return sql
        
        return ""
    
    def _parse_workflows(self, llm_response: str) -> List[str]:
        """
        从 LLM 响应中解析 workflows
        
        期望格式 (遵循 AWM):
        # Workflow 1: Description
        1. Step description
        2. Step description
        
        # Workflow 2: Description
        1. Step description
        ...
        
        Args:
            llm_response: LLM 的原始响应
            
        Returns:
            解析出的 workflow 列表
        """
        workflows = []
        
        # 模式1: 标准的 "# Workflow N:" 格式
        workflow_pattern = r'#\s*Workflow\s*\d+:?\s*.+?(?=\n#\s*Workflow|\Z)'
        matches = re.finditer(workflow_pattern, llm_response, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            workflow = match.group(0).strip()
            # 验证 workflow 的质量
            if self._is_valid_workflow(workflow):
                workflows.append(workflow)
        
        # 如果没有找到标准格式，尝试其他方式
        if len(workflows) == 0:
            SafeLogger.warning("[DBBenchWorkflow] 未找到标准格式的 workflows，尝试备用解析")
            
            # 尝试按双换行分割
            blocks = [b.strip() for b in llm_response.split('\n\n') if b.strip()]
            for idx, block in enumerate(blocks, 1):
                # 检查是否包含步骤编号（1. 2. 等）
                if re.search(r'^\s*\d+\.', block, re.MULTILINE):
                    if self._is_valid_workflow(block):
                        # 添加一个标题
                        workflows.append(f"# Workflow {idx}:\n{block}")
        
        SafeLogger.info(f"[DBBenchWorkflow] 解析出 {len(workflows)} 个 workflows")
        
        # 限制数量
        return workflows[:self.max_workflows]
    
    def _is_valid_workflow(self, workflow: str) -> bool:
        """
        验证 workflow 是否有效
        
        标准:
        1. 长度足够（至少 50 个字符）
        2. 包含至少 2 个步骤
        
        Args:
            workflow: workflow 字符串
            
        Returns:
            是否有效
        """
        # 检查长度
        if len(workflow) < 50:
            return False
        
        # 检查是否包含至少 2 个步骤
        step_pattern = r'^\s*\d+\.'
        steps = re.findall(step_pattern, workflow, re.MULTILINE)
        if len(steps) < 2:
            return False
        
        return True
    
    def _format_workflows_for_prompt(self) -> str:
        """
        格式化 workflows 用于注入到 Agent 的 prompt
        
        注入格式:
        # Common SQL Workflows (learned from previous tasks):
        
        <workflow 1>
        
        <workflow 2>
        
        ========================================
        
        Returns:
            格式化的 workflow 字符串
        """
        if len(self.workflows) == 0:
            return ""
        
        header = (
            "# Common SQL Workflows (learned from previous successful tasks)\n"
            "# Use these patterns as reference when solving similar problems:\n"
        )
        
        # 重新统一编号，避免序号重复
        renumbered_workflows = []
        for idx, workflow in enumerate(self.workflows, 1):
            # 移除原有的 "# Workflow N:" 标题（如果存在）
            workflow_content = re.sub(r'^#\s*Workflow\s*\d+:?\s*', '', workflow, flags=re.IGNORECASE).strip()
            # 添加新的统一编号
            renumbered_workflow = f"# Workflow {idx}:\n{workflow_content}"
            renumbered_workflows.append(renumbered_workflow)
        
        workflow_text = '\n\n'.join(renumbered_workflows)
        separator = "=" * 70
        
        return f"{header}\n{workflow_text}\n\n{separator}"
    
    def _get_default_instruction(self) -> str:
        """
        获取默认的归纳指令（如果文件不存在）
        完全遵循 AWM 的风格
        
        Returns:
            归纳指令字符串
        """
        return """Given a list of successful database task executions, extract common SQL workflows.

Each task contains a natural language question and a series of SQL queries to solve it.
You need to find the repetitive subset of SQL patterns across multiple tasks, and extract them as workflows.

Each workflow should:
1. Be a commonly-reused pattern that appears in multiple tasks
2. Contain at least 2 steps
3. Use descriptive variable names for tables, columns, and values (e.g., <table_name>, <column_name>, <value>)
4. Focus on the SQL pattern and logic flow, not specific table/column names

Format each workflow as:
# Workflow N: <Brief Description>
1. <Step 1 with generic placeholders>
2. <Step 2 with generic placeholders>
...

Example:
# Workflow 1: Filter records by single condition
1. SELECT * FROM <table_name> WHERE <column_name> = '<value>'
2. Verify the result contains expected records

# Workflow 2: Aggregate data with GROUP BY
1. SELECT <column_name>, COUNT(*) FROM <table_name> GROUP BY <column_name>
2. Analyze the grouped results to answer the question

Extract 3-5 most common and useful workflows. Do not generate similar or overlapping workflows."""

