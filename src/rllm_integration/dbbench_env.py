import re
from typing import Any, Dict, Tuple

from rllm.environments.base.base_env import BaseEnv

from src.tasks.instance.db_bench.container import DBBenchContainer
from src.tasks.instance.db_bench.task import DirectTypeAnswerValidator


def _build_init_sql(sample: Dict[str, Any], database_name: str) -> str:
    table_info = sample["table_info"]
    column_info_list = table_info["column_info_list"]
    column_str = ",".join([f"`{c['name']}` {c['type']}" for c in column_info_list])
    column_name_str = ",".join([f"`{c['name']}`" for c in column_info_list])

    row_list = table_info["row_list"]
    item_list = []
    item_value_list = []
    for row in row_list:
        item = "("
        for value in row:
            item += "'%s',"
            if isinstance(value, str):
                value = value.replace("'", "''")
            item_value_list.append(value)
        item = item[:-1] + ")"
        item_list.append(item)
    item_str = ",".join(item_list)
    item_str = item_str % tuple(item_value_list)

    table_name = table_info["name"]
    sql = (
        f"CREATE DATABASE IF NOT EXISTS `{database_name}`;\n"
        f"USE `{database_name}`;\n"
        f"CREATE TABLE IF NOT EXISTS `{table_name}` ({column_str});\n"
        f"INSERT INTO `{table_name}` ({column_name_str}) VALUES {item_str};\n"
        f"COMMIT;\n"
    )
    return sql


def _parse_agent_response(agent_response: str) -> Tuple[str, str | None, str | None]:
    match = re.search(r"Action: (Operation|Answer)", agent_response)
    if match is None:
        return "invalid", None, 'Can not find action. Pattern: "Action: (Operation|Answer)"'
    action = match.group(1)
    if action == "Operation":
        sql_match = re.search(r"```sql\n([\s\S]*?)\n```", agent_response)
        if sql_match is None:
            return "invalid", None, 'Can not find SQL. Pattern: "```sql\\n...\\n```"'
        sql = sql_match.group(1).strip().replace("\n", " ")
        return "operation", sql, None
    if action == "Answer":
        ans_match = re.search(r"\nFinal Answer:(.*)", agent_response)
        if ans_match is None:
            return "invalid", None, "Can not find Final Answer."
        answer = ans_match.group(1).strip()
        return "answer", answer, None
    return "invalid", None, "Unexpected action."


class DBBenchEnv(BaseEnv):
    def __init__(self, sample: Dict[str, Any], max_turns: int = 3):
        self.sample = sample
        self.max_turns = max_turns
        self.step_count = 0
        self.done = False
        self.correct = False
        self.database_name = f"dbbench_{sample.get('sample_index', '0')}"
        self.container = DBBenchContainer()

    def reset(self) -> tuple[dict, dict]:
        self.step_count = 0
        self.done = False
        self.correct = False
        init_sql = _build_init_sql(self.sample, self.database_name)
        self.container.execute(init_sql)
        observation = {
            "message": self.sample["instruction"],
            "prelude": None,
        }
        return observation, {}

    def _evaluate_answer(self, answer: str) -> bool:
        answer_info = self.sample["answer_info"]
        if answer_info.get("md5"):
            return answer.strip() == answer_info["md5"]
        if answer_info.get("direct") is not None:
            gt = answer_info["direct"]
            return DirectTypeAnswerValidator.validate(answer, gt)
        return False

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        if self.done:
            return {"message": ""}, 0.0, True, {}
        self.step_count += 1
        action_type, content, reason = _parse_agent_response(str(action))
        info: dict[str, Any] = {}
        reward = 0.0  # 按整条轨迹给分，非终止步奖励为 0
        obs_msg = ""

        if action_type == "invalid":
            obs_msg = reason or "Invalid action."
            self.done = True
            reward = -1.0  # 格式错误当作错误答案
            info["finish_reason"] = reason
        elif action_type == "operation":
            obs_msg = self.container.execute(content or "", self.database_name)  # type: ignore[arg-type]
            if self.step_count >= self.max_turns:
                self.done = True
                reward = -1.0  # 超轮当作错误
                info["finish_reason"] = "MAX_TURNS"
        elif action_type == "answer":
            self.done = True
            self.correct = self._evaluate_answer(content or "")
            reward = 1.0 if self.correct else -1.0  # 终局判定
            obs_msg = "Final answer received."
            info["correct"] = self.correct
        else:
            self.done = True
            reward = -1.0  # 意外动作当作错误
            obs_msg = "Unexpected action."

        observation = {"message": obs_msg}
        return observation, reward, self.done, info

    def close(self):
        try:
            self.container.delete()
        except Exception:
            return

    @staticmethod
    def from_dict(info: dict) -> "DBBenchEnv":
        sample = info.get("sample", info)
        return DBBenchEnv(sample=sample, max_turns=info.get("max_turns", 3))

    @staticmethod
    def is_multithread_safe() -> bool:
        return True
