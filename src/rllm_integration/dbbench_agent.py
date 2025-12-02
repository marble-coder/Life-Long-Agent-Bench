import copy
import json
from typing import Any, Dict, List

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


def load_prelude_messages(path: str = "chat_history_items/standard/db_bench.json") -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("value", {})
    messages: List[Dict[str, str]] = []
    for key, item in sorted(raw.items(), key=lambda kv: int(kv[0])):
        role = item["role"]
        # normalize role name to chat template fields
        role = "assistant" if role == "agent" else role
        messages.append({"role": role, "content": item["content"]})
    return messages


class DBBenchAgent(BaseAgent):
    def __init__(self, prelude_messages: List[Dict[str, str]]):
        self.prelude_messages = prelude_messages
        self.messages: List[Dict[str, str]] = []
        self._trajectory = Trajectory()

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def reset(self):
        self.messages = []
        self._trajectory = Trajectory()

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        if not self.messages:
            self.messages.extend(copy.deepcopy(self.prelude_messages))
            self.messages.append({"role": "user", "content": observation["message"]})
        else:
            self.messages.append({"role": "user", "content": observation["message"]})

    def update_from_model(self, response: str, **kwargs) -> Action:
        self.messages.append({"role": "assistant", "content": response})
        new_step = Step(chat_completions=copy.deepcopy(self.messages))
        self._trajectory.steps.append(new_step)
        return Action(action=response)

    def get_current_state(self) -> Step | None:
        if not self._trajectory.steps:
            return None
        return self._trajectory.steps[-1]
