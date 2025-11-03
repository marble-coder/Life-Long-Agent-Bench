from __future__ import annotations

import copy
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence

import numpy as np
import yaml
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.callbacks.callback import Callback, CallbackArguments
from src.typings import (
    Session,
    SessionEvaluationOutcome,
    SampleStatus,
    Role,
    TaskName,
)
from src.utils import SafeLogger


@dataclass
class MemoryItem:
    """Structured memory item produced from a successful trajectory."""

    id: str
    title: str
    description: str
    content: str
    embedding: list[float]
    source_sample_index: str
    outcome: str = SessionEvaluationOutcome.UNKNOWN.value
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, str | list[float]]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "embedding": self.embedding,
            "source_sample_index": self.source_sample_index,
            "outcome": self.outcome,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "MemoryItem":
        return cls(
            id=str(payload["id"]),
            title=str(payload["title"]),
            description=str(payload["description"]),
            content=str(payload["content"]),
            embedding=[float(v) for v in payload["embedding"]],  # type: ignore[arg-type]
            source_sample_index=str(payload.get("source_sample_index", "")),
            outcome=str(
                payload.get("outcome", SessionEvaluationOutcome.UNKNOWN.value)
            ),
            created_at=str(payload.get("created_at", "")),
        )


@dataclass
class PromptBundle:
    system: str
    user: str


DEFAULT_PROMPT_CONFIG: dict[str, Any] = {
    "defaults": {
        "success": {
            "system": (
                "You are an expert agent coach reflecting on a successful trajectory together with prior memory context.\n"
                "Blend the new execution with the retrieved memories to create richer, reusable guidance.\n"
                "For each item, make clear what is reinforced or newly learned, why it works, when to reuse it, and any cautions.\n"
                "Keep identifiers abstract and privacy-safe.\n"
                "Output format:\n"
                "```\n"
                "# Memory Item i\n"
                "## Title <concise title>\n"
                "## Description <1-2 sentences summarizing the improvement or lesson>\n"
                "## Content\n"
                "- Key Insight: <core tactic or lesson>\n"
                "- Evidence: <salient steps or observations that justify it>\n"
                "- Reuse When: <signals or scenarios where it applies>\n"
                "- Watch Outs: <risks, limits, or follow-ups>\n"
                "- Update: <which related memory items this refines; write 'none' if brand new>\n"
                "```"
            ),
            "user": (
                "Outcome: {outcome}\n"
                "Task: {task_name}\n"
                "Related memories:\n{related_memories}\n\n"
                "Query:\n{query}\n\n"
                "Trajectory:\n{trajectory}\n"
            ),
        },
        "failure": {
            "system": (
                "You are an expert agent coach analyzing an unsuccessful trajectory alongside related memories.\n"
                "Diagnose the failure mode, how to detect it earlier, how to recover, and how to avoid repeating it.\n"
                "Spell out whether prior memories already cover this pitfall and what nuance to store now.\n"
                "Keep identifiers abstract and privacy-safe.\n"
                "Output format:\n"
                "```\n"
                "# Memory Item i\n"
                "## Title <concise title>\n"
                "## Description <1-2 sentences summarizing the root cause>\n"
                "## Content\n"
                "- Failure Mode: <what went wrong>\n"
                "- Root Cause: <underlying reasoning or assumption that failed>\n"
                "- Recovery Plan: <how to fix it when detected>\n"
                "- Prevention: <signals or guardrails to avoid it next time>\n"
                "- Update: <which related memory items this refines; write 'none' if brand new>\n"
                "```"
            ),
            "user": (
                "Outcome: {outcome}\n"
                "Task: {task_name}\n"
                "Related memories:\n{related_memories}\n\n"
                "Query:\n{query}\n\n"
                "Trajectory:\n{trajectory}\n"
            ),
        },
        "unknown": {
            "system": (
                "You are an expert agent coach reviewing a trajectory with uncertain evaluation using related memories.\n"
                "Capture promising hypotheses, open risks, and next checks so that future attempts benefit.\n"
                "Clarify what builds on existing memories and what remains unresolved.\n"
                "Keep identifiers abstract and privacy-safe.\n"
                "Output format:\n"
                "```\n"
                "# Memory Item i\n"
                "## Title <concise title>\n"
                "## Description <1-2 sentences summarizing the hypothesis>\n"
                "## Content\n"
                "- Working Theory: <current best explanation or approach>\n"
                "- Supporting Evidence: <key observations from this run>\n"
                "- Next Steps: <experiments or checks to validate it>\n"
                "- Risks: <uncertainties or failure modes to monitor>\n"
                "- Update: <which related memory items this refines; write 'none' if brand new>\n"
                "```"
            ),
            "user": (
                "Outcome: {outcome}\n"
                "Task: {task_name}\n"
                "Related memories:\n{related_memories}\n\n"
                "Query:\n{query}\n\n"
                "Trajectory:\n{trajectory}\n"
            ),
        },
    },
    "tasks": {
        TaskName.DB_BENCH.value: {
            "success": {
                "system": (
                    "You are an expert SQL tutor reflecting on a successful database reasoning trajectory with related memories.\n"
                    "Distill reusable query planning, constraint checks, and validation tactics while noting refinements to prior memories.\n"
                    "Refer to schema elements abstractly; do not expose literal table or column names.\n"
                    "Output format:\n"
                    "```\n"
                    "# Memory Item i\n"
                    "## Title <concise title>\n"
                    "## Description <1-2 sentences summarizing the insight>\n"
                    "## Content\n"
                    "- Key Insight: <core SQL tactic or reasoning pattern>\n"
                    "- Evidence: <steps or intermediate checks that proved it>\n"
                    "- Reuse When: <query shapes or question types where it helps>\n"
                    "- Watch Outs: <schema-specific traps to abstract away>\n"
                    "- Update: <which related memory items this refines; write 'none' if brand new>\n"
                    "```"
                )
            },
            "failure": {
                "system": (
                    "You are an expert SQL tutor analyzing an unsuccessful database reasoning trajectory with related memories.\n"
                    "Explain the incorrect assumptions or SQL issues, how to detect them, corrective strategies, and whether existing memories already cover them.\n"
                    "Refer to schema elements abstractly; do not expose literal table or column names.\n"
                    "Output format:\n"
                    "```\n"
                    "# Memory Item i\n"
                    "## Title <concise title>\n"
                    "## Description <1-2 sentences summarizing the pitfall>\n"
                    "## Content\n"
                    "- Failure Mode: <what went wrong>\n"
                    "- Root Cause: <why the SQL or reasoning failed>\n"
                    "- Recovery Plan: <how to correct it>\n"
                    "- Prevention: <checks to avoid repeat>\n"
                    "- Update: <which related memory items this refines; write 'none' if brand new>\n"
                    "```"
                )
            },
        }
    },
}


class TrajectoryMemoryCallback(Callback):
    """Summarize successful trajectories into reusable memory items and retrieve them for future queries."""

    MEMORY_MARKER = (
        "\n\n"
        + "=" * 80
        + "\nIntegrated Memory Items (automatically summarized from successful trajectories):\n"
    )
    
    MEMORY_END_MARKER = "\n" + "=" * 80

    def __init__(
        self,
        *,
        api_model: str = "qwen-plus",
        api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env: str = "DASHSCOPE_API_KEY",
        summary_max_items: int = 3,
        top_k: int = 3,
        max_memory_items: int = 256,
        min_similarity: float = 0.3,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_device: str = "cpu",
        inject_marker: Optional[str] = None,
        inject_end_marker: Optional[str] = None,
        prompt_config_path: Optional[str] = None,
        require_completed_status: bool = True,
    ) -> None:
        super().__init__()
        self.api_model = api_model
        self.api_base_url = api_base_url
        self.api_key_env = api_key_env
        self.summary_max_items = summary_max_items
        self.top_k = top_k
        self.max_memory_items = max_memory_items
        self.min_similarity = min_similarity
        self.embedding_model_name = embedding_model_name
        self.embedding_device = embedding_device
        self.prompt_config_path = prompt_config_path
        self.require_completed_status = require_completed_status
        if inject_marker:
            self.MEMORY_MARKER = inject_marker
        if inject_end_marker:
            self.MEMORY_END_MARKER = inject_end_marker

        self._client: Optional[OpenAI] = None
        self._embedding_model: Optional[SentenceTransformer] = None
        self._memory_items: list[MemoryItem] = []
        self._memory_index: dict[str, MemoryItem] = {}
        self._prompt_config: Optional[dict[str, Any]] = None

    @classmethod
    def is_unique(cls) -> bool:
        return True

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def restore_state(self) -> None:
        """Load persisted memory items when resuming an experiment."""
        memory_path = self._get_memory_file_path()
        if not os.path.exists(memory_path):
            return
        try:
            with open(memory_path, "r", encoding="utf-8") as f:
                raw_items = json.load(f)
        except Exception as exc:  # pragma: no cover - best effort restoration
            SafeLogger.error(f"[TrajectoryMemory] Failed to load memory file: {exc}")
            return
        self._memory_items = []
        self._memory_index = {}
        for payload in raw_items:
            try:
                item = MemoryItem.from_dict(payload)
            except Exception as exc:  # pragma: no cover
                SafeLogger.warning(f"[TrajectoryMemory] Skip invalid memory item: {exc}")
                continue
            self._append_memory_item(item, persist=False)
        SafeLogger.info(
            f"[TrajectoryMemory] Restored {len(self._memory_items)} memory items from disk."
        )

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        """Persist all memory items to disk."""
        self._persist_memory_items()

    def on_task_reset(self, callback_args: CallbackArguments) -> None:
        """Retrieve relevant memories for the incoming query and inject into the prompt."""
        if not self._memory_items:
            return
        session = callback_args.current_session
        query_text = self._extract_query_text(session)
        if not query_text:
            return
        selected = self._select_relevant_memories(query_text, limit=self.top_k)
        if not selected:
            return
        task = callback_args.session_context.task
        try:
            first_prompt = task.chat_history_item_factory.construct(0, Role.USER).content
        except Exception as exc:  # pragma: no cover
            SafeLogger.warning(
                f"[TrajectoryMemory] Unable to fetch initial user prompt: {exc}"
            )
            return
        base_prompt = first_prompt.split(self.MEMORY_MARKER)[0]
        injected_prompt = base_prompt + self.MEMORY_MARKER
        for idx, memory in enumerate(selected, 1):
            injected_prompt += self._format_memory_item(idx, memory)
        injected_prompt = injected_prompt.rstrip() + self.MEMORY_END_MARKER
        task.chat_history_item_factory.set(
            0,
            Role.USER,
            injected_prompt,
        )
        SafeLogger.info(
            f"[TrajectoryMemory] Injected {len(selected)} memory items for sample "
            f"{session.sample_index}."
        )

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        """After a successful trajectory, summarize and store new memories."""
        session = callback_args.current_session
        if self.require_completed_status and session.sample_status != SampleStatus.COMPLETED:
            return
        query_text = self._extract_query_text(session)
        trajectory_text = self._extract_trajectory_text(session)
        if not query_text or not trajectory_text:
            return
        outcome = session.evaluation_record.outcome
        if outcome == SessionEvaluationOutcome.UNSET:
            outcome = SessionEvaluationOutcome.UNKNOWN
        related_memories = self._select_relevant_memories(query_text, limit=self.top_k)
        related_memories_text = self._render_memories_for_prompt(related_memories)
        summary = self._summarize_trajectory(
            session=session,
            query=query_text,
            trajectory=trajectory_text,
            outcome=outcome,
            related_memories=related_memories_text,
        )
        if not summary:
            return
        new_items = self._parse_memory_items(summary)
        if not new_items:
            SafeLogger.warning(
                "[TrajectoryMemory] No valid memory items parsed from summarization output."
            )
            return
        contents = [item["content"] for item in new_items]
        embeddings = self._embed_texts(contents)
        appended = 0
        for raw_item, embedding in zip(new_items, embeddings):
            if embedding is None:
                continue
            memory = MemoryItem(
                id=str(uuid.uuid4()),
                title=raw_item["title"],
                description=raw_item["description"],
                content=raw_item["content"],
                embedding=embedding.tolist(),
                source_sample_index=str(session.sample_index),
                outcome=outcome.value,
            )
            if self._append_memory_item(memory, persist=True):
                appended += 1
        if appended:
            SafeLogger.info(
                f"[TrajectoryMemory] Added {appended} new memory items "
                f"(total={len(self._memory_items)})."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _append_memory_item(self, memory: MemoryItem, *, persist: bool) -> bool:
        """Insert a memory item if it is not duplicated."""
        content_signature = memory.content.strip().lower()
        if content_signature in self._memory_index:
            return False
        self._memory_items.append(memory)
        self._memory_index[content_signature] = memory
        if len(self._memory_items) > self.max_memory_items:
            removed = self._memory_items.pop(0)
            self._memory_index.pop(removed.content.strip().lower(), None)
        if persist:
            self._persist_memory_items()
        return True

    def _select_relevant_memories(
        self, query_text: str, *, limit: Optional[int] = None
    ) -> list[MemoryItem]:
        if not self._memory_items:
            return []
        query_embedding = self._embed_texts([query_text])[0]
        if query_embedding is None:
            return []
        scored: list[tuple[MemoryItem, float]] = []
        query_vector = np.array(query_embedding, dtype=float)
        for memory in self._memory_items:
            score = float(np.dot(query_vector, np.array(memory.embedding, dtype=float)))
            if score >= self.min_similarity:
                scored.append((memory, score))
        if not scored:
            return []
        scored.sort(key=lambda item: item[1], reverse=True)
        if limit is not None:
            scored = scored[:limit]
        return [memory for memory, _ in scored]

    def _render_memories_for_prompt(self, memories: Sequence[MemoryItem]) -> str:
        if not memories:
            return "None."
        parts: list[str] = []
        for idx, memory in enumerate(memories, start=1):
            parts.append(
                f"- Memory {idx}\n"
                f"  Title: {memory.title}\n"
                f"  Description: {memory.description}\n"
                f"  Outcome: {memory.outcome}\n"
                f"  Content: {memory.content}"
            )
        return "\n".join(parts)

    def _extract_query_text(self, session: Session) -> Optional[str]:
        chat_history = session.chat_history
        length = chat_history.get_value_length()
        for idx in range(length):
            item = chat_history.get_item_deep_copy(idx)
            if item.role == Role.USER:
                return item.content.strip()
        return None

    def _extract_trajectory_text(self, session: Session) -> str:
        parts: list[str] = []
        chat_history = session.chat_history
        for idx in range(chat_history.get_value_length()):
            item = chat_history.get_item_deep_copy(idx)
            parts.append(f"{item.role.value.upper()}: {item.content.strip()}")
        return "\n".join(parts)

    def _format_memory_item(self, index: int, memory: MemoryItem) -> str:
        return (
            f"# Memory Item {index}\n"
            f"## Title {memory.title}\n"
            f"## Description {memory.description}\n"
            f"## Content {memory.content.strip()}\n\n"
        )

    def _summarize_trajectory(
        self,
        *,
        session: Session,
        query: str,
        trajectory: str,
        outcome: SessionEvaluationOutcome,
        related_memories: str,
    ) -> Optional[str]:
        if self._client is None:
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                SafeLogger.warning(
                    f"[TrajectoryMemory] Environment variable {self.api_key_env} is not set."
                )
                return None
            self._client = OpenAI(api_key=api_key, base_url=self.api_base_url)
        prompt_bundle = self._get_prompt_bundle(session.task_name, outcome)
        try:
            user_prompt = prompt_bundle.user.format(
                query=query,
                trajectory=trajectory,
                outcome=outcome.value,
                task_name=session.task_name.value,
                related_memories=related_memories,
            )
        except Exception as exc:
            SafeLogger.error(f"[TrajectoryMemory] Failed to render user prompt: {exc}")
            return None
        try:
            completion = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self.api_model,
                messages=[
                    {"role": "system", "content": prompt_bundle.system},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            SafeLogger.error(f"[TrajectoryMemory] Summarization API failed: {exc}")
            return None
        choices = completion.choices if completion else None
        if not choices:
            return None
        return choices[0].message.content  # type: ignore[index]

    def _parse_memory_items(self, content: str) -> list[dict[str, str]]:
        pattern = re.compile(
            r"#\s*Memory Item\s*\d+\s*"
            r"\n##\s*Title\s*(?P<title>.+?)"
            r"\n##\s*Description\s*(?P<description>.+?)"
            r"\n##\s*Content\s*(?P<content>(?:.+?)(?=\n# Memory Item|\Z))",
            re.DOTALL,
        )
        items: list[dict[str, str]] = []
        for match in pattern.finditer(content):
            title = match.group("title").strip()
            description = match.group("description").strip()
            raw_content = match.group("content")
            raw_content = re.sub(r"[ \t]+", " ", raw_content)
            raw_content = re.sub(r"\n\s*\n", "\n\n", raw_content)
            raw_content = raw_content.strip()
            if title and raw_content:
                items.append(
                    {
                        "title": title,
                        "description": description,
                        "content": raw_content,
                    }
                )
            if len(items) >= self.summary_max_items:
                break
        return items

    def _embed_texts(self, texts: Sequence[str]) -> list[Optional[np.ndarray]]:
        if self._embedding_model is None:
            try:
                self._embedding_model = SentenceTransformer(
                    self.embedding_model_name, device=self.embedding_device
                )
            except Exception as exc:
                SafeLogger.error(
                    f"[TrajectoryMemory] Failed to load embedding model "
                    f"{self.embedding_model_name}: {exc}"
                )
                return [None for _ in texts]
        if not texts:
            return []
        try:
            embeddings = self._embedding_model.encode(
                list(texts), normalize_embeddings=True, convert_to_numpy=True
            )
        except Exception as exc:
            SafeLogger.error(f"[TrajectoryMemory] Embedding encoding failed: {exc}")
            return [None for _ in texts]
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return [embeddings[i] for i in range(embeddings.shape[0])]

    def _get_memory_file_path(self) -> str:
        return os.path.join(self.get_state_dir(), "trajectory_memory_items.json")

    def _persist_memory_items(self) -> None:
        memory_path = self._get_memory_file_path()
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump([item.to_dict() for item in self._memory_items], f, indent=2)

    def _get_prompt_bundle(
        self, task_name: TaskName, outcome: SessionEvaluationOutcome
    ) -> PromptBundle:
        self._ensure_prompt_config_loaded()
        config = self._prompt_config or DEFAULT_PROMPT_CONFIG
        outcome_key = self._normalize_outcome_key(outcome)

        defaults = config.get("defaults", {})
        tasks = config.get("tasks", {})

        task_section = tasks.get(task_name.value, {}) if isinstance(tasks, dict) else {}
        task_bundle = self._coerce_prompt_section(
            task_section.get(outcome_key) if isinstance(task_section, dict) else None,
            {},
        )
        default_bundle = self._coerce_prompt_section(
            defaults.get(outcome_key) if isinstance(defaults, dict) else None,
            {},
        )

        system_text = task_bundle.get("system") or default_bundle.get("system")
        user_text = task_bundle.get("user") or default_bundle.get("user")

        if not system_text or not user_text:
            builtin_defaults = DEFAULT_PROMPT_CONFIG["defaults"]
            fallback = self._coerce_prompt_section(
                builtin_defaults.get(outcome_key)
                or builtin_defaults.get("failure")
                or {},
                {},
            )
            system_text = system_text or fallback.get("system")
            user_text = user_text or fallback.get("user")

        if not system_text or not user_text:
            raise ValueError("Prompt configuration missing required system/user text.")

        return PromptBundle(system=system_text, user=user_text)

    def _normalize_outcome_key(
        self, outcome: SessionEvaluationOutcome
    ) -> str:
        if outcome == SessionEvaluationOutcome.CORRECT:
            return "success"
        if outcome == SessionEvaluationOutcome.INCORRECT:
            return "failure"
        return outcome.value

    def _ensure_prompt_config_loaded(self) -> None:
        if self._prompt_config is not None:
            return
        config = copy.deepcopy(DEFAULT_PROMPT_CONFIG)
        if self.prompt_config_path:
            path = self.prompt_config_path
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            if not os.path.exists(path):
                SafeLogger.error(
                    f"[TrajectoryMemory] Prompt config path not found: {path}"
                )
            else:
                try:
                    with open(path, "r", encoding="utf-8") as fp:
                        loaded = yaml.safe_load(fp) or {}
                    if not isinstance(loaded, dict):
                        SafeLogger.error(
                            "[TrajectoryMemory] Prompt config must be a mapping."
                        )
                    else:
                        config = self._merge_prompt_config(config, loaded)
                except Exception as exc:
                    SafeLogger.error(
                        f"[TrajectoryMemory] Failed to load prompt config: {exc}"
                    )
        self._prompt_config = config

    def _merge_prompt_config(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_prompt_config(
                    merged.get(key, {}), value  # type: ignore[arg-type]
                )
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    def _coerce_prompt_section(
        self,
        section: Optional[dict[str, Any]],
        default: dict[str, Any],
    ) -> dict[str, str]:
        if not isinstance(section, dict):
            return default
        result: dict[str, str] = {}
        system_text = section.get("system")
        user_text = section.get("user")
        if isinstance(system_text, str):
            result["system"] = system_text
        if isinstance(user_text, str):
            result["user"] = user_text
        return result
