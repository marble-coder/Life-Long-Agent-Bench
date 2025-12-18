from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.callbacks.callback import Callback, CallbackArguments
from src.typings import (
    Role,
    Session,
    SessionEvaluationOutcome,
    TaskName,
    SampleStatus,
    ChatHistory,
    ChatHistoryItem,
)
from src.language_models.language_model import LanguageModel
from src.language_models.instance.huggingface_language_model import (
    HuggingfaceLanguageModel,
)
from src.utils import SafeLogger


@dataclass
class ReflectiveMemoryShard:
    """Reflection-oriented memory shard distilled from a trajectory."""

    id: str
    title: str
    signal: str
    strategy: str
    verification: str
    lesson_type: str
    delta: str
    embedding: list[float]
    source_sample_index: str
    outcome: str
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "signal": self.signal,
            "strategy": self.strategy,
            "verification": self.verification,
            "lesson_type": self.lesson_type,
            "delta": self.delta,
            "embedding": self.embedding,
            "source_sample_index": self.source_sample_index,
            "outcome": self.outcome,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReflectiveMemoryShard":
        return cls(
            id=str(payload["id"]),
            title=str(payload["title"]),
            signal=str(payload["signal"]),
            strategy=str(payload["strategy"]),
            verification=str(payload["verification"]),
            lesson_type=str(payload.get("lesson_type", "pattern")),
            delta=str(payload.get("delta", "novel")),
            embedding=[float(v) for v in payload.get("embedding", [])],
            source_sample_index=str(payload.get("source_sample_index", "")),
            outcome=str(payload.get("outcome", SessionEvaluationOutcome.UNKNOWN.value)),
            created_at=str(payload.get("created_at", "")),
        )


class ReflectiveMemoryCallback(Callback):
    """Generate reflection-style memory shards for retrieval."""

    MEMORY_HEADER = (
        "\n\n=== Reflective Memory Shards (freshly synthesized reasoning heuristics) ===\n"
    )
    MEMORY_FOOTER = "\n=== End of Reflective Memory Shards ==="
    JSON_FENCE = re.compile(r"```json(.*?)```", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        *,
        api_model: str = "qwen-plus",
        api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env: str = "DASHSCOPE_API_KEY",
        summarizer_mode: str = "api",
        local_inference_config: Optional[dict[str, Any]] = None,
        local_model_path: Optional[str] = None,
        local_model_kwargs: Optional[dict[str, Any]] = None,
        max_new_shards: int = 2,
        retrieval_top_k: int = 3,
        max_memory_items: int = 256,
        min_similarity: float = 0.35,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_device: str = "cpu",
        inject_marker: Optional[str] = None,
        inject_end_marker: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.api_model = api_model
        self.api_base_url = api_base_url
        self.api_key_env = api_key_env
        self.summarizer_mode = summarizer_mode
        self.local_inference_config = local_inference_config or {}
        self.local_model_path = local_model_path
        self.local_model_kwargs = local_model_kwargs or {}
        self.max_new_shards = max_new_shards
        self.retrieval_top_k = retrieval_top_k
        self.max_memory_items = max_memory_items
        self.min_similarity = min_similarity
        self.embedding_model_name = embedding_model_name
        self.embedding_device = embedding_device
        if inject_marker:
            self.MEMORY_HEADER = inject_marker
        if inject_end_marker:
            self.MEMORY_FOOTER = inject_end_marker

        self._client: Optional[OpenAI] = None
        self._embedding_model: Optional[SentenceTransformer] = None
        self._local_summarizer: Optional[LanguageModel] = None
        self._memory: list[ReflectiveMemoryShard] = []

    @classmethod
    def is_unique(cls) -> bool:
        return True

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def restore_state(self) -> None:
        path = self._memory_path()
        if not os.path.exists(path):
            return
        try:
            payload = json.load(open(path, "r", encoding="utf-8"))
        except Exception as exc:
            SafeLogger.error(f"[ReflectiveMemory] Failed to load memory: {exc}")
            return
        self._memory = []
        for item in payload:
            try:
                shard = ReflectiveMemoryShard.from_dict(item)
            except Exception:
                continue
            self._append_shard(shard, persist=False)
        SafeLogger.info(
            f"[ReflectiveMemory] Restored {len(self._memory)} shards from disk."
        )

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        self._persist()

    def on_task_reset(self, callback_args: CallbackArguments) -> None:
        if not self._memory:
            return
        session = callback_args.current_session
        query = self._prepare_query_text(session)
        if not query:
            return
        chosen = self._select_shards(query, limit=self.retrieval_top_k)
        if not chosen:
            return
        task = callback_args.session_context.task
        first_prompt = task.chat_history_item_factory.construct(0, Role.USER).content
        base_prompt = first_prompt.split(self.MEMORY_HEADER)[0]
        injected = base_prompt + self.MEMORY_HEADER
        for idx, shard in enumerate(chosen, 1):
            injected += self._format_shard(idx, shard)
        injected = injected.rstrip() + self.MEMORY_FOOTER
        task.chat_history_item_factory.set(0, Role.USER, injected)

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        session = callback_args.current_session
        outcome = session.evaluation_record.outcome
        if outcome == SessionEvaluationOutcome.UNSET:
            outcome = SessionEvaluationOutcome.UNKNOWN
        query = self._prepare_query_text(session)
        if not query:
            return
        trajectory = self._extract_trajectory_text(session)
        related = self._select_shards(query, limit=2)
        related_digest = self._render_related(related)
        content = self._call_reflection_api(
            session,
            query=query,
            trajectory=trajectory,
            related_memories=related_digest,
            outcome=outcome,
            agent=callback_args.session_context.agent,
        )
        if not content:
            return
        shards_payload = self._parse_shard_payload(content)
        if not shards_payload:
            return
        embeddings = self._embed_texts(
            [f"{s['signal']}\n{s['strategy']}" for s in shards_payload]
        )
        for shard_data, embed in zip(shards_payload, embeddings):
            if embed is None:
                continue
            shard = ReflectiveMemoryShard(
                id=str(uuid.uuid4()),
                title=shard_data["title"],
                signal=shard_data["signal"],
                strategy=shard_data["strategy"],
                verification=shard_data["verification"],
                lesson_type=shard_data["lesson_type"],
                delta=shard_data["delta"],
                embedding=embed.tolist(),
                source_sample_index=str(session.sample_index),
                outcome=outcome.value,
            )
            self._append_shard(shard)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _memory_path(self) -> str:
        return os.path.join(self.get_state_dir(), "reflective_memory_shards.json")

    def _persist(self) -> None:
        path = self._memory_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump([item.to_dict() for item in self._memory], fp, indent=2)

    def _append_shard(
        self, shard: ReflectiveMemoryShard, *, persist: bool = True
    ) -> None:
        self._memory.append(shard)
        self._memory.sort(key=lambda item: item.created_at)
        if len(self._memory) > self.max_memory_items:
            self._memory = self._memory[-self.max_memory_items :]
        if persist:
            self._persist()

    def _embed_texts(self, texts: Sequence[str]) -> list[Optional[np.ndarray]]:
        if self._embedding_model is None:
            try:
                self._embedding_model = SentenceTransformer(
                    self.embedding_model_name, device=self.embedding_device
                )
            except Exception as exc:
                SafeLogger.error(
                    f"[ReflectiveMemory] Failed to load embedding model: {exc}"
                )
                return [None for _ in texts]
        try:
            embeddings = self._embedding_model.encode(
                list(texts), normalize_embeddings=True, convert_to_numpy=True
            )
        except Exception as exc:
            SafeLogger.error(
                f"[ReflectiveMemory] Embedding encoding failed: {exc}"
            )
            return [None for _ in texts]
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return [embeddings[i] for i in range(embeddings.shape[0])]

    def _select_shards(
        self, query: str, *, limit: int
    ) -> list[ReflectiveMemoryShard]:
        if not self._memory:
            return []
        query_embedding = self._encode_query_text(query)
        if query_embedding is None:
            return []
        scored: list[tuple[float, ReflectiveMemoryShard]] = []
        for shard in self._memory:
            shard_vec = np.array(shard.embedding, dtype=float)
            score = float(np.dot(query_embedding, shard_vec))
            if score >= self.min_similarity:
                scored.append((score, shard))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [shard for _, shard in scored[:limit]]

    def _format_shard(self, idx: int, shard: ReflectiveMemoryShard) -> str:
        return (
            f"# Shard {idx}: {shard.title}\n"
            f"[Signal] {shard.signal}\n"
            f"[Strategy] {shard.strategy}\n"
            f"[Verification] {shard.verification}\n"
            f"[Lesson] {shard.lesson_type}; Δ: {shard.delta}\n\n"
        )

    def _render_related(self, shards: Sequence[ReflectiveMemoryShard]) -> str:
        blocks: list[str] = []
        for shard in shards:
            blocks.append(
                f"- {shard.title}: Signal={shard.signal}; Strategy={shard.strategy}; Lesson={shard.lesson_type}"
            )
        return "\n".join(blocks) if blocks else "(none)\n"

    def _prepare_query_text(self, session: Session) -> Optional[str]:
        chat_history = session.chat_history
        try:
            item = chat_history.get_item_deep_copy(2)
        except IndexError:
            item = None
        if item and item.role == Role.USER:
            content = item.content.strip()
            if content:
                return self._strip_schema_hint(content)
        for idx in range(chat_history.get_value_length()):
            candidate = chat_history.get_item_deep_copy(idx)
            if candidate.role == Role.USER and candidate.content.strip():
                return self._strip_schema_hint(candidate.content.strip())
        return None

    def _strip_schema_hint(self, query: str) -> str:
        schema_markers = [
            "\nThe name of this table is",
            "\nTable:",
            "\nSchema:",
        ]
        for marker in schema_markers:
            if marker in query:
                query = query.split(marker)[0].strip()
                break
        return query

    def _encode_query_text(self, text: str) -> Optional[np.ndarray]:
        text = text.strip()
        if not text:
            return None
        embedding = self._embed_texts([text])[0]
        if embedding is None:
            return None
        return embedding

    def _extract_trajectory_text(self, session: Session) -> str:
        parts: list[str] = []
        chat_history = session.chat_history
        for idx in range(chat_history.get_value_length()):
            item = chat_history.get_item_deep_copy(idx)
            parts.append(f"{item.role.value.upper()}: {item.content.strip()}")
        return "\n".join(parts)

    def _call_reflection_api(
        self,
        session: Session,
        *,
        query: str,
        trajectory: str,
        related_memories: str,
        outcome: SessionEvaluationOutcome,
        agent,
    ) -> Optional[str]:
        system_prompt, user_prompt = self._build_reflection_prompts(
            session=session,
            query=query,
            trajectory=trajectory,
            related_memories=related_memories,
            outcome=outcome,
        )
        mode = self.summarizer_mode.lower().strip()
        if mode == "local":
            return self._call_local_reflection(
                system_prompt,
                user_prompt,
                agent=agent,
            )
        return self._call_remote_reflection(
            system_prompt,
            user_prompt,
        )

    def _call_remote_reflection(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Optional[str]:
        if self._client is None:
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                SafeLogger.warning(
                    f"[ReflectiveMemory] Missing API key env: {self.api_key_env}"
                )
                return None
            self._client = OpenAI(api_key=api_key, base_url=self.api_base_url)
        try:
            completion = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self.api_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            SafeLogger.error(f"[ReflectiveMemory] API call failed: {exc}")
            return None
        choices = completion.choices if completion else None
        if not choices:
            return None
        return choices[0].message.content or ""

    def _call_local_reflection(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        agent,
    ) -> Optional[str]:
        language_model = self._get_local_language_model(agent)
        if language_model is None:
            SafeLogger.warning(
                "[ReflectiveMemory] Local summarizer requested but no compatible language model found."
            )
            return None
        chat_history = ChatHistory()
        chat_history.inject(ChatHistoryItem(role=Role.USER, content=user_prompt))
        SafeLogger.info(
            "[ReflectiveMemory] Local summarizer prompts:\n"
            f"SYSTEM:\n{system_prompt}\n---\nUSER:\n{user_prompt}\n"
        )
        try:
            response_items = language_model.inference(
                [chat_history],
                self.local_inference_config,
                system_prompt=system_prompt,
            )
        except Exception as exc:  # pragma: no cover
            SafeLogger.error(f"[ReflectiveMemory] Local summarizer failed: {exc}")
            return None
        if not response_items:
            return None
        return response_items[0].content

    def _build_reflection_prompts(
        self,
        *,
        session: Session,
        query: str,
        trajectory: str,
        related_memories: str,
        outcome: SessionEvaluationOutcome,
    ) -> tuple[str, str]:
        system_prompt = (
            "You are a reflective DB reasoning coach. Given the query, trajectory, and prior memory shards, "
            "produce up to {max_items} fresh shards capturing (signal, strategy, verification, lesson_type, delta). "
            "Output strictly as JSON array matching schema: "
            "{{\"title\": str, \"signal\": str, \"strategy\": str, \"verification\": str, "
            "\"lesson_type\": one of ['pattern','pitfall','debug','shortcut'], \"delta\": str}}."
        ).format(max_items=self.max_new_shards)  # noqa: P103
        user_prompt = (
            f"Task={session.task_name.value}\nOutcome={outcome.value}\n"
            f"Query:\n{query}\n\n"
            f"Trajectory:\n{trajectory}\n\n"
            f"Related memory shards:\n{related_memories}\n"
        )
        return system_prompt, user_prompt

    def _get_local_language_model(self, agent) -> Optional[LanguageModel]:
        if self.local_model_path:
            if self._local_summarizer is None:
                try:
                    self._local_summarizer = HuggingfaceLanguageModel(
                        model_name_or_path=self.local_model_path,
                        role_dict={
                            Role.USER.value: "user",
                            Role.AGENT.value: "assistant",
                        },
                        **self.local_model_kwargs,
                    )
                except Exception as exc:
                    SafeLogger.error(
                        f"[ReflectiveMemory] Failed to load local summarizer model: {exc}"
                    )
                    return None
            return self._local_summarizer
        language_model = getattr(agent, "_language_model", None)
        if isinstance(language_model, LanguageModel):
            return language_model
        return None

    def _parse_shard_payload(self, content: str) -> list[dict[str, str]]:
        json_str: Optional[str] = None
        match = self.JSON_FENCE.search(content)
        if match:
            json_str = match.group(1).strip()
        else:
            content = content.strip()
            if content.startswith("[") and content.endswith("]"):
                json_str = content
        if not json_str:
            SafeLogger.warning(
                f"[ReflectiveMemory] Unable to find JSON payload in response: {content[:200]}"
            )
            return []
        try:
            data = json.loads(json_str)
        except Exception as exc:
            SafeLogger.error(f"[ReflectiveMemory] JSON parse error: {exc}")
            return []
        shards: list[dict[str, str]] = []
        if not isinstance(data, list):
            return shards
        for item in data[: self.max_new_shards]:
            if not isinstance(item, dict):
                continue
            try:
                shard = {
                    "title": str(item["title"]).strip(),
                    "signal": str(item["signal"]).strip(),
                    "strategy": str(item["strategy"]).strip(),
                    "verification": str(item.get("verification", "")).strip(),
                    "lesson_type": str(item.get("lesson_type", "pattern")).strip(),
                    "delta": str(item.get("delta", "novel")),
                }
            except Exception:
                continue
            if shard["title"] and shard["signal"] and shard["strategy"]:
                shards.append(shard)
        return shards
