import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from src.callbacks.callback import Callback, CallbackArguments
from src.typings import (
    ChatHistoryItem,
    Role,
    SampleStatus,
    Session,
    SessionEvaluationOutcome,
)
from src.utils import SafeLogger


@dataclass
class _SessionRecord:
    """Stored successful session enriched with query text and embedding."""

    session: Session
    query_text: str = ""
    embedding: Optional[list[float]] = None
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "session": self.session.model_dump(),
            "query_text": self.query_text,
            "created_at": self.created_at,
        }
        if self.embedding is not None:
            payload["embedding"] = [float(v) for v in self.embedding]
        return payload


class PreviousSampleEmbeddingCallback(Callback):
    """
    Embedding-based variant of previous sample utilization.

    When resetting a new session, it retrieves the most similar successful trajectories
    based on cosine similarity between user queries and injects them into the prompt.
    """

    def __init__(
        self,
        original_first_user_prompt: str,
        utilized_sample_count: int,
        *,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_device: str = "cpu",
        max_cached_session_count: Optional[int] = None,
        min_similarity: float = -0.2,
    ):
        super().__init__()
        self.original_first_user_prompt = original_first_user_prompt
        self.pattern = "{previous_sample_utilization_target_position}"
        assert self.original_first_user_prompt.count(self.pattern) == 1
        assert utilized_sample_count > 0
        self.utilized_sample_count = utilized_sample_count
        if max_cached_session_count is None:
            max_cached_session_count = max(utilized_sample_count * 4, utilized_sample_count)
        assert max_cached_session_count >= utilized_sample_count
        self.max_cached_session_count = max_cached_session_count
        self.embedding_model_name = embedding_model_name
        self.embedding_device = embedding_device
        self.min_similarity = min_similarity
        self._embedding_model: Optional[SentenceTransformer] = None
        self._stored_records: list[_SessionRecord] = []

    def _get_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "utilized_session_list.json")

    def restore_state(self) -> None:
        state_path = self._get_state_path()
        if not os.path.exists(state_path):
            return
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                raw_payload = json.load(f)
        except Exception as exc:  # pragma: no cover - best effort restore
            SafeLogger.warning(
                f"[PreviousSampleEmbedding] Failed to load state file: {exc}"
            )
            return
        if not isinstance(raw_payload, list):
            SafeLogger.warning(
                "[PreviousSampleEmbedding] State file is not a list, ignoring."
            )
            return
        records: list[_SessionRecord] = []
        for item in raw_payload:
            try:
                if isinstance(item, dict) and "session" in item:
                    session_payload = item["session"]
                    query_text = str(item.get("query_text", ""))
                    embedding_payload = item.get("embedding")
                    created_at = str(item.get("created_at", "")) or None
                else:
                    # Legacy format: raw session object without metadata
                    session_payload = item
                    query_text = ""
                    embedding_payload = None
                    created_at = None
                session = Session.model_validate(session_payload)
                record = _SessionRecord(session=session)
                record.query_text = query_text
                if created_at:
                    record.created_at = created_at
                if embedding_payload is not None:
                    record.embedding = [float(v) for v in embedding_payload]
                if not record.query_text:
                    extracted = self._extract_query_text(session)
                    record.query_text = extracted or ""
                records.append(record)
            except Exception as exc:  # pragma: no cover
                SafeLogger.warning(
                    f"[PreviousSampleEmbedding] Skip invalid stored session: {exc}"
                )
                continue
        if len(records) > self.max_cached_session_count:
            records = records[-self.max_cached_session_count :]
        self._stored_records = records

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        current_session = callback_args.current_session
        if not (
            current_session.evaluation_record.outcome
            == SessionEvaluationOutcome.CORRECT
            and current_session.sample_status == SampleStatus.COMPLETED
        ):
            return
        query_text = self._extract_query_text(current_session) or ""
        # DEBUG: Print the full query text being stored
        query_display = query_text.replace("\n", "\\n")
        SafeLogger.info(
            f"[PreviousSampleEmbedding] STORING query for sample {current_session.sample_index}:\n{query_display}"
        )
        embedding = self._encode_query(query_text) if query_text else None
        record = _SessionRecord(
            session=current_session.model_copy(deep=True),
            query_text=query_text,
            embedding=embedding.tolist() if embedding is not None else None,
        )
        self._stored_records.append(record)
        if len(self._stored_records) > self.max_cached_session_count:
            self._stored_records.pop(0)

    def on_session_create(self, callback_args: CallbackArguments) -> None:
        assert callback_args.current_session.chat_history.get_value_length() == 0
        callback_args.session_context.task.chat_history_item_factory.set(
            0, Role.USER, self.original_first_user_prompt
        )

    def on_task_reset(self, callback_args: CallbackArguments) -> None:
        session = callback_args.current_session
        current_query = self._extract_query_text(session) or ""
        # DEBUG: Print the full query text being used for retrieval
        query_display = current_query.replace("\n", "\\n")
        SafeLogger.info(
            f"[PreviousSampleEmbedding] RETRIEVING with query for sample {session.sample_index}:\n{query_display}"
        )
        agent_role_dict = callback_args.session_context.agent.get_role_dict()
        selected_records = (
            self._select_topk_records(current_query)
            if current_query
            else []
        )
        example_text = self._render_example_text(selected_records, agent_role_dict)
        first_user_prompt = self.original_first_user_prompt.replace(
            self.pattern, example_text
        )
        task = callback_args.session_context.task
        task.chat_history_item_factory.set(0, Role.USER, first_user_prompt)
        session.chat_history.set(
            0,
            ChatHistoryItem(
                role=Role.USER,
                content=first_user_prompt,
            ),
        )

    def on_agent_inference(self, callback_args: CallbackArguments) -> None:
        last_chat_history_item = (
            callback_args.current_session.chat_history.get_item_deep_copy(-1)
        )
        assert last_chat_history_item.role == Role.AGENT
        last_agent_response = last_chat_history_item.content
        counterfeit_user_response_location = last_agent_response.find("\nuser: ")
        if counterfeit_user_response_location != -1:
            last_agent_response = last_agent_response[
                :counterfeit_user_response_location
            ]
        callback_args.current_session.chat_history.set(
            -1,
            ChatHistoryItem(
                role=Role.AGENT,
                content=last_agent_response,
            ),
        )

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        state_path = self._get_state_path()
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(
                [record.to_dict() for record in self._stored_records],
                f,
                indent=2,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_query_text(self, session: Session) -> Optional[str]:
        try:
            candidate = session.chat_history.get_item_deep_copy(2)
            if candidate.role == Role.USER:
                content = candidate.content.strip()
                if content:
                    # DEBUG: Log that we extracted from index 2
                    SafeLogger.info(
                        f"[PreviousSampleEmbedding] Extracted query from chat_history[2] (length={len(content)} chars)"
                    )
                    # Remove table schema description for better embedding quality
                    content = self._remove_table_schema_info(content)
                    return content
        except IndexError:
            SafeLogger.warning(
                f"[PreviousSampleEmbedding] chat_history[2] not found, using fallback (chat_history length={session.chat_history.get_value_length()})"
            )
            pass
        length = session.chat_history.get_value_length()
        for idx in range(length - 1, -1, -1):
            item = session.chat_history.get_item_deep_copy(idx)
            if item.role == Role.USER:
                content = item.content.strip()
                if content:
                    # DEBUG: Log which index was used in fallback
                    SafeLogger.warning(
                        f"[PreviousSampleEmbedding] FALLBACK: Extracted query from chat_history[{idx}] (length={len(content)} chars)"
                    )
                    # Remove table schema description for better embedding quality
                    content = self._remove_table_schema_info(content)
                    return content
        return None
    
    def _remove_table_schema_info(self, query: str) -> str:
        """
        Remove table schema information from query to improve embedding quality.
        
        In DB_Bench, queries are formatted as:
        "User question\nThe name of this table is xxx, and the headers of this table are ..."
        
        We only want to embed the user question part, not the schema info,
        to avoid false similarity between different questions on the same table.
        """
        # Pattern to detect schema information
        schema_markers = [
            "\nThe name of this table is",
            "\nTable:",  # Alternative format
            "\nSchema:",  # Alternative format
        ]
        
        for marker in schema_markers:
            if marker in query:
                # Split and take only the part before the schema info
                query = query.split(marker)[0].strip()
                SafeLogger.info(
                    f"[PreviousSampleEmbedding] Removed schema info, "
                    f"clean query length: {len(query)} chars"
                )
                break
        
        return query

    def _ensure_embedding_model(self) -> bool:
        if self._embedding_model is not None:
            return True
        try:
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name, device=self.embedding_device
            )
        except Exception as exc:  # pragma: no cover
            SafeLogger.error(
                "[PreviousSampleEmbedding] Failed to load embedding model "
                f"{self.embedding_model_name}: {exc}"
            )
            return False
        return True

    def _encode_query(self, text: str) -> Optional[np.ndarray]:
        if not text.strip():
            return None
        if not self._ensure_embedding_model():
            return None
        # DEBUG: Print the full text we're actually encoding
        text_display = text.replace("\n", "\\n")
        SafeLogger.info(
            f"[PreviousSampleEmbedding] ENCODING text (length={len(text)} chars):\n{text_display}"
        )
        try:
            embedding = self._embedding_model.encode(  # type: ignore[union-attr]
                [text],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except Exception as exc:
            SafeLogger.error(
                "[PreviousSampleEmbedding] Failed to encode embedding: "
                f"{exc}"
            )
            return None
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        return embedding[0].astype(float)

    def _get_record_embedding(self, record: _SessionRecord) -> Optional[np.ndarray]:
        if record.embedding is not None:
            return np.array(record.embedding, dtype=float)
        if not record.query_text.strip():
            return None
        embedding = self._encode_query(record.query_text)
        if embedding is None:
            return None
        record.embedding = embedding.tolist()
        return embedding

    def _select_topk_records(
        self,
        query_text: str,
    ) -> list[_SessionRecord]:
        if not self._stored_records:
            return []
        query_embedding = self._encode_query(query_text)
        if query_embedding is None:
            return self._stored_records[-self.utilized_sample_count :]
        scored: list[tuple[float, _SessionRecord]] = []
        for record in self._stored_records:
            embedding = self._get_record_embedding(record)
            if embedding is None:
                continue
            score = float(np.dot(query_embedding, embedding))
            if score >= self.min_similarity:
                scored.append((score, record))
        if not scored:
            SafeLogger.warning(
                f"[PreviousSampleEmbedding] No records above min_similarity={self.min_similarity}, "
                f"using most recent {self.utilized_sample_count} records"
            )
            return self._stored_records[-self.utilized_sample_count :]
        scored.sort(key=lambda item: item[0], reverse=True)
        # DEBUG: Print top scores
        top_scores = [f"{score:.3f}" for score, _ in scored[: self.utilized_sample_count]]
        SafeLogger.info(
            f"[PreviousSampleEmbedding] Selected {len(scored[: self.utilized_sample_count])} records "
            f"with similarity scores: {', '.join(top_scores)}"
        )
        return [record for _, record in scored[: self.utilized_sample_count]]

    def _render_example_text(
        self,
        records: Sequence[_SessionRecord],
        agent_role_dict: Mapping[Role, str],
    ) -> str:
        if not records:
            return ""
        parts = [
            "\nBelow are prior trajectories related to the current query; use them as guidance before planning SQL:\n"
        ]
        for record in records:
            question = record.query_text
            if not question:
                extracted = self._extract_query_text(record.session)
                question = extracted or ""
            try:
                conversation = record.session.chat_history.get_value_str(
                    agent_role_dict, start_index=3, end_index=None
                )
            except Exception:
                conversation = ""
            parts.append(f"Question {question}:\n{conversation}\n")
        return "".join(parts)
