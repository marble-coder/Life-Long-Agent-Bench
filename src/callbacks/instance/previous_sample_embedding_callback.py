import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from openai import OpenAI
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
    """Stored session enriched with query text, embedding, outcome and optional reflection."""

    session: Session
    query_text: str = ""
    embedding: Optional[list[float]] = None
    reflection: str = ""  # 对轨迹的反思/总结（可选）
    outcome: str = "success"  # "success" or "failure"
    error_message: str = ""  # 失败时的错误信息
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "session": self.session.model_dump(),
            "query_text": self.query_text,
            "outcome": self.outcome,
            "created_at": self.created_at,
        }
        if self.embedding is not None:
            payload["embedding"] = [float(v) for v in self.embedding]
        if self.reflection:
            payload["reflection"] = self.reflection
        if self.error_message:
            payload["error_message"] = self.error_message
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
        enable_memory_review: bool = False,  # 是否启用 memory review 格式
        store_failed_trajectories: bool = False,  # 是否存储失败轨迹
        # 反思功能相关参数
        enable_reflection: bool = False,  # 是否启用轨迹反思/总结
        reflection_use_local_model: bool = False,  # 是否使用本地模型（而非 API）
        reflection_local_model_path: Optional[str] = None,  # 本地模型路径
        reflection_local_model_device: str = "cuda",  # 本地模型设备
        reflection_local_do_sample: bool = False,  # 本地模型是否采样
        reflection_api_model: str = "qwen-plus",
        reflection_api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        reflection_api_key_env: str = "DASHSCOPE_API_KEY",
        reflection_api_temperature: Optional[float] = 0.0,  # 默认贪婪解码
        reflection_api_seed: Optional[int] = 42,  # 随机种子，固定输出
        reflection_system_prompt: Optional[str] = None,  # 外部传入的 system prompt
        reflection_user_prompt_template: Optional[str] = None,  # 外部传入的 user prompt 模板
        reflection_enable_thinking: bool = False,  # 反思时是否启用思考模式
        reflection_thinking_budget_tokens: Optional[int] = None,  # 思考 token 预算
        reflection_max_new_tokens: int = 1024,  # 本地模型最大生成 token 数
        # Rerank 功能相关参数
        enable_rerank: bool = False,  # 是否启用 rerank
        rerank_candidate_count: int = 10,  # embedding 召回的候选数量
        rerank_use_local_model: bool = False,  # 是否使用本地模型（而非 API）
        rerank_local_model_path: Optional[str] = None,  # 本地模型路径（可与 reflection 共用）
        rerank_local_model_device: str = "cuda",  # 本地模型设备
        rerank_local_do_sample: bool = False,  # 本地模型是否采样
        rerank_api_model: str = "qwen-plus",  # rerank 使用的模型
        rerank_api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        rerank_api_key_env: str = "DASHSCOPE_API_KEY",
        rerank_api_temperature: Optional[float] = 0.0,
        rerank_api_seed: Optional[int] = 42,  # 随机种子，固定输出
        rerank_system_prompt: Optional[str] = None,
        rerank_user_prompt_template: Optional[str] = None,
        rerank_enable_thinking: bool = False,  # rerank 时是否启用思考模式
        rerank_thinking_budget_tokens: Optional[int] = None,  # 思考 token 预算
        rerank_include_guidance: bool = True,  # 是否将 rerank guidance 添加到 prompt
        rerank_max_new_tokens: int = 512,  # 本地模型最大生成 token 数
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
        self.enable_memory_review = enable_memory_review
        self.store_failed_trajectories = store_failed_trajectories

        # 反思功能
        self.enable_reflection = enable_reflection
        self.reflection_use_local_model = reflection_use_local_model
        self.reflection_local_model_path = reflection_local_model_path
        self.reflection_local_model_device = reflection_local_model_device
        self.reflection_local_do_sample = reflection_local_do_sample
        self.reflection_api_model = reflection_api_model
        self.reflection_api_base_url = reflection_api_base_url
        self.reflection_api_key_env = reflection_api_key_env
        self.reflection_api_temperature = reflection_api_temperature
        self.reflection_api_seed = reflection_api_seed
        self.reflection_enable_thinking = reflection_enable_thinking
        self.reflection_thinking_budget_tokens = reflection_thinking_budget_tokens
        self.reflection_max_new_tokens = reflection_max_new_tokens

        # 默认反思 prompt（可通过参数覆盖）
        self.reflection_system_prompt = reflection_system_prompt or self._default_reflection_system_prompt()
        self.reflection_user_prompt_template = reflection_user_prompt_template or self._default_reflection_user_prompt_template()

        # Rerank 功能
        self.enable_rerank = enable_rerank
        self.rerank_candidate_count = rerank_candidate_count
        self.rerank_use_local_model = rerank_use_local_model
        self.rerank_local_model_path = rerank_local_model_path
        self.rerank_local_model_device = rerank_local_model_device
        self.rerank_local_do_sample = rerank_local_do_sample
        self.rerank_api_model = rerank_api_model
        self.rerank_api_base_url = rerank_api_base_url
        self.rerank_api_key_env = rerank_api_key_env
        self.rerank_api_temperature = rerank_api_temperature
        self.rerank_api_seed = rerank_api_seed
        self.rerank_enable_thinking = rerank_enable_thinking
        self.rerank_thinking_budget_tokens = rerank_thinking_budget_tokens
        self.rerank_include_guidance = rerank_include_guidance
        self.rerank_max_new_tokens = rerank_max_new_tokens
        self.rerank_system_prompt = rerank_system_prompt or self._default_rerank_system_prompt()
        self.rerank_user_prompt_template = rerank_user_prompt_template or self._default_rerank_user_prompt_template()

        self._embedding_model: Optional[SentenceTransformer] = None
        self._reflection_client: Optional[OpenAI] = None
        self._rerank_client: Optional[OpenAI] = None
        self._stored_records: list[_SessionRecord] = []
        # 延迟存储的 session: (session, outcome, error_message)
        self._pending_session_to_store: Optional[tuple[Session, str, str]] = None
        # 存储最近一次 rerank 的 guidance
        self._last_rerank_guidance: str = ""
        # 本地模型（reflection 和 rerank 可共用）
        self._local_model: Optional[Any] = None
        self._local_tokenizer: Optional[Any] = None

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
                # 读取反思字段
                reflection_payload = item.get("reflection") if isinstance(item, dict) else None
                if reflection_payload:
                    record.reflection = str(reflection_payload)
                # 读取 outcome 和 error_message 字段
                outcome_payload = item.get("outcome") if isinstance(item, dict) else None
                if outcome_payload:
                    record.outcome = str(outcome_payload)
                error_message_payload = item.get("error_message") if isinstance(item, dict) else None
                if error_message_payload:
                    record.error_message = str(error_message_payload)
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

        # 只记录 greedy 评估的轨迹，但延迟到 on_state_save 时才存储
        # 这样可以避免在 GRPO 采样时召回到当前 query 自己的轨迹
        finish_reason = getattr(current_session, 'finish_reason', None)
        if finish_reason != "GREEDY_EVAL":
            # 不是 greedy 评估，跳过
            return

        is_success = (
            current_session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT
            and current_session.sample_status == SampleStatus.COMPLETED
        )

        if is_success:
            # 成功轨迹
            self._pending_session_to_store = (
                current_session.model_copy(deep=True),
                "success",
                ""
            )
        elif self.store_failed_trajectories:
            # 失败轨迹（仅当 store_failed_trajectories=True 时存储）
            error_message = ""
            if current_session.evaluation_record.outcome == SessionEvaluationOutcome.INCORRECT:
                error_message = "Wrong answer"
            elif current_session.evaluation_record.outcome == SessionEvaluationOutcome.UNKNOWN:
                # 获取详细错误信息（如果有）
                detail_dict = current_session.evaluation_record.detail_dict
                if detail_dict:
                    error_message = f"Evaluation error: {detail_dict}"
                else:
                    error_message = "Evaluation error"
            elif current_session.sample_status != SampleStatus.COMPLETED:
                error_message = f"Sample status: {current_session.sample_status.value}"

            self._pending_session_to_store = (
                current_session.model_copy(deep=True),
                "failure",
                error_message
            )

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
            self._select_topk_records(current_query, agent_role_dict)
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
        # GRPO 训练完成后，将暂存的 session 存入召回库
        if self._pending_session_to_store is not None:
            pending_session, outcome, error_message = self._pending_session_to_store
            query_text = self._extract_query_text(pending_session) or ""
            # DEBUG: Print the full query text being stored
            query_display = query_text.replace("\n", "\\n")
            SafeLogger.info(
                f"[PreviousSampleEmbedding] STORING query for sample {pending_session.sample_index} "
                f"(outcome={outcome}, after GRPO):\n{query_display}"
            )
            embedding = self._encode_query(query_text) if query_text else None

            # 生成反思（如果启用）
            reflection = ""
            if self.enable_reflection and query_text:
                agent_role_dict = callback_args.session_context.agent.get_role_dict()
                trajectory_text = self._extract_trajectory_text(pending_session, agent_role_dict)
                reflection = self._call_reflection_api(
                    query_text, trajectory_text, outcome, error_message
                ) or ""

            record = _SessionRecord(
                session=pending_session,
                query_text=query_text,
                embedding=embedding.tolist() if embedding is not None else None,
                reflection=reflection,
                outcome=outcome,
                error_message=error_message,
            )
            self._stored_records.append(record)
            if len(self._stored_records) > self.max_cached_session_count:
                self._stored_records.pop(0)
            self._pending_session_to_store = None  # 清空暂存

        # 保存状态到文件
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

    def _ensure_local_model(self, model_path: str, device: str) -> bool:
        """确保本地模型已加载（reflection 和 rerank 可共用）"""
        if self._local_model is not None and self._local_tokenizer is not None:
            return True
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            SafeLogger.info(f"[PreviousSampleEmbedding] Loading local model from {model_path}...")
            self._local_tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            self._local_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
            )
            self._local_model.eval()
            SafeLogger.info(f"[PreviousSampleEmbedding] Local model loaded successfully on {device}")
            return True
        except Exception as exc:
            SafeLogger.error(f"[PreviousSampleEmbedding] Failed to load local model: {exc}")
            return False

    def _call_local_model(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> Optional[str]:
        """使用本地模型生成回复"""
        if self._local_model is None or self._local_tokenizer is None:
            SafeLogger.error("[PreviousSampleEmbedding] Local model not loaded")
            return None

        try:
            import torch

            # 构建 chat messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # 使用 tokenizer 的 apply_chat_template
            text = self._local_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._local_tokenizer(text, return_tensors="pt").to(
                self._local_model.device
            )

            # 生成参数
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self._local_tokenizer.eos_token_id,
            }
            # 只有 do_sample=True 时才传 temperature
            if do_sample and temperature > 0:
                gen_kwargs["temperature"] = temperature

            SafeLogger.info(
                f"[PreviousSampleEmbedding] Local model generate with: "
                f"max_new_tokens={max_new_tokens}, do_sample={do_sample}, temperature={temperature}"
            )

            # 生成
            with torch.no_grad():
                outputs = self._local_model.generate(**inputs, **gen_kwargs)

            # 解码，只取新生成的部分
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response = self._local_tokenizer.decode(generated_ids, skip_special_tokens=True)
            return response.strip()

        except Exception as exc:
            SafeLogger.error(f"[PreviousSampleEmbedding] Local model inference failed: {exc}")
            return None

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
        agent_role_dict: Optional[Mapping[Role, str]] = None,
    ) -> list[_SessionRecord]:
        if not self._stored_records:
            return []

        # 确定召回数量：如果启用 rerank，先召回更多候选
        recall_count = self.rerank_candidate_count if self.enable_rerank else self.utilized_sample_count

        query_embedding = self._encode_query(query_text)
        if query_embedding is None:
            candidates = self._stored_records[-recall_count:]
        else:
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
                    f"using most recent {recall_count} records"
                )
                candidates = self._stored_records[-recall_count:]
            else:
                scored.sort(key=lambda item: item[0], reverse=True)
                # DEBUG: Print top scores
                top_scores = [f"{score:.3f}" for score, _ in scored[:recall_count]]
                SafeLogger.info(
                    f"[PreviousSampleEmbedding] Retrieved {len(scored[:recall_count])} candidates "
                    f"with similarity scores: {', '.join(top_scores)}"
                )
                candidates = [record for _, record in scored[:recall_count]]

        # 如果启用 rerank，调用 API 进行重排序
        if self.enable_rerank and len(candidates) > 0 and agent_role_dict is not None:
            SafeLogger.info(
                f"[PreviousSampleEmbedding] Calling rerank API with {len(candidates)} candidates, "
                f"selecting top {self.utilized_sample_count}"
            )
            selected_indices, guidance = self._call_rerank_api(
                query_text, candidates, agent_role_dict
            )
            self._last_rerank_guidance = guidance
            reranked = [candidates[i] for i in selected_indices if i < len(candidates)]
            SafeLogger.info(
                f"[PreviousSampleEmbedding] Rerank selected indices: {selected_indices}, "
                f"guidance: {guidance[:100]}..."
            )
            return reranked[:self.utilized_sample_count]
        else:
            self._last_rerank_guidance = ""
            return candidates[:self.utilized_sample_count]

    def _render_example_text(
        self,
        records: Sequence[_SessionRecord],
        agent_role_dict: Mapping[Role, str],
    ) -> str:
        if not records:
            return "\n"

        parts = []

        # 如果有 rerank guidance，在开头添加
        if self.enable_rerank and self._last_rerank_guidance:
            parts.append(f"\n**How to Use the Following Examples:**\n{self._last_rerank_guidance}\n\n")

        # 如果启用 memory review，添加指导
        if self.enable_memory_review:
            memory_review_instruction = """
Before solving the task, you MUST first review the retrieved memories and output your analysis in the following format:

<memory_review>
Analysis: [For each memory, explain whether it is relevant to the current task and why]
- Memory 1: [relevant/irrelevant] because [reason]
- Memory 2: [relevant/irrelevant] because [reason]
...
Decision: [Which memories will you follow? How will you adapt them to the current task?]
confidence_score: [A float between 0.0 and 1.0 indicating how confident you are that the memories will help solve the current task. 0.0 means memories are not helpful at all, 1.0 means memories provide a direct solution]
</memory_review>

After the memory review, proceed with your solution.

"""
            parts.append(memory_review_instruction)
            parts.append("Below are prior trajectories related to the current query:\n")

            for i, record in enumerate(records):
                question = record.query_text
                if not question:
                    extracted = self._extract_query_text(record.session)
                    question = extracted or ""

                if record.outcome == "success":
                    # 成功轨迹：显示完整轨迹
                    try:
                        conversation = record.session.chat_history.get_value_str(
                            agent_role_dict, start_index=3, end_index=None
                        )
                    except Exception:
                        conversation = ""
                    memory_text = f"[Memory {i+1}] [SUCCESS] Question: {question}\nTrajectory:\n{conversation}\n"
                    if record.reflection:
                        memory_text += f"\nInsight:\n{record.reflection}\n"
                else:
                    # 失败轨迹：只显示 question 和 insight，不显示错误的轨迹
                    memory_text = f"[Memory {i+1}] [FAILURE] Question: {question}\n"
                    if record.error_message:
                        memory_text += f"Error: {record.error_message}\n"
                    if record.reflection:
                        memory_text += f"Lesson learned:\n{record.reflection}\n"
                    else:
                        memory_text += "(No insight available)\n"
                parts.append(memory_text)
        else:
            # 原有逻辑
            parts.append(
                "\nBelow are prior trajectories related to the current query; use them as guidance before planning SQL:\n"
            )
            for i, record in enumerate(records):
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
                memory_text = f"Question {i+1}: {question}\n{conversation}\n"
                if record.reflection:
                    memory_text += f"\nInsight:\n{record.reflection}\n"
                parts.append(memory_text)

        return "".join(parts)

    # ------------------------------------------------------------------
    # Reflection (反思/总结) 相关方法
    # ------------------------------------------------------------------
    def _default_reflection_system_prompt(self) -> str:
        """默认的反思 system prompt"""
        return """You are an expert AI Agent Analyst and Prompt Engineer specialized in optimizing Large Language Model agents for complex reasoning tasks (e.g., Text-to-SQL, Coding, Planning).
Your goal is to analyze the execution trajectory of an agent and distill **generalizable, actionable insights** that can serve as "rules of thumb" to improve future performance on similar (but not identical) tasks."""

    def _default_reflection_user_prompt_template(self) -> str:
        """默认的反思 user prompt 模板，支持 {query}, {trajectory}, {outcome_status}, {error_message} 占位符"""
        return """### Input Context
**1. User Query:**
{query}

**2. Agent Trajectory:**
{trajectory}

**3. Evaluation Outcome:**
{outcome_status}
{error_message}

---

### Analysis Instructions (Chain of Thought)
Please analyze the trajectory deeply and output a structured analysis. Follow these reasoning steps:

**Step 1: Diagnosis (Root Cause Analysis)**
* **If Failure:** pinpoint the *exact* turn where the logic diverged. Was it a syntax error? A hallucination of a column name? A logical gap? Did the agent misunderstand the schema?
* **If Success:** Identify the *critical decision* or "Aha!" moment that made this solution work. Why was this path effective compared to potential pitfalls?

**Step 2: Abstraction (Generalization)**
* **DO NOT** just summarize "The agent wrote a SQL query." (This is useless).
* **DO NOT** mention specific variable names (like `user_id = 5`) unless necessary for the rule pattern.
* **DO** formulate a general heuristic.
    * *Bad Insight:* "The agent forgot to join table A and B."
    * *SOTA Insight:* "When querying metric X, always perform an INNER JOIN between A and B on 'id' to filter out incomplete records, as relying on implicit joins leads to ambiguous column errors."

**Step 3: Refinement (Actionability)**
* Condense the insight into a single, high-impact "Tip" (under 50 words) that can be injected into a future system prompt.
* The insight must be self-contained.

### Output Format
You must output a valid JSON object strictly matching this schema:

```json
{{
  "diagnosis_reasoning": "Your step-by-step analysis of the failure or success factors...",
  "error_type": "Syntax Error | Logic Error | Schema Misunderstanding | Optimal Path | ...",
  "insight": "The refined, generalizable rule or tip.",
  "tags": ["relevant_tool_name", "relevant_concept"]
}}
```"""

    def _call_reflection_api(
        self, query: str, trajectory: str, outcome: str = "success", error_message: str = ""
    ) -> Optional[str]:
        """调用 API 或本地模型生成结构化轨迹反思"""
        # 构建 outcome_status 字符串
        outcome_status = "Success" if outcome == "success" else "Failure"
        error_msg_str = error_message if error_message else ""

        # 构建 user prompt，支持多种占位符格式
        user_prompt = self.reflection_user_prompt_template.format(
            query=query,
            trajectory=trajectory,
            outcome_status=outcome_status,
            error_message=error_msg_str,
        )

        # 使用本地模型
        if self.reflection_use_local_model:
            if not self.reflection_local_model_path:
                SafeLogger.error("[PreviousSampleEmbedding] reflection_local_model_path not set")
                return None
            if not self._ensure_local_model(
                self.reflection_local_model_path, self.reflection_local_model_device
            ):
                return None

            SafeLogger.info(
                f"[PreviousSampleEmbedding] Calling local model for reflection, "
                f"max_new_tokens={self.reflection_max_new_tokens}"
            )
            reflection = self._call_local_model(
                system_prompt=self.reflection_system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=self.reflection_max_new_tokens,
                temperature=self.reflection_api_temperature or 0.0,
                do_sample=self.reflection_local_do_sample,
            )
            if reflection:
                SafeLogger.info(f"[PreviousSampleEmbedding] Generated reflection: {reflection[:200]}...")
            return reflection

        # 使用 API
        if self._reflection_client is None:
            api_key = os.getenv(self.reflection_api_key_env)
            if not api_key:
                SafeLogger.warning(
                    f"[PreviousSampleEmbedding] Missing API key env: {self.reflection_api_key_env}"
                )
                return None
            self._reflection_client = OpenAI(
                api_key=api_key,
                base_url=self.reflection_api_base_url
            )

        try:
            # 构建 API 调用参数
            api_kwargs = {
                "model": self.reflection_api_model,
                "messages": [
                    {"role": "system", "content": self.reflection_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            # 添加 temperature 参数（0.0 为贪婪解码）
            if self.reflection_api_temperature is not None:
                api_kwargs["temperature"] = self.reflection_api_temperature

            # 添加 seed 参数（固定输出）
            if self.reflection_api_seed is not None:
                api_kwargs["seed"] = self.reflection_api_seed

            # 添加思考模式（如果启用）
            if self.reflection_enable_thinking:
                extra_body = {}
                extra_body["enable_thinking"] = True
                if self.reflection_thinking_budget_tokens is not None:
                    extra_body["thinking_budget_tokens"] = self.reflection_thinking_budget_tokens
                api_kwargs["extra_body"] = extra_body

            SafeLogger.info(
                f"[PreviousSampleEmbedding] Reflection API call with: "
                f"model={api_kwargs.get('model')}, "
                f"temperature={api_kwargs.get('temperature')}, "
                f"seed={api_kwargs.get('seed')}"
            )
            completion = self._reflection_client.chat.completions.create(**api_kwargs)
        except Exception as exc:
            SafeLogger.error(f"[PreviousSampleEmbedding] Reflection API call failed: {exc}")
            return None

        choices = completion.choices if completion else None
        if not choices:
            return None

        reflection = choices[0].message.content or ""
        SafeLogger.info(f"[PreviousSampleEmbedding] Generated reflection: {reflection[:200]}...")
        return reflection.strip()

    def _extract_trajectory_text(self, session: Session, agent_role_dict: Mapping[Role, str]) -> str:
        """提取轨迹文本用于反思"""
        try:
            return session.chat_history.get_value_str(
                agent_role_dict, start_index=3, end_index=None
            )
        except Exception:
            parts = []
            for idx in range(session.chat_history.get_value_length()):
                item = session.chat_history.get_item_deep_copy(idx)
                role_name = agent_role_dict.get(item.role, item.role.value)
                parts.append(f"{role_name}: {item.content.strip()}")
            return "\n".join(parts)

    # ------------------------------------------------------------------
    # Rerank 相关方法
    # ------------------------------------------------------------------
    def _default_rerank_system_prompt(self) -> str:
        """默认的 rerank system prompt"""
        return """You are an expert AI assistant specialized in selecting and synthesizing relevant historical examples for Text-to-SQL tasks.

Your task is to:
1. Analyze the current query and candidate historical trajectories
2. Select the TOP-K most relevant trajectories that will help solve the current task
3. Generate a concise "How to Use" guidance explaining how to apply the selected trajectories to the current task

Focus on:
- Semantic similarity between queries
- Schema/table structure similarity
- SQL pattern applicability (JOINs, aggregations, subqueries, etc.)
- Potential pitfalls to avoid based on past failures"""

    def _default_rerank_user_prompt_template(self) -> str:
        """默认的 rerank user prompt 模板"""
        return """### Current Task
**Query:** {current_query}

### Candidate Historical Trajectories
Below are {candidate_count} candidate trajectories retrieved by embedding similarity. Please select the TOP {top_k} most relevant ones.

{candidates}

---

### Instructions
1. Analyze each candidate's relevance to the current task
2. Select exactly {top_k} trajectories that are most helpful
3. Generate a "How to Use" guidance for applying these trajectories

### Output Format
You must output a valid JSON object strictly matching this schema:

```json
{{{{
  "selected_indices": [1, 3, 5, 7],
  "reasoning": "Brief explanation of why these were selected...",
  "guidance": "How to use these historical examples for the current task: ..."
}}}}
```

Notes:
- `selected_indices` should contain exactly {top_k} numbers (1-indexed)
- `guidance` should be a concise, actionable instruction (under 100 words) that tells the model how to leverage the selected examples"""

    def _call_rerank_api(
        self,
        current_query: str,
        candidate_records: list[_SessionRecord],
        agent_role_dict: Mapping[Role, str],
    ) -> tuple[list[int], str]:
        """
        调用 API 或本地模型进行 rerank，返回选中的索引列表和 guidance。

        Returns:
            tuple[list[int], str]: (选中的记录索引列表（0-indexed）, guidance 文本)
        """
        # 构建候选轨迹的文本表示
        candidates_text_parts = []
        for i, record in enumerate(candidate_records):
            question = record.query_text or self._extract_query_text(record.session) or ""
            try:
                trajectory = record.session.chat_history.get_value_str(
                    agent_role_dict, start_index=3, end_index=None
                )
            except Exception:
                trajectory = ""

            outcome_str = "[SUCCESS]" if record.outcome == "success" else "[FAILURE]"
            insight_str = f"\nInsight: {record.reflection}" if record.reflection else ""

            candidate_text = f"""**Candidate {i+1}** {outcome_str}
Question: {question}
Trajectory:
{trajectory}{insight_str}
"""
            candidates_text_parts.append(candidate_text)

        candidates_text = "\n---\n".join(candidates_text_parts)

        # 构建 user prompt
        user_prompt = self.rerank_user_prompt_template.format(
            current_query=current_query,
            candidate_count=len(candidate_records),
            top_k=self.utilized_sample_count,
            candidates=candidates_text,
        )

        # 使用本地模型
        if self.rerank_use_local_model:
            if not self.rerank_local_model_path:
                SafeLogger.error("[PreviousSampleEmbedding] rerank_local_model_path not set")
                return list(range(min(self.utilized_sample_count, len(candidate_records)))), ""
            if not self._ensure_local_model(
                self.rerank_local_model_path, self.rerank_local_model_device
            ):
                return list(range(min(self.utilized_sample_count, len(candidate_records)))), ""

            SafeLogger.info(
                f"[PreviousSampleEmbedding] Calling local model for rerank, "
                f"max_new_tokens={self.rerank_max_new_tokens}"
            )
            response_text = self._call_local_model(
                system_prompt=self.rerank_system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=self.rerank_max_new_tokens,
                temperature=self.rerank_api_temperature or 0.0,
                do_sample=self.rerank_local_do_sample,
            )
            if not response_text:
                return list(range(min(self.utilized_sample_count, len(candidate_records)))), ""
        else:
            # 使用 API
            if self._rerank_client is None:
                api_key = os.getenv(self.rerank_api_key_env)
                if not api_key:
                    SafeLogger.warning(
                        f"[PreviousSampleEmbedding] Missing API key env for rerank: {self.rerank_api_key_env}"
                    )
                    return list(range(min(self.utilized_sample_count, len(candidate_records)))), ""
                self._rerank_client = OpenAI(
                    api_key=api_key,
                    base_url=self.rerank_api_base_url
                )

            try:
                api_kwargs = {
                    "model": self.rerank_api_model,
                    "messages": [
                        {"role": "system", "content": self.rerank_system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                if self.rerank_api_temperature is not None:
                    api_kwargs["temperature"] = self.rerank_api_temperature

                # 添加 seed 参数（固定输出）
                if self.rerank_api_seed is not None:
                    api_kwargs["seed"] = self.rerank_api_seed

                # 添加思考模式（如果启用）
                if self.rerank_enable_thinking:
                    extra_body = {}
                    extra_body["enable_thinking"] = True
                    if self.rerank_thinking_budget_tokens is not None:
                        extra_body["thinking_budget_tokens"] = self.rerank_thinking_budget_tokens
                    api_kwargs["extra_body"] = extra_body

                completion = self._rerank_client.chat.completions.create(**api_kwargs)
            except Exception as exc:
                SafeLogger.error(f"[PreviousSampleEmbedding] Rerank API call failed: {exc}")
                return list(range(min(self.utilized_sample_count, len(candidate_records)))), ""

            choices = completion.choices if completion else None
            if not choices:
                return list(range(min(self.utilized_sample_count, len(candidate_records)))), ""

            response_text = choices[0].message.content or ""

        SafeLogger.info(f"[PreviousSampleEmbedding] Rerank response: {response_text[:300]}...")

        # 解析 JSON 响应
        try:
            # 尝试提取 JSON 块
            json_match = response_text
            if "```json" in response_text:
                json_match = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_match = response_text.split("```")[1].split("```")[0]

            result = json.loads(json_match.strip())
            selected_indices_1indexed = result.get("selected_indices", [])
            guidance = result.get("guidance", "")

            # 转换为 0-indexed，并验证范围
            selected_indices = []
            for idx in selected_indices_1indexed:
                idx_0 = idx - 1  # 1-indexed -> 0-indexed
                if 0 <= idx_0 < len(candidate_records):
                    selected_indices.append(idx_0)

            # 如果解析失败或数量不对，补齐
            if len(selected_indices) < self.utilized_sample_count:
                for i in range(len(candidate_records)):
                    if i not in selected_indices:
                        selected_indices.append(i)
                        if len(selected_indices) >= self.utilized_sample_count:
                            break

            return selected_indices[:self.utilized_sample_count], guidance

        except Exception as exc:
            SafeLogger.warning(f"[PreviousSampleEmbedding] Failed to parse rerank response: {exc}")
            return list(range(min(self.utilized_sample_count, len(candidate_records)))), ""
