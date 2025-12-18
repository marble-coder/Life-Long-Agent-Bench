import torch
import os
from typing import Any, Optional, Mapping, Sequence
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]
import niuload  # type: ignore[import-untyped]

from src.language_models.language_model import LanguageModel
from src.typings import (
    Role,
    ChatHistoryItem,
    LanguageModelContextLimitException,
    LanguageModelOutOfMemoryException,
    ChatHistory,
)


class HuggingfaceLanguageModel(LanguageModel):
    def __init__(
        self,
        model_name_or_path: str,
        role_dict: Mapping[str, str],
        dtype: torch.dtype | str = torch.bfloat16,
        device_map: str | Mapping[str, Any] = "auto",
        max_input_length: Optional[int] = None,  # 添加这行
    ):
        """
        Config explanations
        dtype: `dtype` can be one of: `torch.dtype`, `"auto"` or a string of a valid `torch.dtype`
        device_map: I cannot find the detail documents.
            But it seems that it can be set to "cuda" or {"": "cuda"} to use GPU.
            Set "auto" can use multiple GPUs. (Amazing!)
        """
        super().__init__(role_dict)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_input_length = max_input_length  # 添加这行
        # 🔧 方法1：修改模型配置
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name_or_path)
        
        if max_input_length is not None:
            print(f"⚠️  当前 max_position_embeddings: {config.max_position_embeddings:,}")
            #config.max_position_embeddings = max_input_length
            
        
        if device_map == "niuload":
            # https://zhuanlan.zhihu.com/p/792303768
            device_map = niuload.balanced_load(
                model_name_or_path, return_device_map_only=True
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map=device_map, torch_dtype=dtype
        )
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name_or_path, 
        #     #config=config,  # 使用修改后的配置
        #     device_map=device_map, 
        #     torch_dtype=dtype
        # )

    def _convert_message_list_to_model_input_dict(
        self, batch_message_list: Sequence[Sequence[Mapping[str, str]]]
    ) -> Mapping[str, torch.Tensor]:
        # https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template
        batch_input_ids: torch.Tensor = self.tokenizer.apply_chat_template(
            batch_message_list,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        no_padding_mask: torch.Tensor = batch_input_ids != self.tokenizer.pad_token_id
        # https://discuss.pytorch.org/t/first-nonzero-index/24769/9
        first_no_padding_token_index: torch.Tensor
        _, first_no_padding_token_index = (
            (no_padding_mask.cumsum(-1) == 1) & no_padding_mask
        ).max(-1)
        row_repeat: torch.Tensor = (
            torch.arange(batch_input_ids.shape[1])
            .unsqueeze(0)
            .repeat(batch_input_ids.shape[0], 1)
        ).to(self.model.device)
        column_repeat: torch.Tensor = first_no_padding_token_index.unsqueeze(1).repeat(
            1, batch_input_ids.shape[1]
        )
        batch_attention_mask_ne: torch.Tensor = row_repeat < column_repeat
        batch_attention_mask: torch.Tensor = ~batch_attention_mask_ne
        return {
            "batch_input_ids": batch_input_ids,
            "batch_attention_mask": batch_attention_mask,
        }

    @staticmethod
    def _is_any_gpu_memory_high() -> bool:
        for i in list(range(torch.cuda.device_count())):
            device = torch.device(f"cuda:{i}")
            free, total = torch.cuda.mem_get_info(device)
            # mem_used_gb = (total - free) / 1024**3
            if free / total < 0.1:  # Threshold set to 90%
                return True
        return False

    def _inference(
        self,
        batch_chat_history: Sequence[ChatHistory],
        inference_config_dict: Mapping[str, Any],
        system_prompt: str,
    ) -> Sequence[ChatHistoryItem]:
        """
        Parameters explanations
        system_prompt: The system prompt that is added at the beginning of the conversation.
            The system_prompt constructed by PreTrainedTokenizerFas.apply_chat_template()
            or transformers.pipeline() is the same as the system_prompt in __init__.
            For example, the system_prompt of Llama-3.1-8B-Instruct is:
            ```
            <|start_header_id|>system<|end_header_id|>
            Cutting Knowledge Date: December 2023
            Today Date: 26 Jul 2024

            {system_prompt}<|eot_id|>
            ```
            Reference:
            https://huggingface.co/shenzhi-wang/Llama3.1-8B-Chinese-Chat#21-usage-of-our-bf16-model
            https://huggingface.co/shenzhi-wang/Llama3.1-8B-Chinese-Chat#21-usage-of-our-bf16-model
        inference_config_dict: Additional configs for the transformers.generation.GenerationMixin.generate().
            GenerationMixin is usually a parent class of classes like LlamaForCausalLM.
            It allows to set parameters like max_length, max_new_tokens, etc.
            Example:
            Greedy decoding without maximum length: {do_sample: False, num_beams: 1}  # this is also the default value.
            Documentation:
            https://huggingface.co/docs/transformers/v4.47.1/en/main_classes/text_generation#transformers.GenerationConfig
            https://huggingface.co/docs/transformers/v4.47.1/en/main_classes/text_generation#transformers.GenerationMixin
        """
        # region Set the tokenizer attributes to realize correct padding
        original_tokenizer_padding_side = self.tokenizer.padding_side
        original_tokenizer_pad_token = self.tokenizer.pad_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # endregion
        # region Construct batch_message_list
        message_list_prefix: list[Mapping[str, str]]
        if len(system_prompt) > 0:
            message_list_prefix = [{"role": "system", "content": system_prompt}]
        else:
            message_list_prefix = []
        batch_message_list: Sequence[Sequence[Mapping[str, str]]] = [
            message_list_prefix
            + self._convert_chat_history_to_message_list(chat_history)
            for chat_history in batch_chat_history
        ]
        # endregion
        # 在第149行后添加
        print(f"🎯 Generation parameters: {inference_config_dict}")
        print(f"📝 max_new_tokens = {inference_config_dict.get('max_new_tokens', 'Not set')}")
        # region Generate output
        model_input_dict: Mapping[str, torch.Tensor] = (
            self._convert_message_list_to_model_input_dict(batch_message_list)
        )
        batch_input_ids, batch_attention_mask = (
            model_input_dict["batch_input_ids"],
            model_input_dict["batch_attention_mask"],
        )
        del model_input_dict
        # --------------添加调试信息和长度检查------------------------------------------------------
        current_length = batch_input_ids.shape[-1]
        model_max_length = self.model.config.max_position_embeddings
        effective_max_length = self.max_input_length if self.max_input_length is not None else model_max_length
        usage_percent = (current_length / effective_max_length) * 100

        # 🎯 在推理开始时输出配置信息
        print(f"🔍 Context Length: {current_length:,}/{effective_max_length:,} tokens ({usage_percent:.1f}%)")
        if self.max_input_length is not None:
            print(f"📏 Custom max_input_length: {self.max_input_length:,} (model supports: {model_max_length:,})")
        else:
            print(f"📏 Using model's max_position_embeddings: {model_max_length:,}")
        #-----------------------------------------------------------------------------------------------

        if current_length >= effective_max_length:
            raise LanguageModelContextLimitException(
                f"Input length {current_length} exceeds the configured max_input_length "
                f"{effective_max_length}."
            )
        if batch_input_ids.shape[-1] >= self.model.config.max_position_embeddings:
            raise LanguageModelContextLimitException(
                f"Input length {batch_input_ids.shape[-1]} exceeds the model's max_position_embeddings "
                f"{self.model.config.max_position_embeddings}."
            )
        torch.cuda.synchronize()
        try:
            output_tensor: torch.Tensor = self.model.generate(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,  # Mute warning
                **inference_config_dict,
            )
        except Exception as e:
            if (
                isinstance(e, torch.cuda.OutOfMemoryError)
                or HuggingfaceLanguageModel._is_any_gpu_memory_high()
            ):
                torch.cuda.empty_cache()
                raise LanguageModelOutOfMemoryException(str(e)) from e
            else:
                raise e
        finally:
            torch.cuda.synchronize()
        # endregion
        # region Convert output to ChatHistoryItem
        output_str_list: Sequence[str] = self.tokenizer.batch_decode(
            output_tensor[:, batch_input_ids.shape[1] :], skip_special_tokens=True
        )
        output_list: Sequence[ChatHistoryItem] = [
            ChatHistoryItem(role=Role.AGENT, content=output_str)
            for output_str in output_str_list
        ]
        # endregion
        # region Reset the tokenizer attributes
        self.tokenizer.padding_side = original_tokenizer_padding_side
        self.tokenizer.pad_token = original_tokenizer_pad_token
        # endregion
        return output_list
