from transformers import PreTrainedTokenizerBase

from rllm.parser import ChatTemplateParser


def get_recent_assistant_user_messages(chat_completions_messages):
    """
    Extracts the most recent assistant message and environment messages (user/tool) from a chat completions list.

    Args:
        chat_completions_messages (List[Dict]): List of message dictionaries from chat completions.

    Returns:
        Tuple[Dict, List[Dict]]: A tuple containing:
            - The most recent assistant message (or None if not found)
            - A list of environment messages (user/tool) that occurred after the last assistant message,
              in chronological order.
    """
    # Loop backwards to get the last assistant message and environment messages
    env_messages = []
    assistant_message = None
    seen_assistant_message = False
    for message in reversed(chat_completions_messages):
        role = message.get("role", None)
        if role == "assistant":
            if assistant_message:
                break
            seen_assistant_message = True
            assistant_message = message
        elif role in ["user", "tool"] and not seen_assistant_message:
            env_messages.append(message)
    # Reverse the env_messages to maintain chronological order
    env_messages = list(reversed(env_messages))

    return assistant_message, env_messages


def convert_messages_to_tokens_and_masks(messages: list[dict[str, str]], tokenizer: PreTrainedTokenizerBase, parser: ChatTemplateParser, contains_first_msg=False, contains_generation_msg=False):
    """
    Converts multiple messages to tokens and masks.
    contains_first_msg flag and contains_generaiton_msg flag are used to indicate whether the conversation is for beginning or contains the generation.
    The first and last message is assumed to be the special message respectively

    Args:
        messages (List[Dict]): The messages to convert.
        tokenizer: The tokenizer to use.
        contains_first_msg (bool): Whether the first message is a special message.
        contains_generation_msg (bool): Whether the last message is a special message.

    Returns:
        Tuple[List[int], List[int]]: A tuple containing all tokens and all masks.
    """
    all_msg_tokens = []
    all_msg_masks = []

    def _convert_message_to_tokens_and_masks(msg, first_msg=False, generation_msg=False, prev_added_gen_prompt=False):
        """
        Args:
            msg: The message to convert.
            first_msg: Whether this is the first message (adds BOS token).
            generation_msg: Whether to add generation prompt after this message.
            prev_added_gen_prompt: Whether the previous message added a generation prompt.
                                   If True, assistant messages should remove their header token.
        """
        msg_text = parser.parse([msg], add_generation_prompt=generation_msg, is_first_msg=first_msg)

        # Remove the assistant token only if the previous message added a generation prompt
        # This ensures proper chat template structure for multi-turn conversations
        if msg["role"] == "assistant" and prev_added_gen_prompt:
            assert msg_text.startswith(parser.assistant_token), f"Expected assistant token {parser.assistant_token} but got {msg_text}"
            msg_text = msg_text.replace(parser.assistant_token, "", 1)  # Only replace first occurrence

        msg_tokens = tokenizer.encode(msg_text, add_special_tokens=False)
        mask_value = 1 if msg["role"] == "assistant" else 0
        msg_mask = [mask_value] * len(msg_tokens)

        return msg_tokens, msg_mask

    for i, msg in enumerate(messages):
        is_first_msg = contains_first_msg and i == 0
        is_last_msg = i == len(messages) - 1

        # Determine if we should add generation prompt:
        # 1. If this is the last message and contains_generation_msg is True
        # 2. If this is NOT an assistant message AND the next message IS an assistant message
        #    (to prepare the assistant token for the next message)
        if is_last_msg:
            add_gen_prompt = contains_generation_msg
        elif msg["role"] != "assistant" and i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
            add_gen_prompt = True
        else:
            add_gen_prompt = False

        # Check if previous message added a generation prompt
        prev_added_gen_prompt = False
        if i > 0:
            prev_msg = messages[i - 1]
            prev_is_last = (i - 1) == len(messages) - 1
            if prev_is_last:
                prev_added_gen_prompt = contains_generation_msg
            elif prev_msg["role"] != "assistant" and msg["role"] == "assistant":
                prev_added_gen_prompt = True

        msg_tokens, msg_mask = _convert_message_to_tokens_and_masks(
            msg,
            first_msg=is_first_msg,
            generation_msg=add_gen_prompt,
            prev_added_gen_prompt=prev_added_gen_prompt
        )
        all_msg_tokens.extend(msg_tokens)
        all_msg_masks.extend(msg_mask)

    return all_msg_tokens, all_msg_masks
