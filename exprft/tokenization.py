from __future__ import annotations


def _to_ids(encoded) -> list[int]:
    if hasattr(encoded, "keys"):          # BatchEncoding / dict
        encoded = encoded["input_ids"]
    return [int(t) for t in encoded]


def render_prefix(tokenizer, prefix_messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(prefix_messages, tokenize=False, add_generation_prompt=True)


def encode_prefix(tokenizer, prefix_messages: list[dict]) -> list[int]:
    """Token ids of the rendered prefix (the chat template supplies BOS; no extra specials)."""
    ids = _to_ids(tokenizer.encode(render_prefix(tokenizer, prefix_messages), add_special_tokens=False))
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and len(ids) > 1 and ids[0] == bos and ids[1] == bos:
        raise ValueError("duplicate BOS token in prefix encoding")
    return ids


def split_prefix_and_response(tokenizer, prefix_messages: list[dict], response: str) -> tuple[str, str]:
    """Render with the chat template and split into (prompt text, response text + EOT)."""
    full = tokenizer.apply_chat_template(
        prefix_messages + [{"role": "assistant", "content": response}],
        tokenize=False, add_generation_prompt=False,
    )
    prefix = render_prefix(tokenizer, prefix_messages)
    if not full.startswith(prefix):
        raise ValueError("chat template: generation prompt is not a prefix of the full rendering")
    return prefix, full[len(prefix):]


def encode_example(
    tokenizer, prefix_messages: list[dict], response: str, max_seq_length: int
) -> dict | None:
    """Return {input_ids, labels, n_response_tokens} or None if the response does not fit."""
    _, response_text = split_prefix_and_response(tokenizer, prefix_messages, response)
    prefix_ids = encode_prefix(tokenizer, prefix_messages)
    response_ids = _to_ids(tokenizer.encode(response_text, add_special_tokens=False))
    if len(prefix_ids) + len(response_ids) > max_seq_length or not response_ids:
        return None
    return {
        "input_ids": prefix_ids + response_ids,
        "labels": [-100] * len(prefix_ids) + response_ids,
        "n_response_tokens": len(response_ids),
    }


def expand_trajectory(trajectory_messages: list[dict]) -> list[tuple[list[dict], str]]:
    """Split a full chat into one (prefix, response) pair per assistant turn."""
    pairs = []
    for i, msg in enumerate(trajectory_messages):
        if msg["role"] == "assistant":
            pairs.append((trajectory_messages[:i], msg["content"]))
    return pairs
