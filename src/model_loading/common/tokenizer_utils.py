from typing import Any


def get_actual_tokenizer(tokenizer: Any) -> Any:
    """Returns the underlying tokenizer, unwrapping VLM processors that wrap one."""
    return tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
