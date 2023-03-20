import json

OPENING_BRACKET: str = "["
CLOSING_BRACKET: str = "]"


def _is_a_list(text: str) -> bool:
    if len(text) < 2:
        return False

    return text[0] == OPENING_BRACKET and text[-1] == CLOSING_BRACKET


def reformat_text(text: str) -> str:
    if not _is_a_list(text):
        return [text]
    return json.loads(text)
