import re


def tokenize(text: str) -> list[str]:
    """Tokenize Latin words/numbers and individual CJK characters for offline fallbacks."""
    return re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())

