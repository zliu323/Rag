from app.domain import SearchResult

SYSTEM_PROMPT = """You are an enterprise knowledge-base assistant.
Answer only from the supplied context. If the context is insufficient, say so clearly.
Cite supporting passages inline as [1], [2], etc. Never invent a source or fact.
Keep the answer concise and preserve important numbers and conditions."""


def build_prompt(question: str, contexts: list[SearchResult]) -> str:
    rendered = "\n\n".join(
        f"[{index}] Source: {item.citation()}\n{item.chunk.text}"
        for index, item in enumerate(contexts, start=1)
    )
    return f"{SYSTEM_PROMPT}\n\nContext:\n{rendered}\n\nQuestion: {question}\nAnswer:"

