from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        retrieved = self.store.search(question, top_k=top_k)
        context_lines = []
        for item in retrieved:
            content = item.get("content", "")
            metadata = item.get("metadata", {})
            if metadata:
                context_lines.append(f"[{metadata}] {content}")
            else:
                context_lines.append(content)

        prompt = (
            "Use the retrieved context to answer the question.\n\n"
            "Context:\n"
            + "\n\n".join(context_lines)
            + "\n\n"
            + f"Question: {question}\n"
            + "Answer:"
        )
        return self.llm_fn(prompt)
