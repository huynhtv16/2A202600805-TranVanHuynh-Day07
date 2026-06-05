from pathlib import Path
import json
import os
import sys
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunking import (
    FixedSizeChunker,
    SentenceChunker,
    ParagraphChunker,
    SlidingWindowChunker,
    RecursiveChunker,
)
from src.embeddings import LocalEmbedder, _mock_embed
from src.models import Document
from src.store import EmbeddingStore

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sot"

EVAL = [
    {
        "query": "Làm sao phân biệt sốt virus và sốt xuất huyết ở giai đoạn sớm?",
        "gold_docs": [
            "huong-dan-phan-biet-sot-virus-voi-sot-xuat-huyet-vi.md",
            "phan-biet-sot-thuong-sot-virus-va-sot-xuat-huyet-vi.md",
        ],
    },
    {
        "query": "Triệu chứng đặc trưng và cách điều trị sốt tinh hồng nhiệt (scarlet fever)?",
        "gold_docs": ["benh-sot-tinh-hong-nhiet-sot-scarlet-la-benh-gi-vi.md"],
    },
    {
        "query": "Dấu hiệu giúp phân biệt sốt phát ban (sởi/rubella) và sốt xuất huyết trên da?",
        "gold_docs": ["sot-phat-ban-khac-sot-xuat-huyet-nhu-nao-vi.md"],
    },
    {
        "query": "Khi nào cần đưa trẻ nghi sốt xuất huyết đến cơ sở y tế ngay?",
        "gold_docs": [
            "huong-dan-phan-biet-sot-virus-voi-sot-xuat-huyet-vi.md",
            "sot-xuat-huyet-va-sot-xuat-huyet-nang.md",
            "phan-biet-sot-thuong-sot-virus-va-sot-xuat-huyet-vi.md",
        ],
    },
    {
        "query": "Làm sao phân biệt sốt rét và sốt xuất huyết?",
        "gold_docs": [
            "phan-biet-sot-ret-va-sot-xuat-huyet-vi.md",
            "phan-biet-sot-thuong-sot-virus-va-sot-xuat-huyet-vi.md",
        ],
    },
]

FILES = [
    "benh-sot-tinh-hong-nhiet-sot-scarlet-la-benh-gi-vi.md",
    "huong-dan-phan-biet-sot-virus-voi-sot-xuat-huyet-vi.md",
    "phan-biet-sot-ret-va-sot-xuat-huyet-vi.md",
    "phan-biet-sot-thuong-sot-virus-va-sot-xuat-huyet-vi.md",
    "sot-phat-ban-khac-sot-xuat-huyet-nhu-nao-vi.md",
    "sot-xuat-huyet-va-sot-xuat-huyet-nang.md",
]


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2].strip()
    return text.strip()


def build_store(file_list, chunker, embedder):
    store = EmbeddingStore(embedding_fn=embedder)
    docs = []
    for fname in file_list:
        p = DATA_DIR / fname
        raw = read_text(p)
        chunks = chunker.chunk(raw)
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    id=f"{fname}__{i}",
                    content=chunk,
                    metadata={"doc_id": fname},
                )
            )
    store.add_documents(docs)
    return store, docs


def evaluate(store):
    total = 0
    for item in EVAL:
        hits = store.search(item["query"], top_k=3)
        hit_docs = [h.get("metadata", {}).get("doc_id") for h in hits]
        if any(doc in item["gold_docs"] for doc in hit_docs if doc):
            total += 2
    return total


def summarize_chunks(docs):
    lengths = [len(doc.content) for doc in docs]
    return {
        "chunk_count": len(lengths),
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
    }


def main():
    load_dotenv(override=False)
    provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder()
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    strategies = [
        ("FixedSizeChunker", FixedSizeChunker(chunk_size=400, overlap=50)),
        ("SentenceChunker", SentenceChunker(max_sentences_per_chunk=3)),
        ("ParagraphChunker", ParagraphChunker()),
        ("SlidingWindowChunker", SlidingWindowChunker(max_sentences_per_chunk=3)),
        ("RecursiveChunker", RecursiveChunker(chunk_size=400)),
    ]

    results = []
    for name, chunker in strategies:
        store, docs = build_store(FILES, chunker, embedder)
        score = evaluate(store)
        summary = summarize_chunks(docs)
        results.append({"strategy": name, "score": score, **summary})

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
