from pathlib import Path
import sys
import os
import json

from dotenv import load_dotenv

# ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunking import RecursiveChunker
from src.embeddings import LocalEmbedder, _mock_embed
from src.models import Document
from src.store import EmbeddingStore

DATA_DIR = ROOT / "data" / "sot"

FILES = [
    "benh-sot-tinh-hong-nhiet-sot-scarlet-la-benh-gi-vi.md",
    "huong-dan-phan-biet-sot-virus-voi-sot-xuat-huyet-vi.md",
    "phan-biet-sot-ret-va-sot-xuat-huyet-vi.md",
    "phan-biet-sot-thuong-sot-virus-va-sot-xuat-huyet-vi.md",
    "sot-phat-ban-khac-sot-xuat-huyet-nhu-nao-vi.md",
    "sot-xuat-huyet-va-sot-xuat-huyet-nang.md",
]

QUERIES = [
    "Làm sao phân biệt sốt virus và sốt xuất huyết ở giai đoạn sớm?",
    "Triệu chứng đặc trưng và cách điều trị sốt tinh hồng nhiệt (scarlet fever)?",
    "Dấu hiệu giúp phân biệt sốt phát ban (sởi/rubella) và sốt xuất huyết trên da?",
    "Khi nào cần đưa trẻ nghi sốt xuất huyết đến cơ sở y tế ngay?",
    "Làm sao phân biệt sốt rét và sốt xuất huyết?",
]


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    # strip YAML frontmatter if present
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2].strip()
    return text.strip()


def main():
    load_dotenv(override=False)
    provider = os.getenv("EMBEDDING_PROVIDER", "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder()
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    chunker = RecursiveChunker(chunk_size=400)
    store = EmbeddingStore(embedding_fn=embedder)

    docs = []
    for fname in FILES:
        p = DATA_DIR / fname
        if not p.exists():
            print(f"WARNING: missing {p}")
            continue
        raw = read_text(p)
        chunks = chunker.chunk(raw)
        for i, chunk in enumerate(chunks):
            doc_id = f"{fname}__{i}"
            metadata = {"doc_id": fname}
            docs.append(Document(id=doc_id, content=chunk, metadata=metadata))

    store.add_documents(docs)
    print(f"Indexed {store.get_collection_size()} chunks from {len(FILES)} files")

    results = {}
    for q in QUERIES:
        hits = store.search(q, top_k=3)
        simplified = []
        for h in hits:
            simplified.append({
                "id": h.get("id"),
                "score": float(h.get("score", 0.0)),
                "preview": h.get("content")[:240].replace("\n", " ")
            })
        results[q] = simplified

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
