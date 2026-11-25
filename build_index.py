import json
import numpy as np
import faiss
import ollama
from ingest import load_all_chunks

EMBED_MODEL = "bge-m3"
INDEX_PATH = "index.faiss"
CHUNKS_PATH = "chunks.json"


def embed(texts, batch_size=32):
    """Wandelt Texte mit bge-m3 in Vektoren um (in Paketen von 32 Texten)."""
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = ollama.embed(model=EMBED_MODEL, input=batch)
        vectors.extend(response["embeddings"])
        print(f"{min(i + batch_size, len(texts))}/{len(texts)} Abschnitte eingebettet")
    vectors = np.array(vectors, dtype="float32")
    faiss.normalize_L2(vectors)  # auf Länge 1 bringen -> Skalarprodukt = Kosinus-Ähnlichkeit
    return vectors


if __name__ == "__main__":
    chunks = load_all_chunks()
    vectors = embed([c["text"] for c in chunks])

    index = faiss.IndexFlatIP(vectors.shape[1])  # exakte Suche per Skalarprodukt
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"\nIndex mit {index.ntotal} Vektoren ({vectors.shape[1]} Dimensionen) gespeichert.")