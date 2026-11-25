import json
import numpy as np
import faiss
import ollama
from build_index import EMBED_MODEL, INDEX_PATH, CHUNKS_PATH

LLM_MODEL = "llama3.2:3b"
TOP_K = 4  # Anzahl der Abschnitte, die dem Modell gegeben werden

index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, encoding="utf-8") as f:
    chunks = json.load(f)


def search(question, k=TOP_K):
    """Bettet die Frage ein und sucht die k ähnlichsten Abschnitte."""
    q = np.array(ollama.embed(model=EMBED_MODEL, input=[question])["embeddings"], dtype="float32")
    faiss.normalize_L2(q)
    scores, ids = index.search(q, k)
    return [{**chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0])]


def ask_llm(prompt):
    """Schickt einen Prompt an das lokale Sprachmodell."""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},  # möglichst gleiche Antworten bei gleicher Frage
    )
    return response["message"]["content"]


def answer_without_rag(question):
    return ask_llm(question)


def answer_with_rag(question):
    hits = search(question)
    context = "\n\n".join(
        f"[{n}] ({h['source']}, S. {h['page']}): {h['text']}" for n, h in enumerate(hits, 1)
    )
    prompt = f"""Beantworte die Frage ausschließlich mit Hilfe der folgenden Textabschnitte.
Nenne am Ende die Nummern der Abschnitte, die du verwendet hast, z. B. [1].
Wenn die Antwort nicht in den Abschnitten steht, antworte genau mit:
"Das kann ich anhand der Dokumente nicht beantworten."

Textabschnitte:
{context}

Frage: {question}
Antwort:"""
    return ask_llm(prompt), hits


if __name__ == "__main__":
    while True:
        question = input("\nFrage (Enter zum Beenden): ").strip()
        if not question:
            break
        print("\n--- OHNE RAG ---")
        print(answer_without_rag(question))

        answer, hits = answer_with_rag(question)
        print("\n--- MIT RAG ---")
        print(answer)
        print("\nGefundene Quellen:")
        for n, h in enumerate(hits, 1):
            print(f"  [{n}] {h['source']}, S. {h['page']} (Ähnlichkeit {h['score']:.2f})")