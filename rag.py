import json
import numpy as np
import faiss
import ollama
from build_index import EMBED_MODEL, INDEX_PATH, CHUNKS_PATH
import re
from rank_bm25 import BM25Okapi

LLM_MODEL = "llama3.2:3b"
TOP_K = 6  # Anzahl der Abschnitte, die dem Modell gegeben werden

index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, encoding="utf-8") as f:
    chunks = json.load(f)


def tokenize(text):
    """Zerlegt Text in kleingeschriebene Wörter für die Stichwortsuche."""
    return re.findall(r"\w+", text.lower())


bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])


def search(question, k=TOP_K, candidates=20):
    """Hybrid-Suche: Vektorsuche + BM25, kombiniert per Reciprocal Rank Fusion."""
    # 1) Bedeutungssuche mit Embeddings
    q = np.array(ollama.embed(model=EMBED_MODEL, input=[question])["embeddings"], dtype="float32")
    faiss.normalize_L2(q)
    _, vec_ids = index.search(q, candidates)
    vector_ranking = [int(i) for i in vec_ids[0]]

    # 2) Stichwortsuche mit BM25
    bm25_scores = bm25.get_scores(tokenize(question))
    bm25_ranking = [int(i) for i in np.argsort(bm25_scores)[::-1][:candidates]]

        # 3) Abwechselnd je einen Treffer aus beiden Listen nehmen (ohne Doppelte)
    found = {}  # Abschnitt-Nummer -> womit gefunden
    for vec_i, bm_i in zip(vector_ranking, bm25_ranking):
        for i, method in ((vec_i, "Vektor"), (bm_i, "BM25")):
            if i in found:
                if found[i] != method:
                    found[i] = "beide"
            elif len(found) < k:
                found[i] = method
        if len(found) >= k:
            break
    return [{**chunks[i], "method": m} for i, m in found.items()]

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
            print(f"  [{n}] {h['source']}, S. {h['page']} (gefunden mit: {h['method']})")