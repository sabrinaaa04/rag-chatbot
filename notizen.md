# Projektnotizen RAG-Chatbot

## Entscheidungen
- LLM lokal über Ollama (llama3.2:3b): kostenlos, offline lauffähig für KI-Con, Daten bleiben lokal. Nachteil: kleines Modell, schwächer als z. B. GPT-4.
- Embedding-Modell bge-m3 statt nomic-embed-text, weil die Dokumente deutsch sind (bge-m3 ist mehrsprachig).
- FAISS IndexFlatIP mit normalisierten Vektoren = Kosinus-Ähnlichkeit, exakte Suche (bei ~700 Abschnitten schnell genug).
- Chunking: 800 Zeichen, 150 Überlappung, pro Seite (damit Quelle + Seitenzahl bekannt sind).
- temperature = 0, damit die Antworten für die Auswertung wiederholbar sind.

## Beobachtungen
- Ohne RAG: Modell sagt oft ehrlich „weiß ich nicht“, erfindet aber manchmal (z. B. Mensa-Preise).
  → Auswertung mit 3 Kategorien: richtig / falsch (halluziniert) / abgelehnt.
- Kopfzeilen im Modulhandbuch („… DAISY Seite X von 147“) haben die Suche verfälscht → per Regex entfernt.
- Frage „Regelstudienzeit im Studiengang DAISY“: falsche Seiten gefunden, weil die PO nie „DAISY“ schreibt.
  Ohne „DAISY“ in der Frage → richtige Antwort (7 Semester).
  → Schwäche reiner Vektorsuche bei unterschiedlicher Wortwahl → Hybrid-Suche (BM25 + Vektoren, RRF).
- Frage zur Wiederholung: richtige Seite (PO S. 12, § 17 Abs. 3) gefunden, Modell lehnt trotzdem ab.
  → Generierungsfehler, nicht Retrieval-Fehler.
- Hybrid-Suche zuerst mit Reciprocal Rank Fusion (RRF): hat nicht geholfen, weil RRF Treffer bevorzugt,
  die in beiden Listen vorkommen. § 6 wurde nur von BM25 gefunden und ging dadurch verloren.
  → Stattdessen abwechselnd Treffer aus beiden Listen nehmen (Interleaving), TOP_K = 6.
  → Ergebnis: „Regelstudienzeit im Studiengang DAISY“ jetzt richtig beantwortet (7 Semester, PO S. 4 über BM25).
- Quellenangabe des Modells teilweise falsch: Antwort richtig, aber es nennt auch eine Quelle,
  in der die Antwort nicht steht. → Limitation kleiner Modelle; in der Auswertung Antwort und Quelle getrennt bewerten.

## Offene Punkte
- Hybrid-Suche testen ✅ erledigt
- Generierungsfehler: besserer Prompt oder anderes Modell testen