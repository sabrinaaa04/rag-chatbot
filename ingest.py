from pathlib import Path
from pypdf import PdfReader

DATA_DIR = Path("data")
CHUNK_SIZE = 800   # Zeichen pro Abschnitt
OVERLAP = 150      # Überlappung zwischen zwei Abschnitten


def load_pages(pdf_path):
    """Liest ein PDF Seite für Seite und merkt sich Dateiname und Seitenzahl."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"source": pdf_path.name, "page": i + 1, "text": text})
    return pages


def split_into_chunks(pages, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Zerlegt jede Seite in überlappende Abschnitte fester Länge."""
    chunks = []
    for p in pages:
        text = " ".join(p["text"].split())  # überflüssige Leerzeichen/Umbrüche entfernen
        start = 0
        while start < len(text):
            chunks.append({
                "source": p["source"],
                "page": p["page"],
                "text": text[start:start + chunk_size],
            })
            start += chunk_size - overlap
    return chunks

def load_all_chunks(data_dir=DATA_DIR, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Liest alle PDFs im data-Ordner ein und gibt alle Abschnitte zurück."""
    pages = []
    for pdf in sorted(Path(data_dir).glob("*.pdf")):
        pages += load_pages(pdf)
    return split_into_chunks(pages, chunk_size, overlap)

if __name__ == "__main__":
    pages = []
    for pdf in DATA_DIR.glob("*.pdf"):
        pages += load_pages(pdf)

    chunks = split_into_chunks(pages)
    print(f"{len(pages)} Seiten eingelesen, {len(chunks)} Abschnitte erzeugt\n")
    print("Beispiel-Abschnitt:")
    print(chunks[0])