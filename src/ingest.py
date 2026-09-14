from pathlib import Path
from pypdf import PdfReader

def load_documents(path):
    data = Path(path)
    files = list(data.glob('*.txt')) + list(data.glob('*.md')) + list(data.glob('*.pdf'))
    documents = []

    for file in files:
        if file.suffix == '.pdf':
            reader = PdfReader(file)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            meta = reader.metadata or {}
            title = meta.title if meta and meta.title else file.stem
        else:
            text = file.read_text()
            title = file.stem

        f = {
            'text': text,
            'metadata': {
                'source': file.name,
                'title': title
            }
        }
        documents.append(f)

    return documents