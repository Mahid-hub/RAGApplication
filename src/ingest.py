from pathlib import Path
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

def load_documents(path):
    data = Path(path)
    files = list(data.glob('*.txt')) + list(data.glob('*.md')) + list(data.glob('*.pdf'))
    documents = []

    for file in files:
        if file.suffix == '.pdf':
            result = converter.convert(file)
            text = result.document.export_to_markdown()
            title = file.stem
            
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