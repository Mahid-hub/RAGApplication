def recursive_chunk_text(text, c_size, doc):
    chunks = []

    if len(text) <= c_size:
        chunks.append({
            "document_title": doc["metadata"].get("title", ""),
            "source": doc["metadata"].get("source", ""),
            "text": text.strip()
        })
        return chunks

    if "\n\n" in text:
        parts = text.split("\n\n")
        separator = "\n\n"

    elif ". " in text:
        parts = text.split(". ")
        separator = ". "

    else:
        parts = text.split(" ")
        separator = " "

    current_chunk = ""
    for part in parts:
        part = part.strip()

        if not part:
            continue

        if current_chunk:
            candidate = current_chunk + separator + part
        else:
            candidate = part

        if len(candidate) <= c_size:
            current_chunk = candidate

        else:
            if current_chunk:
                chunks.append({
                    "document_title": doc["metadata"].get("title", ""),
                    "source": doc["metadata"].get("source", ""),
                    "text": current_chunk.strip()
                })

            current_chunk = part

    if current_chunk:
        chunks.append({
            "document_title": doc["metadata"].get("title", ""),
            "source": doc["metadata"].get("source", ""),
            "text": current_chunk.strip()
        })

    return chunks