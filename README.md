# RAG QA System

A Python Retrieval-Augmented Generation (RAG) application for indexing local
documents, searching them with multiple retrieval strategies, generating cited
answers, and evaluating retrieval and answer faithfulness.

## Features

- Loads `.txt`, `.md`, and `.pdf` files from `data/raw/`
- Converts PDFs to Markdown with Docling
- Splits documents into size-limited chunks
- Generates dense embeddings with Sentence Transformers
- Stores vectors and document payloads in Qdrant
- Combines BM25 keyword search and dense vector search with Reciprocal Rank Fusion
- Generates multiple search queries for multi-query retrieval
- Reranks retrieved chunks with a cross-encoder
- Supports HyDE retrieval using a hypothetical answer
- Supports adaptive routing between RAG and direct LLM responses
- Generates answers with source citations
- Evaluates answer faithfulness against retrieved context
- Measures retrieval quality with Recall@5 and Recall@10

## Project Structure

```text
.
├── data/   
│   └── raw/                # Source .txt, .md, and .pdf documents
├── eval/
│   ├── evaulate.py         # Retrieval and generation evaluation runner
│   ├── faithfulness.py     # Faithfulness scoring
│   ├── questions.json      # Evaluation questions and relevant chunk IDs
│   └── results.json        # Generated evaluation report
├── src/
│   ├── chunker.py          # Document chunking
│   ├── dense_search.py     # Qdrant vector search
│   ├── embeddings.py       # Sentence Transformer embeddings
│   ├── generator.py        # Context construction and answer generation
│   ├── hyde.py             # Hypothetical-document retrieval
│   ├── hybrid_search.py    # BM25 plus dense search using RRF
│   ├── ingest.py           # Document loading
│   ├── keyword_search.py   # BM25 search
│   ├── multi_query.py      # Query generation and multi-query retrieval
│   ├── qdrant_db.py        # Qdrant connection and indexing
│   ├── query_router.py     # Adaptive RAG routing
│   └── reranker.py         # Cross-encoder reranking
├── main.py                # Interactive CLI entry point
├── requirements.txt
└── README.md
```

## Architecture

```text
Documents in data/raw/
        |
        v
Ingest -> Chunk -> Embed -> Qdrant
                              |
                    +---------+---------+
                    |                   |
                 BM25             Dense Search
                    |                   |
                    +------ RRF -------+
                              |
                         Reranker
                              |
                         LLM Answer
                              |
                    Faithfulness Evaluation
```

The standard multi-query path first asks the LLM to create four query
variations. Each variation is searched with hybrid retrieval, the results are
combined with RRF, and the combined results are reranked before answer
generation.

## Models

| Purpose | Model or service |
| --- | --- |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Embedding size | 384 dimensions |
| Reranking | `BAAI/bge-reranker-base` |
| Chat generation and routing | `openai/gpt-oss-20b` through Groq's OpenAI-compatible API |
| Vector database | Qdrant |

The first use of the embedding or reranking models may download model files
from Hugging Face.

## Installation

Create and activate a virtual environment from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the application with the same environment:

```powershell
python main.py
```

If the environment is not activated, use the interpreter directly:

```powershell
.\.venv\Scripts\python.exe main.py
```

## Configuration

Create a `.env` file in the project root. The variable names are case-sensitive
because they are read exactly as shown by the application.

```env
Endpoint=https://your-qdrant-endpoint
cluster-api-key=YOUR_QDRANT_API_KEY
collection-name=your-collection-name

chunk-size=500
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

Required variables:

- `Endpoint`: Qdrant server or Qdrant Cloud endpoint
- `cluster-api-key`: Qdrant API key
- `collection-name`: collection used for indexed chunks
- `chunk-size`: maximum chunk size used during indexing
- `GROQ_API_KEY`: API key for routing, query generation, HyDE, answer
generation, and faithfulness evaluation

Do not commit `.env`, `.venv/`, model caches, or API keys.

## Add Documents

Put source files directly inside `data/raw/`:

```text
data/raw/
├── company-report.pdf
├── meeting-notes.txt
└── product-overview.md
```

Only files directly inside `data/raw/` are loaded. Nested directories are not
searched. PDF text is extracted with Docling; text and Markdown files are read
as regular text files.

## Run the Application

Start the CLI with `python main.py` and select an option:

1. **Index Documents**: deletes and recreates the configured Qdrant collection,
   then uploads embeddings and metadata for every document in `data/raw/`.
2. **Run Multi-Query**: generates query variations, performs hybrid retrieval,
   reranks results, generates a cited answer, and evaluates faithfulness.
3. **Run HyDE**: generates a hypothetical answer, embeds it, retrieves dense
   results, then generates and evaluates the final answer.
4. **Run Adaptive RAG**: uses the LLM router to choose either multi-query RAG
   or a direct LLM response for casual questions.
5. **Run Retrieval Evaluation**: evaluates every question in
   `eval/questions.json`, reports Recall@5, Recall@10, and faithfulness, and
   writes the report to `eval/results.json`.
6. **Exit**.

Run **Index Documents** before the first search. Indexing is destructive for
the configured collection because it recreates the collection from the current
contents of `data/raw/`.

## Retrieval Details

Keyword search uses BM25. Dense search embeds the user query and searches the
Qdrant collection. Hybrid search combines both rankings with Reciprocal Rank
Fusion:

```text
RRF score = 1 / (k + rank)
```

The implementation uses `k = 60`. Results are then reranked with
`BAAI/bge-reranker-base`; the final answer is built from the highest-ranked
chunks and instructed to cite sources as `[1]`, `[2]`, and so on.

## Evaluation

`eval/questions.json` contains questions and their expected relevant chunk IDs:

```json
{
  "id": "q01",
  "question": "What is the company performance?",
  "relevant_chunks": ["1", "4"]
}
```

The evaluation calculates:

- Recall@5: proportion of relevant chunks found in the first five results
- Recall@10: proportion of relevant chunks found in the first ten results
- Faithfulness: score from 0.0 to 1.0 based on whether the generated answer is
  supported by its retrieved context

Run evaluation from the CLI, or directly with:

```powershell
python -m eval.evaulate
```

The generated report is saved to `eval/results.json`.

## Troubleshooting

### `ModuleNotFoundError`

Make sure packages were installed into the same environment used to run the
application:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

### Groq connection errors

Check that `GROQ_API_KEY` is present in `.env` and that the machine can reach
`https://api.groq.com`. The adaptive router, multi-query generation, HyDE,
answer generation, and faithfulness evaluation all use the Groq API.

### Qdrant errors

Check `Endpoint`, `cluster-api-key`, and `collection-name` in `.env`. Run
**Index Documents** again after changing the source documents or collection
configuration.
