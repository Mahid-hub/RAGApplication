# RAG QA System

A Retrieval-Augmented Generation (RAG) question-answering system built in Python.

This project implements a complete RAG pipeline that loads documents, splits them into chunks, creates embeddings, stores them in Qdrant, performs keyword and semantic search, combines the results using Reciprocal Rank Fusion (RRF), reranks the retrieved chunks, generates an answer using an LLM, and evaluates both answer faithfulness and retrieval quality.

---

## Features

* Load `.txt`, `.md`, and `.pdf` documents
* Recursive text chunking
* Dense vector embeddings
* Qdrant vector database
* Keyword search using BM25
* Semantic search using vector similarity
* Hybrid search using Reciprocal Rank Fusion (RRF)
* Cross-encoder reranking
* LLM-based answer generation
* Source-based citations
* Faithfulness evaluation
* Retrieval evaluation using Recall@5
* Complete pipeline execution from `main.py`

---

## Project Structure

```text
rag-qa/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── __init__.py
│   ├── ingest.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── keyword_search.py
│   ├── dense_search.py
│   ├── hybrid_search.py
│   ├── reranker.py
│   ├── generator.py
│   └── qdrant_db.py
│
├── eval/
│   ├── questions.json
│   ├── evaulate.py
│   ├── faithfulness.py
│   └── results.json
│
├── main.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## RAG Pipeline

The complete system works in the following order:

```text
Documents
    │
    ▼
Document Ingestion
    │
    ▼
Text Chunking
    │
    ▼
Embedding Generation
    │
    ▼
Qdrant Vector Database
    │
    ├───────────────┐
    ▼               ▼
Dense Search    Keyword Search
    │               │
    └───────┬───────┘
            ▼
      Hybrid Search
        (RRF)
            │
            ▼
        Reranking
            │
            ▼
      Context Building
            │
            ▼
       LLM Generation
            │
            ▼
      Final Answer
            │
            ▼
 Faithfulness Evaluation
```

Retrieval quality is separately evaluated using:

```text
Evaluation Questions
        │
        ▼
   Hybrid Search
        │
        ▼
      Top-K
        │
        ▼
   Ground Truth
        │
        ▼
    Recall@5
```

---

## Technologies Used

* Python
* Sentence Transformers
* Qdrant
* BM25
* Rank-BM25
* Cross-Encoder
* OpenAI-compatible LLM API
* Groq API
* PyPDF
* python-dotenv

---

## Models

### Embedding Model

The project uses:

```text
BAAI/bge-small-en-v1.5
```

This model converts text into dense numerical vectors.

The embedding dimension is:

```text
384
```

These vectors are stored in Qdrant for semantic search.

### Reranking Model

The project uses:

```text
BAAI/bge-reranker-base
```

The cross-encoder compares the user query with each retrieved chunk and produces a relevance score.

### Generation Model

The project uses:

```text
openai/gpt-oss-20b
```

through the Groq OpenAI-compatible API.

---

## Installation

### 1. Clone the project

```bash
git clone <your-repository-url>
cd rag-qa
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Windows CMD:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

Example:

```env
Endpoint=YOUR_QDRANT_ENDPOINT
cluster-api-key=YOUR_QDRANT_API_KEY
collection-name=My-first-collection

chunk-size=500
chunk-overlap=50
INITIAL_TOP_K=20
RERANK_TOP_K=5
EVALUATE_GENERATION=false

GROQ_API_KEY=YOUR_GROQ_API_KEY
```

`Endpoint`, `cluster-api-key`, and `collection-name` are used by Qdrant. The
collection must be created and populated before a dense or hybrid query can
run. `chunk-overlap` is retained for configuration compatibility; the current
chunker uses `chunk-size` and does not apply overlap.

Do not upload `.env` to GitHub.

Add it to `.gitignore`:

```text
.env
.venv/
__pycache__/
*.pyc
```

---

## Adding Documents

Place your documents inside:

```text
data/raw/
```

Supported formats:

```text
.txt
.md
.pdf
```

Example:

```text
data/
└── raw/
    ├── document1.pdf
    ├── document2.pdf
    ├── notes.txt
    └── documentation.md
```

The ingestion module extracts text and basic metadata such as:

* Document title
* Source filename
* Document text

Only files directly inside `data/raw/` are loaded. Files in nested folders are
not included.

---

## Document Chunking

Large documents are divided into smaller chunks before embedding.

The chunker first tries to split using:

1. Paragraphs
2. Sentences
3. Spaces

This makes the chunks more suitable for retrieval.

Example:

```text
Large Document
      │
      ▼
Chunk 1
Chunk 2
Chunk 3
Chunk 4
...
```

Each chunk receives a unique `chunk_id`.

---

## Embeddings

Each text chunk is converted into a vector using:

```text
BAAI/bge-small-en-v1.5
```

Example:

```text
"Qdrant is a vector database"
             │
             ▼
      Embedding Model
             │
             ▼
      384-dimensional vector
```

The vectors are stored in Qdrant.

---

## Qdrant Vector Database

Qdrant is used as the vector database.

Each stored point contains:

```text
chunk_id
document_title
source
text
vector
```

The vector is used for semantic similarity search, while the payload stores the original chunk information.

---

## Keyword Search

Keyword retrieval is implemented using BM25.

The user query is tokenized and compared with the document chunks.

```text
User Query
    │
    ▼
Tokenization
    │
    ▼
BM25
    │
    ▼
Ranked Chunks
```

BM25 is useful when exact words or terms appear in the relevant document.

---

## Dense Search

Dense search converts the user query into an embedding and searches Qdrant for semantically similar vectors.

```text
User Query
    │
    ▼
Embedding Model
    │
    ▼
Query Vector
    │
    ▼
Qdrant
    │
    ▼
Similar Chunks
```

Dense search can retrieve relevant information even when the query and document use different wording.

---

## Hybrid Search

The project combines:

* BM25 keyword search
* Dense semantic search

The two rankings are combined using:

```text
Reciprocal Rank Fusion (RRF)
```

RRF gives each result a score based on its ranking position.

The formula used is:

```text
RRF Score = 1 / (k + rank)
```

where:

```text
k = 60
```

Results appearing in both retrieval methods receive a stronger combined ranking.

---

## Reranking

After hybrid retrieval, the top results are passed to:

```text
BAAI/bge-reranker-base
```

The reranker evaluates:

```text
Query + Retrieved Chunk
```

and assigns a relevance score.

The results are then sorted according to the reranking score.

```text
Hybrid Search
      │
      ▼
Top 20 Results
      │
      ▼
Cross Encoder
      │
      ▼
Top 5 Reranked Results
```

---

## Context Building

The final retrieved chunks are converted into a context for the LLM.

Each source is numbered:

```text
SOURCE [1]
Source: document5.txt
Chunk ID: 1
Text: ...

SOURCE [2]
Source: document4.md
Chunk ID: 17
Text: ...
```

This allows the LLM to cite the retrieved information.

---

## Answer Generation

The LLM receives:

```text
User Query
+
Retrieved Context
```

It is instructed to:

* Use only the retrieved context
* Avoid outside knowledge
* Cite factual statements
* Keep the answer concise
* Return an insufficient-information message when the context does not contain the answer

The generated answer uses source citations such as:

```text
[1]
[2]
[3]
```

These citations correspond to the retrieved context sources.

---

## Faithfulness Evaluation

The generated answer is evaluated against the retrieved context.

The evaluator checks whether every factual claim in the answer is supported by the retrieved context.

The evaluator returns:

```json
{
    "faithful": true,
    "score": 1.0,
    "reason": "All factual claims in the answer are supported by the context."
}
```

A score closer to:

```text
1.0
```

means the answer is strongly supported by the retrieved context.

---

## Retrieval Evaluation

Retrieval quality is evaluated using manually defined questions in:

```text
eval/questions.json
```

Each question contains the expected relevant chunk IDs.

Example:

```json
{
    "id": "q01",
    "question": "Who was on the call?",
    "relevant_chunks": ["1"]
}
```

The evaluation process:

```text
Question
   │
   ▼
Hybrid Search
   │
   ▼
Top 5 Results
   │
   ▼
Compare Chunk IDs
   │
   ▼
Recall@5
```

### Recall@5

Recall@5 measures whether the relevant ground-truth chunks appear within the first five retrieved results.

For example:

```text
Relevant chunks:
[1]

Retrieved Top 5:
[1, 4, 5, 3, 23]
```

The relevant chunk `1` appears in Top 5, therefore:

```text
Recall@5 = 100%
```

---

## Evaluation Output

Evaluation results are written to `eval/results.json`. The report includes the
number of questions, Recall@5 and Recall@10 for hybrid retrieval and reranked
retrieval, average faithfulness, and per-question details. The values change
when the source documents, models, or evaluation questions change.

---

## Running the Project

The interactive application is started from:

```text
main.py
```

Run:

```bash
python main.py
```

Choose one of the following menu options:

1. **Index Documents**: creates the configured Qdrant collection if needed and
    uploads embeddings and payloads for documents in `data/raw/`.
2. **Run Query**: runs BM25 and dense retrieval, combines them with RRF,
    reranks the results, generates a cited answer, and evaluates faithfulness.
3. **Run Retrieval Evaluation**: runs the questions in `eval/questions.json`,
    calculates Recall@5 and Recall@10 before and after reranking, and writes
    `eval/results.json`.
4. **Exit**.

Run **Index Documents** before the first query or retrieval evaluation. If the
configured Qdrant collection is missing, dense search reports that it must be
indexed first.

Retrieval evaluation does not call the LLM by default. Set
`EVALUATE_GENERATION=true` in `.env` to generate answers and run faithfulness
evaluation for every question. This can consume substantial API tokens and
may be affected by provider rate limits.

The retrieval evaluation can also be run directly with:

```bash
python -m eval.evaulate
```

---

## Example Output

```text
======================================================================
                    RAG QA SYSTEM
======================================================================

======================================================================
                    DOCUMENT INDEXING
======================================================================

Documents indexed: 5
Total points: 25

======================================================================
                    RAG PIPELINE
======================================================================

Query: engineering sync notes

----------------------------------------------------------------------
STEP 1: HYBRID SEARCH
----------------------------------------------------------------------

Retrieved results: 20

----------------------------------------------------------------------
STEP 2: RERANKING
----------------------------------------------------------------------

Reranked results: 5

----------------------------------------------------------------------
STEP 3: BUILD CONTEXT
----------------------------------------------------------------------

Context successfully created.

----------------------------------------------------------------------
STEP 4: GENERATE ANSWER
----------------------------------------------------------------------

RAG ANSWER

Engineering sync notes include the following key points...

Sources: [1][4]

----------------------------------------------------------------------
STEP 5: FAITHFULNESS EVALUATION
----------------------------------------------------------------------

Faithful: True
Score:    1.0
Reason:   All factual claims in the answer are supported by the context.

======================================================================
                 RETRIEVAL EVALUATION
======================================================================

Questions:   <number of questions>
Recall@5:    <measured result>

======================================================================
                 PIPELINE COMPLETED
======================================================================
```

---

## Evaluation Files

### `questions.json`

Contains evaluation questions and their ground-truth relevant chunks.

### `evaulate.py`

Runs retrieval evaluation and calculates Recall@5 and Recall@10. Generation
and faithfulness evaluation are optional and controlled by
`EVALUATE_GENERATION`.

### `results.json`

Stores the evaluation results, including:

* Total questions
* Average Recall@5
* Retrieved chunks
* Relevant chunks
* Per-question recall

---

## Main Components

### `ingest.py`

Loads `.txt`, `.md`, and `.pdf` files.

### `chunker.py`

Splits documents into smaller chunks.

### `embeddings.py`

Generates vector embeddings.

### `qdrant_db.py`

Creates the Qdrant collection and stores document embeddings.

### `keyword_search.py`

Performs BM25 keyword retrieval.

### `dense_search.py`

Performs semantic vector retrieval using Qdrant.

### `hybrid_search.py`

Combines keyword and dense retrieval using RRF.

### `reranker.py`

Reranks retrieved chunks using a cross-encoder.

### `generator.py`

Builds context and generates the final answer using the LLM.

### `faithfulness.py`

Evaluates whether the generated answer is supported by the retrieved context.

### `evaulate.py`

Measures retrieval performance using Recall@5 and Recall@10, and records
faithfulness results for generated answers.

### `main.py`

Runs the complete RAG pipeline from one command.

---

## Advantages of This Approach

### Hybrid Retrieval

Combining BM25 and dense retrieval improves retrieval because:

* BM25 handles exact keyword matches.
* Dense retrieval handles semantic similarity.
* RRF combines both rankings.

### Reranking

Reranking improves the quality of the final context by selecting the most relevant chunks from the initial retrieval results.

### Faithfulness Evaluation

The system checks whether generated answers are actually supported by the retrieved documents.

### Retrieval Evaluation

Recall@5 provides a measurable way to evaluate the retrieval system instead of relying only on subjective answer quality.

---

## Limitations

The current system has some limitations:

* Chunking is based mainly on character length and simple separators.
* BM25 uses basic whitespace tokenization.
* Retrieval performance depends on chunk size.
* The system currently uses a fixed top-K configuration.
* Ground-truth evaluation questions are manually created.
* Faithfulness evaluation depends on another LLM judgment.
* The current citation system maps LLM source references to retrieved chunk IDs.

---

## Future Improvements

Possible improvements include:

* Better semantic chunking
* Contextual retrieval
* Query rewriting
* Multi-query retrieval
* HyDE
* Query decomposition
* Adaptive RAG
* Agentic RAG
* Metadata filtering
* Better citation validation
* More comprehensive evaluation datasets
* Precision@K and MRR evaluation
* Answer relevance evaluation
* Retrieval latency measurement
* RAGAS-based evaluation
* Streaming LLM responses
* Persistent local Qdrant storage
* Improved document metadata handling

---

## Security

Never commit API keys or secrets.

Make sure `.env` is included in `.gitignore`:

```text
.env
```

If an API key is accidentally pushed to GitHub, revoke and regenerate the key immediately.

---

## Conclusion

This project demonstrates a complete production-style RAG workflow:

```text
Ingestion
   ↓
Chunking
   ↓
Embedding
   ↓
Vector Database
   ↓
Keyword + Dense Retrieval
   ↓
Hybrid Search
   ↓
Reranking
   ↓
Context Construction
   ↓
LLM Generation
   ↓
Faithfulness Evaluation
   ↓
Retrieval Evaluation
```

Evaluation metrics are written to `eval/results.json` after each evaluation
run. The measured values depend on the indexed documents, model versions, and
the questions in `eval/questions.json`.

The project provides a practical foundation for building more advanced RAG
systems using query transformation, contextual retrieval, adaptive retrieval,
and agentic approaches.
