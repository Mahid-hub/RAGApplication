import os
import sys
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from .hybrid_search import hybrid_search
from .qdrant_db import index_documents
from eval.faithfulness import evaluate_faithfulness

load_dotenv()

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")

def build_context(results):
    context = ""

    for i, result in enumerate(results, start=1):
        context += f"""SOURCE [{i}]
                        Source: {result["source"]}
                        Chunk ID: {result["chunk_id"]}
                        Text: {result["text"]}
                        -------------------------
                        """

    return context


def generate_answer(query, results):

    context = build_context(results)
    
    prompt = f"""
        You are a helpful RAG question-answering assistant.

        Answer the user's question using ONLY the provided context.

        CITATION RULES:
        1. Every factual statement MUST include a citation.
        2. Use ONLY citations in this exact format: [1], [2], [3], etc.
        3. The citation number must match the SOURCE number in the context.
        4. Do NOT use citations like 【1】, 【2】, (1), or [Source 1].
        5. Do NOT invent citation numbers.
        6. If a statement is supported by multiple sources, use multiple citations, for example [1][2].
        7. If the answer cannot be found in the context, say:
        "I don't have enough information in the provided documents."
        8. Do not use outside knowledge.
        9. Keep the answer concise.
        10. At the end, provide a "Sources:" line containing ALL SOURCE numbers actually used to generate the answer.
        11. Do not include a source number unless information from that source was used in the answer.

        CONTEXT:
        {context}

        QUESTION:
        {query}

        Write the answer with citations:
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": """You are a RAG question-answering assistant. 
                            You MUST answer only using the provided context. 
                            Every factual claim must have a citation in the exact format [1], [2], [3], etc. 
                            Never use citation formats such as 【1】 or (1)."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content
