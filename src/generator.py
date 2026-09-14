import os
import sys
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

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
    
    system_prompt = """
                    You are a RAG question-answering assistant.

                    Answer ONLY using the provided context.

                    Rules:
                    - Every factual claim must have a citation.
                    - Citations must use [1], [2], [3], etc.
                    - Citation numbers must match SOURCE numbers.
                    - Never invent citations.
                    - If the context does not contain the answer, say:
                    "I don't have enough information in the provided documents."
                    - Do not use outside knowledge.
                    - Keep the answer concise.
                    - End with a Sources: line listing the source numbers actually used.
                """
                
    user_prompt = f"""
                    CONTEXT:
                    {context}

                    QUESTION:
                    {query}

                    Answer with citations.
                """       

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0,
        reasoning_effort="low",
        # include_reasoning=False,
        max_completion_tokens=512
    )

    return response.choices[0].message.content
