import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.embeddings import embed_text
from src.dense_search import dense_search_with_embedding

load_dotenv()

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")

def generate_hypothetical_answer(query):

    prompt = """
            Write a short hypothetical document that could contain
            the answer to the user's question.

            The document is only for information retrieval.
            Do not mention that it is hypothetical.
            Do not say you do not know the answer.
            Do not use bullet points.
            Write 3 to 5 sentences.
            """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0,
        reasoning_effort="low",
        max_completion_tokens=512
    )
    hypothetical_answer = response.choices[0].message.content.strip()

    return hypothetical_answer


def hyde_search(query, limit=20):
    
    hypothetical_answer = generate_hypothetical_answer(query)
    hyde_embedding = embed_text(hypothetical_answer)
    results = dense_search_with_embedding(hyde_embedding, limit)
    return results

