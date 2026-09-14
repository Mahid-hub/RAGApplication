import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")

def route_query(query):

    query_lower = query.strip().lower()

    casual_queries = [
        "hello",
        "hi",
        "hey",
        "how are you",
        "how are you?",
        "good morning",
        "good afternoon",
        "good evening",
        "thank you",
        "thanks",
        "bye",
        "goodbye"
    ]

    if query_lower in casual_queries:
        return "no_rag"

    prompt = """
You are a router for a document-based RAG system.

Decide whether the user's query requires information
from the document knowledge base.

Return ONLY one word:

rag
or
no_rag

Use "rag" when the query asks for information that
could exist in the documents.

Use "no_rag" when the query is casual conversation
and does not require document information.

Examples:

What is my name? -> rag
What is the company revenue? -> rag
Who is the CEO? -> rag
What does the document say about backup recovery? -> rag

Hello -> no_rag
How are you? -> no_rag
Tell me a joke -> no_rag
Thank you -> no_rag
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
        max_completion_tokens=20
    )

    route = response.choices[0].message.content.strip().lower()

    if route not in ["rag", "no_rag"]:
        route = "rag"

    return route

