import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.hybrid_search import hybrid_search
from src.reranker import rerank_results

load_dotenv()

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")


def generate_queries(query):
    prompt = """
            Generate 4 different search queries for the user's question.

            Each query should express the same information need
            using different wording.

            Return only 4 queries, one query per line.
            Do not add numbers, bullets, explanations, or any extra text.
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

    result = response.choices[0].message.content

    queries = [
        line.strip()
        for line in result.splitlines()
        if line.strip()
    ]

    return queries


def multi_query_search(query):
    queries = generate_queries(query)
    all_results = []
    
    for search_query in queries:
        results = hybrid_search(search_query, limit=20)
        all_results.append(results)
        
    final_results = calculate_multi_query_rrf(all_results) 
    reranked_results = rerank_results(query, final_results, top_k=10)
    return reranked_results


def calculate_multi_query_rrf(all_query_results, k=60):
    scores = {}
    result_map = {}

    for results in all_query_results:
        for rank, result in enumerate(results, start=1):
            chunk_id = str(result["chunk_id"])

            if chunk_id not in scores:
                scores[chunk_id] = 0.0
                result_map[chunk_id] = result

            scores[chunk_id] += 1 / (k + rank)

    final_results = []

    for chunk_id, score in scores.items():
        result = result_map[chunk_id].copy()
        result["multi_query_rrf_score"] = score
        final_results.append(result)

    final_results.sort(
        key=lambda x: x["multi_query_rrf_score"],
        reverse=True
    )

    return final_results

# if __name__ == "__main__":
#     query = "What are the problems with backup recovery?"

#     # Generate 4 queries
#     queries = generate_queries(query)

#     print("\nGenerated Queries:")
#     for q in queries:
#         print(q)

#     # Multi-query search with duplicate removal
#     all_results = multi_query_search(query)

#     print(f"\nFinal Multi-Query Results: {len(all_results)}")

#     for result in all_results:
#         print(
#             f"Chunk ID: {result['chunk_id']} | "
#             f"Source: {result['source']} | "
#             f"Rerank Score: {result['rerank_score']:.4f}"
#         )