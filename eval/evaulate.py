import json
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.hybrid_search import hybrid_search
from src.reranker import rerank_results
from src.generator import generate_answer, build_context
from eval.faithfulness import evaluate_faithfulness

def load_questions():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "questions.json")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_recall_at_k(results, relevant_chunks, k):
    top_results = results[:k]
    retrieved_ids = set()

    for result in top_results:
        if isinstance(result, dict):
            
            if "chunk_id" in result:
                retrieved_ids.add(str(result["chunk_id"]))

            elif "payload" in result:
                payload = result["payload"]

                if "chunk_id" in payload:
                    retrieved_ids.add(str(payload["chunk_id"]))

    relevant_ids = set(str(chunk_id) for chunk_id in relevant_chunks)
    found = retrieved_ids.intersection(relevant_ids)

    if len(relevant_ids) == 0:
        return 0.0

    recall = len(found) / len(relevant_ids)
    return recall


def get_chunk_ids(results, k):
    chunk_ids = []

    for result in results[:k]:
        if isinstance(result, dict):

            if "chunk_id" in result:
                chunk_ids.append(str(result["chunk_id"]))

            elif "payload" in result:
                payload = result["payload"]

                if "chunk_id" in payload:
                    chunk_ids.append(str(payload["chunk_id"]))

    return chunk_ids


def evaluate_question(question_data):
    question_id = question_data["id"]
    question = question_data["question"]
    relevant_chunks = question_data["relevant_chunks"]
    
    hybrid_results = hybrid_search(question, limit=20)
    hybrid_recall_5 = calculate_recall_at_k(hybrid_results, relevant_chunks, k=5)
    hybrid_recall_10 = calculate_recall_at_k(hybrid_results, relevant_chunks, k=10)
    
    reranked_results = rerank_results(question, hybrid_results, top_k=20)
    reranker_recall_5 = calculate_recall_at_k(reranked_results, relevant_chunks, k=5)
    reranker_recall_10 = calculate_recall_at_k(reranked_results, relevant_chunks, k=10)
    
    hybrid_ids = get_chunk_ids(hybrid_results, k=10)
    reranked_ids = get_chunk_ids(reranked_results, k=10)
    
    context = build_context(reranked_results)
    answer = generate_answer(question, reranked_results)
    faithfulness = evaluate_faithfulness(question, context, answer)

    print("\n" + "=" * 70)
    print(f"Question ID: {question_id}")
    print(f"Question: {question}")
    print(f"Relevant chunks: {relevant_chunks}")
    
    print("\nHybrid Search")
    print(f"Retrieved: {hybrid_ids}")
    print(f"Recall@5: {hybrid_recall_5 * 100:.2f}%")
    print(f"Recall@10: {hybrid_recall_10 * 100:.2f}%")

    print("\nHybrid + Reranker")
    print(f"Retrieved: {reranked_ids}")
    print(f"Recall@5: {reranker_recall_5 * 100:.2f}%")
    print(f"Recall@10: {reranker_recall_10 * 100:.2f}%")

    print("\nGenerated Answer")
    print(answer)

    print("\nFaithfulness")
    print(f"Score: {faithfulness['score'] * 100:.2f}%")
    print("=" * 70)

    return {

        "id": question_id,
        "question": question,
        "relevant_chunks": relevant_chunks,
        "hybrid": {
            "recall_at_5": hybrid_recall_5,
            "recall_at_10": hybrid_recall_10,
            "retrieved_chunks": hybrid_ids
        },
        "hybrid_reranker": {
            "recall_at_5": reranker_recall_5,
            "recall_at_10": reranker_recall_10,
            "retrieved_chunks": reranked_ids
        },
        "answer": answer,
        "faithfulness": faithfulness
    }


def evaluate_main():

    questions = load_questions()
    results = []

    for question_data in questions:
        result = evaluate_question(question_data)
        results.append(result)

    if len(results) > 0:

        average_hybrid_recall_5 = sum(
            result["hybrid"]["recall_at_5"]
            for result in results
        ) / len(results)

        average_hybrid_recall_10 = sum(
            result["hybrid"]["recall_at_10"]
            for result in results
        ) / len(results)

        average_reranker_recall_5 = sum(
            result["hybrid_reranker"]["recall_at_5"]
            for result in results
        ) / len(results)

        average_reranker_recall_10 = sum(
            result["hybrid_reranker"]["recall_at_10"]
            for result in results
        ) / len(results)
        
        average_faithfulness = sum(
            result["faithfulness"]["score"]
            for result in results
        ) / len(results)

    else:
        average_hybrid_recall_5 = 0
        average_hybrid_recall_10 = 0
        average_reranker_recall_5 = 0
        average_reranker_recall_10 = 0
        average_faithfulness = 0
        
    print("\n")
    print("=" * 70)
    print("                    FINAL RESULTS")
    print("=" * 70)
    
    print(f"Questions:   {len(results)}")
    print()
    
    print("Hybrid Search")
    print(f"Recall@5:    {average_hybrid_recall_5 * 100:.2f}%")
    print(f"Recall@10:   {average_hybrid_recall_10 * 100:.2f}%")
    print()
    
    print("Hybrid + Reranker")
    print(f"Recall@5:    {average_reranker_recall_5 * 100:.2f}%")
    print(f"Recall@10:   {average_reranker_recall_10 * 100:.2f}%")
    print()
    
    print("Generation Evaluation")
    print(f"Faithfulness: {average_faithfulness * 100:.2f}%")
    print("=" * 70)
    
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")

    report = {
        "total_questions": len(results),
        "hybrid": {
            "average_recall_at_5": average_hybrid_recall_5,
            "average_recall_at_10": average_hybrid_recall_10
        },
        "hybrid_reranker": {
            "average_recall_at_5": average_reranker_recall_5,
            "average_recall_at_10": average_reranker_recall_10
        },
        "generation": {

            "average_faithfulness":
                average_faithfulness
        },
        "questions": results
    }

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    evaluate_main()