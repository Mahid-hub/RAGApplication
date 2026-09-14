import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from dotenv import load_dotenv
from src.qdrant_db import index_documents
from src.multi_query import multi_query_search
from src.generator import generate_answer, build_context
from eval.faithfulness import evaluate_faithfulness
from eval.evaulate import evaluate_main
from src.hyde import hyde_search

load_dotenv()

def run_hyde(query):

    print("\n" + "-" * 70)
    print("STEP 1: HYDE RETRIEVAL")
    print("-" * 70)
    results = hyde_search(query, limit=5)
    print(f"Retrieved results: {len(results)}")

    for i, result in enumerate(results, start=1):
        print(
            f"{i}. Chunk ID: {result['chunk_id']} | "
            f"Source: {result['source']} | "
            f"Score: {result['score']:.4f}"
        )
        

    print("\n" + "-" * 70)
    print("STEP 2: BUILD CONTEXT")
    print("-" * 70)
    context = build_context(reranked_results)
    print("Context successfully created.")


    print("\n" + "-" * 70)
    print("STEP 3: GENERATE ANSWER")
    print("-" * 70)
    answer = generate_answer(query, reranked_results)
    print(answer)


    print("\n" + "-" * 70)
    print("STEP 4: FAITHFULNESS EVALUATION")
    print("-" * 70)

    faithfulness = evaluate_faithfulness(query, context, answer)
    
    print(f"\nFaithful: {faithfulness['faithful']}")
    print(f"Score:    {faithfulness['score']}")
    print(f"Reason:   {faithfulness['reason']}")
    
    
def run_rag(query):

    print("\n" + "-" * 70)
    print("STEP 1: MULTI-QUERY RETRIEVAL")
    print("-" * 70)

    reranked_results = multi_query_search(query)

    print(f"Final retrieved results: {len(reranked_results)}")

    for i, result in enumerate(reranked_results, start=1):
        print(
            f"{i}. Chunk ID: {result['chunk_id']} | "
            f"Source: {result['source']} | "
            f"Rerank Score: {result['rerank_score']:.4f}"
    )
        

    print("\n" + "-" * 70)
    print("STEP 2: BUILD CONTEXT")
    print("-" * 70)
    context = build_context(reranked_results)
    print("Context successfully created.")


    print("\n" + "-" * 70)
    print("STEP 3: GENERATE ANSWER")
    print("-" * 70)
    answer = generate_answer(query, reranked_results)
    print(answer)


    print("\n" + "-" * 70)
    print("STEP 4: FAITHFULNESS EVALUATION")
    print("-" * 70)

    faithfulness = evaluate_faithfulness(query, context, answer)
    
    print(f"\nFaithful: {faithfulness['faithful']}")
    print(f"Score:    {faithfulness['score']}")
    print(f"Reason:   {faithfulness['reason']}")


def run_retrieval_evaluation():
    print("\n\n" + "=" * 70)
    print("                 RETRIEVAL EVALUATION")
    print("=" * 70)
    evaluate_main()


def run_indexing():
    print("\n" + "=" * 70)
    print("                    DOCUMENT INDEXING")
    print("=" * 70)
    index_documents()


def main():

    while True:
        print("\n")
        print("=" * 70)
        print("                    RAG QA SYSTEM")
        print("=" * 70)

        print("\n1. Index Documents")
        print("2. Run Multi-Query")
        print("3. Run HyDE")
        print("4. Run Retrieval Evaluation")
        print("5. Exit")

        choice = input("\nEnter your choice: ").strip()

        if choice == "1":
            run_indexing()

        elif choice == "2":
            QUERY = input("Ask anything related to documents: ").strip()
            run_rag(QUERY)

        elif choice == "3":
            QUERY = input("Ask anything related to documents: ").strip()
            run_hyde(QUERY)

        elif choice == "4":
            run_retrieval_evaluation()

        elif choice == "5":
            print("\nExiting RAG QA System...")
            break

        else:
            print("\nInvalid choice. Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
