import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")

def evaluate_faithfulness(question, context, answer):

    prompt = f"""
                You are a strict faithfulness evaluator for a RAG system.

                Your job is to determine whether the GENERATED ANSWER
                is fully supported by the RETRIEVED CONTEXT.

                IMPORTANT RULES:
                1. Use ONLY the provided context.
                2. Do NOT use outside knowledge.
                3. Check every factual claim in the generated answer.
                4. If a claim is not supported by the context, mark the answer as unfaithful.
                5. Do not judge whether the answer is well-written.
                6. Only judge whether the answer is supported by the context.
                7. Return ONLY valid JSON.
                8. Do not use markdown code blocks.

                Return exactly this format:

                {{
                    "faithful": true,
                    "score": 1.0,
                    "reason": "All factual claims in the answer are supported by the context."
                }}

                The score must be between 0.0 and 1.0.

                QUESTION:
                {question}

                RETRIEVED CONTEXT:
                {context}

                GENERATED ANSWER:
                {answer}
            """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict RAG faithfulness evaluator. "
                    "Evaluate answers only against the provided context. "
                    "Return only valid JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )
    result_text = response.choices[0].message.content.strip()

    try:
        result = json.loads(result_text)

        return {
            "faithful": bool(result["faithful"]),
            "score": float(result["score"]),
            "reason": result["reason"]
        }

    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {
            "faithful": False,
            "score": 0.0,
            "reason": "The evaluator returned an invalid response."
        }
