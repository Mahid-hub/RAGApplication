import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")

def evaluate_faithfulness(question, context, answer):

    context = context[:7000]
    answer = answer[:2500]

    system_prompt = """
                    You are a strict RAG faithfulness evaluator.
                    Judge ONLY whether the generated answer is supported by the retrieved context.
                    Do not use outside knowledge.

                    Important:
                    - If the answer correctly says that the requested information is not
                    available in the provided context, consider it faithful.
                    - A refusal such as "I don't have enough information in the provided
                    documents" is faithful when the context does not provide the requested
                    information.
                    - Do not penalize an answer simply because it does not answer the question,
                    as long as the refusal is appropriate based on the context.

                    Score:
                    1.0 = all claims are supported or the refusal is appropriate
                    0.5 = partially supported
                    0.0 = unsupported or hallucinated

                    Rules:
                    - If score is 1.0, faithful MUST be true.
                    - If score is less than 1.0, faithful MUST be false.
                    - The faithful field must always match the score.

                    Return valid JSON only.
                    Keep the reason short.
                """

    user_prompt = f"""
                    QUESTION:
                    {question}

                    CONTEXT:
                    {context}

                    ANSWER:
                    {answer}
                """

    try:
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
            max_completion_tokens=256,
            response_format={
                "type": "json_object"
            }
        )

        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)
        score = float(result.get("score", 0.0))
        score = max(0.0, min(1.0, score))

        faithful = score >= 1.0

        return {
            "faithful": faithful,
            "score": score,
            "reason": str(result.get("reason", "No reason provided."))
        }

    except Exception as error:
        print(f"\nFaithfulness evaluation failed: {error}")

        return {
            "faithful": False,
            "score": 0.0,
            "reason": "Faithfulness evaluator failed."
        }