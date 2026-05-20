import asyncio

from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness

print("🎯 Initializing local Async Engines via OpenAI compatibility layer...")

# 1. Setup  Local Async Client
local_ollama_client = AsyncOpenAI(
    api_key="ollama", base_url="http://localhost:11434/v1"
)

# 2. Build the exact InstructorLLM instances Ragas expects
prometheus_judge = llm_factory(
    model="ggozad/prometheus2", provider="openai", client=local_ollama_client
)

local_embeddings = embedding_factory(
    model="llama3.2:latest ", provider="openai", client=local_ollama_client
)

# 3. Create metric objects cleanly
print("🔧 Structuring evaluation metrics...")
faithfulness_metric = Faithfulness(llm=prometheus_judge)
context_precision_metric = ContextPrecision(llm=prometheus_judge)
answer_relevance_metric = AnswerRelevancy(
    llm=prometheus_judge, embeddings=local_embeddings
)

# 4. Define  test sample data explicitly using the new keyword architecture
user_query = "How do we automate cloud image safety reviews?"
retrieved_chunks = [
    "The system uses an AWS S3 bucket trigger to fire an AWS Lambda function. "
    "This Lambda function calls Amazon Rekognition to detect unsafe content."
]
llm_generated_answer = "We handle image safety by triggering an AWS Lambda function from S3. The function uses Amazon Rekognition to scan the image."
ground_truth_reference = "Automated image safety reviews are handled using an AWS SAM pipeline featuring S3 triggers, AWS Lambda, Amazon Rekognition for content analysis, and SNS notifications."


async def run_evaluation():
    print("\n⚖️ Prometheus 2 is actively processing your metrics locally...")

    try:
        print("⏳ Calculating Faithfulness...")
        score_faith = await faithfulness_metric.ascore(
            user_input=user_query,
            response=llm_generated_answer,
            retrieved_contexts=retrieved_chunks,
        )
        print(f"✅ Faithfulness Score: {score_faith}\n")
    except Exception as e:
        print(f"⚠️ Faithfulness failed: {e}\n")

    try:
        print("⏳ Calculating Context Precision...")
        score_precision = await context_precision_metric.ascore(
            user_input=user_query,
            retrieved_contexts=retrieved_chunks,
            reference=ground_truth_reference,
        )
        print(f"✅ Context Precision Score: {score_precision}\n")
    except Exception as e:
        print(f"⚠️ Context Precision failed: {e}\n")

    try:
        print("⏳ Calculating Answer Relevance...")
        score_relevance = await answer_relevance_metric.ascore(
            user_input=user_query, response=llm_generated_answer
        )
        print(f"✅ Answer Relevance Score: {score_relevance}\n")
    except Exception as e:
        print(f"⚠️ Answer Relevance failed: {e}\n")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
