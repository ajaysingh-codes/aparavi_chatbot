import json
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.test_case import LLMTestCase
from .services.retrieval_service import RetrievalService
from .crew import run_crew_pipeline

# Sample test case for Order ID 10560
TEST_QUERIES = [
    {
        "query": "Retrieve order details for Order ID 10560",
        "expected_context": [
            "Order ID: 10560",
            "Ship Name: Frankenversand",
            "Ship Address: Berliner Platz 43",
            "Ship City: München",
            "Ship Region: Western Europe",
            "Ship Postal Code: 80805",
            "Ship Country: Germany",
            "Customer ID: FRANK",
            "Customer Name: Frankenversand",
            "Employee Name: Laura Callahan",
            "Shipper Name: Speedy Express",
            "Order Date: 2017-06-06",
            "Shipped Date: 2017-06-09",
            "Product: Nord-Ost Matjeshering, Quantity: 20, Unit Price: 25.89, Total: 517.8",
            "Product: Tarte au sucre, Quantity: 15, Unit Price: 49.3, Total: 739.5",
            "Total Price: 1257.3"
        ]
    }
]

# Initialize retrieval service
retrieval_service = RetrievalService()

def retrieve_docs(query, top_k=3):
    """Retrieve relevant chunks from Pinecone."""
    retrieved_docs = retrieval_service.retrieve(query=query, top_k=top_k)
    return [match["metadata"]["text"] for match in retrieved_docs.get("matches", [])]

def evaluate_retrieval(query, retrieved_content, expected_context):
    """Evaluate retrieval quality using DeepEval metrics."""
    test_case = LLMTestCase(
        input=query,
        actual_output=retrieved_content,  
        expected_output="\n".join(expected_context), 
        retrieval_context=retrieved_content  
    )

    # Initialize metrics
    contextual_precision = ContextualPrecisionMetric()
    contextual_recall = ContextualRecallMetric()
    contextual_relevancy = ContextualRelevancyMetric()

    # Measure scores
    contextual_precision.measure(test_case)
    contextual_recall.measure(test_case)
    contextual_relevancy.measure(test_case)

    return {
        "contextual_precision_score": float(contextual_precision.score),
        "Reason:": contextual_precision.reason,
        "contextual_recall_score": float(contextual_recall.score), 
        "Reason:": contextual_recall.reason,
        "contextual_relevancy_score": float(contextual_relevancy.score),
        "Reason:": contextual_relevancy.reason
    }

def evaluate_chatbot_res(query, response, retrieved_content):
    """Evaluate chatbot response using Answer Relevancy and Faithfulness."""
    test_case = LLMTestCase(
        input=query,
        actual_output=response, 
        expected_output="\n".join(retrieved_content),  
        retrieval_context=retrieved_content  
    )

    answer_relevancy = AnswerRelevancyMetric()
    faithfulness = FaithfulnessMetric()

    answer_relevancy.measure(test_case)
    faithfulness.measure(test_case)

    return {
        "answer_relevancy_score": float(answer_relevancy.score),
        "Reason:": answer_relevancy.reason,
        "faithfulness_score": float(faithfulness.score), 
        "Reason:": faithfulness.reason
    }

def run_eval():
    """Runs evaluation for retrieval and chatbot response."""
    evaluation_results = []

    for test in TEST_QUERIES:
        query = test["query"]
        expected_context = test["expected_context"]

        # Step 1: Retrieve relevant documents
        retrieved_docs = retrieve_docs(query)
        retrieved_content = retrieved_docs  

        # Step 2: Generate chatbot response (business report)
        chatbot_res = str(run_crew_pipeline(query))

        # Step 3: Evaluate retrieval performance
        retrieval_scores = evaluate_retrieval(query, retrieved_content, expected_context)

        # Step 4: Evaluate chatbot response quality
        generation_scores = evaluate_chatbot_res(query, chatbot_res, retrieved_content)

        # Store results
        evaluation_results.append({
            "query": query,
            "retrieved_content": retrieved_content,
            "chatbot_res": chatbot_res,
            "metrics": {**retrieval_scores, **generation_scores}
        })

    # Save evaluation results
    with open("evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)

    print("Evaluation completed. Results saved to evaluation_results.json")

if __name__ == "__main__":
    run_eval()
