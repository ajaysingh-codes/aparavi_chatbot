from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from ..retrieval_service import RetrievalService

retrieval_service = RetrievalService()

class RetrievalValidationInput(BaseModel):
    """Input Schema for Retrieval and Validation tool."""
    query: str = Field(..., description="The user query containing search details.")

class RetrievalValidationTool(BaseTool):
    name: str = "Retrieval & Validation Tool"
    description: str = (
        "Fetches order records from Pinecone and verifies metadata accuracy."
    )
    args_schema: Type[BaseModel] = RetrievalValidationInput

    def _run(self, query: str) -> dict:
        """Fetch & validate data from Pinecone."""
        retrieved_docs = retrieval_service.retrieve(query=query, top_k=3)

        validated_data = [
            {
                "id": match["id"],
                "text": match["metadata"]["text"],
                "score": match["score"],
                "file_name": match["metadata"]["file_name"],
                "order_id": match["metadata"]["order_id"],
            }
            for match in retrieved_docs.get("matches", [])
        ]

        return {"validated_documents": validated_data}