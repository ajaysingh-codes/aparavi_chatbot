from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import re

class QueryAnalysisToolInput(BaseModel):
    """Schema for extracting metadata from a query."""
    query: str = Field(..., description="The user query from which to extract details.")

class QueryAnalysisTool(BaseTool):
    name: str = "Query Analysis Tool"
    description: str = (
        "Extracts metadata (such as Order ID) from a user query to enhance retrieval."
    )
    args_schema: Type[BaseModel] = QueryAnalysisToolInput

    def _run(self, query: str) -> dict:
        # Extract metadata like Order ID from query.
        match = re.search(r"\b(?:Order\s*ID[:#-]?\s*|order\s*)?(\d{4,})\b", query, re.IGNORECASE)
        order_id = match.group(1) if match else None

        return {"order_id": order_id, "original_query": query}
