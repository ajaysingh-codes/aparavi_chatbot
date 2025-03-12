import re
import pinecone
from openai import OpenAI
from .config import PINECONE_API_KEY, OPENAI_API_KEY, PINECONE_INDEX

class RetrievalService:
    def __init__(self):
        self.pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX)
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def extract_order_id(self, query):
        match = re.search(r"\b(?:Order\s*ID[:#-]?\s*|order\s*)?(\d{4,})\b", query, re.IGNORECASE)
        return match.group(1) if match else None

    def embed_query(self, text):
        """Generate an embedding for a given query using OpenAI."""
        query_response = self.client.embeddings.create(
            model='text-embedding-ada-002',
            input=text
        )
        return query_response.data[0].embedding

    def retrieve(self, query, top_k=5, namespace="company_docs"):
        """Retrieve relevant records from Pinecone, ensuring metadata completeness."""
        order_id = self.extract_order_id(query)
        query_vector = self.embed_query(query)
        filters = {
            "order_id": {"$eq": order_id}
        } if order_id else {}

        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            filter=filters,
            include_metadata=True
        )
        return results
