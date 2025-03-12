import os
import json
import itertools
from openai import OpenAI
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from ..config import PINECONE_API_KEY, OPENAI_API_KEY, PINECONE_INDEX

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKED_DOCS_PATH = os.path.join(BASE_DIR, "knowledge", "chunked_company_docs.json")

class IndexingService:
    def __init__(self):
        """Initialize Pinecone and OpenAI clients"""
        self.pc = Pinecone(api_key=PINECONE_API_KEY, pool_threads=30)
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.index_name = PINECONE_INDEX

        # Checks if the index already exists in Pinecone
        if self.index_name not in [idx['name'] for idx in self.pc.list_indexes()]:
            print(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region='us-east-1'
                )
            )
        self.index = self.pc.Index(self.index_name)

    def get_embedding(self, text):
        """Generate embeddings for chunked text using OpenAI"""
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

    def index_documents(self, chunked_data):
        """Index chunked documents into Pinecone"""
        vectors = []
        for chunk in tqdm(chunked_data, desc="Generating embeddings"):
            embedding = self.get_embedding(chunk["chunk_text"])
            vectors.append({
                "id": chunk["id"],
                "values": embedding,
                "metadata": {
                    "file_name": chunk["file_name"],
                    "document_type": chunk["document_type"],
                    "order_id": chunk["order_id"],
                    "text": chunk["chunk_text"]
                }
            })

        # Helper function for Batching
        def chunks(iterable, batch_size=100):
            it = iter(iterable)
            chunk = tuple(itertools.islice(it, batch_size))
            while chunk:
                yield chunk
                chunk = tuple(itertools.islice(it, batch_size))

        # Load chunked wikipedia data
        with self.index as idx:
            async_results = [idx.upsert(vectors=chunk, namespace="company_docs", async_req=True) for chunk in chunks(vectors, 100)]
            [async_result.get() for async_result in async_results]

        print(f"Indexed {len(vectors)} chunks to Pinecone index {self.index_name}")

# Testing indexing service
if __name__ == "__main__":
    with open(CHUNKED_DOCS_PATH, 'r') as f:
        chunked_data = json.load(f)
    indexing_service = IndexingService()
    indexing_service.index_documents(chunked_data)