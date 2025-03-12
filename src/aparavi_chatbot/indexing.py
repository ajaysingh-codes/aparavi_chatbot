import json
import itertools
from openai import OpenAI
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from .config import PINECONE_API_KEY, OPENAI_API_KEY, PINECONE_INDEX

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, pool_threads=30)
client = OpenAI(api_key=OPENAI_API_KEY)

# Checks if the index already exists in Pinecone
if PINECONE_INDEX in [idx['name'] for idx in pc.list_indexes()]:
    print(f"Index {PINECONE_INDEX} already exists in Pinecone. Skipping index creation.")
else:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region='us-east-1'
        )
    )

# Connect to the index
index = pc.Index(PINECONE_INDEX)

with open('chunked_company_docs.json', 'r') as f:
    chunked_data = json.load(f)

# Generate embeddings for chunked text using OpenAI
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Prepate data with embeddings for Pinecone
vectors = []
for chunk in tqdm(chunked_data, desc="Generating embeddings"):
    embedding = get_embedding(chunk["chunk_text"])
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
with pc.Index('aparavi-rag-index', pool_threads=30) as index:
    async_results = [index.upsert(vectors=chunk, namespace="company_docs", async_req=True) for chunk in chunks(vectors, 100)]
    [async_result.get() for async_result in async_results]

print(f"Indexed {len(vectors)} chunks to Pinecone index {PINECONE_INDEX}")