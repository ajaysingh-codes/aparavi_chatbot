import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = "aparavi-rag-index"

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "text-embedding-ada-002"

# Chunk settings
CHUNK_SIZE = 512