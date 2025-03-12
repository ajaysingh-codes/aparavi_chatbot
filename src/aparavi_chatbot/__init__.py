"""
A multi-agent system for business report generation.
"""

# Make crew pipeline accessible from top level
from .crew import run_crew_pipeline
from .config import PINECONE_API_KEY, PINECONE_INDEX, OPENAI_API_KEY, OPENAI_MODEL, CHUNK_SIZE