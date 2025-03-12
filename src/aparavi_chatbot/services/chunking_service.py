import re
import json
from datasets import load_dataset
from chonkie import SemanticChunker, RecursiveChunker, SentenceChunker, RecursiveRules

class ChunkingService:
    def __init__(self, chunking_type="recursive", max_docs=50):
        """Initialize the chunking service with a specific chunking strategy"""
        self.chunking_type = chunking_type
        self.max_docs = max_docs
        self.knowledge_path = "../../../knowledge/chunked_company_docs.json"

        # Initialize the chosen chunker
        if chunking_type == 'sentence':
            self.chunker = SentenceChunker(
                tokenizer_or_token_counter="gpt2",
                chunk_size=512,
                chunk_overlap=0,
                min_sentences_per_chunk=1,
                min_characters_per_sentence=12,
                approximate=True,
                delim=[".", "?", "!", "\n"],
                include_delim="prev",
                return_type="chunks"
            )
        elif chunking_type == 'recursive':
            self.chunker = RecursiveChunker(
                tokenizer_or_token_counter="gpt2",
                chunk_size=512,
                rules=RecursiveRules(),
                min_characters_per_chunk=12,
                return_type="chunks"
            )
        else:
            raise ValueError("Invalid CHUNKER_TYPE. Choose 'sentence' or 'recursive'.")

    def chunk_documents(self):
        """Load and chunk company documents from dataset."""
        dataset = load_dataset("AyoubChLin/CompanyDocuments", split="train")
        chunked_data = []

        for i, doc in enumerate(dataset):
            if i >= self.max_docs:
                break

            content = doc["file_content"]
            file_name = doc["file_name"]
            doc_type = doc["document_type"]
            order_id = re.search(r"Order ID:\s*(\d+)", content)

            chunks = self.chunker(content)
            for j, chunk in enumerate(chunks):
                chunked_data.append({
                    "id": f"company_doc_{i}_chunk{j}",
                    "chunk_text": str(chunk),
                    "file_name": file_name,
                    "document_type": doc_type,
                    "order_id": order_id.group(1) if order_id else None,
                })

        # Save chunked data as JSON
        with open(self.knowledge_path, "w") as f:
            json.dump(chunked_data, f, indent=4)

        print(f"Saved {len(chunked_data)} chunks to chunked_company_docs.json")
        return chunked_data
    
# Test chunking service
if __name__ == "__main__":
    chunk_service = ChunkingService()
    chunk_service.chunk_documents()
