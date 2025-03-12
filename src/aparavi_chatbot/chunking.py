import re
import json
from datasets import load_dataset
from chonkie import SemanticChunker, RecursiveChunker, SentenceChunker, RecursiveRules
from tqdm import tqdm

# Load wikipedia dataset from Hugging Face
dataset = load_dataset("AyoubChLin/CompanyDocuments", split="train")

# Extract relevant text from wikipedia dataset
chunked_data = []
max_documents = 50

# Choose chunker type: 'sentence' for SentenceChunker, 'recursive' for RecursiveChunker
CHUNKER_TYPE = 'recursive'

# Initialize the chosen chunker
if CHUNKER_TYPE == 'sentence':
    chunker = SentenceChunker(
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
elif CHUNKER_TYPE == 'recursive':
    chunker = RecursiveChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        rules=RecursiveRules(),
        min_characters_per_chunk=12,
        return_type="chunks"
    )
else:
    raise ValueError("Invalid CHUNKER_TYPE. Choose 'sentence' or 'recursive'.")


for i, doc in enumerate(dataset):
    if i >= max_documents:
        break

    content = doc["file_content"]
    file_name = doc["file_name"]
    doc_type = doc["document_type"]
    order_id = re.search(r"Order ID:\s*(\d+)", content)

    chunks = chunker(content)
    for j, chunk in enumerate(chunks):
        chunked_data.append({
            "id": f"company_doc_{i}_chunk{j}",
            "chunk_text": str(chunk),
            "file_name": file_name,
            "document_type": doc_type,
            "order_id": order_id.group(1) if order_id else None,
        })
     
# Save chunked data as JSON
with open("chunked_company_docs.json", "w") as f:
    json.dump(chunked_data, f, indent=4)

print(f"Saved {len(chunked_data)} chunks to chunked_company_docs.json")