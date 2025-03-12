# Multi-Agent RAG System with CrewAI, Pinecone and DeepEval

**Tools used**: CrewAI, Pinecone, Chonkie, DeepEval, Streamlit (Basic UI), HuggingFace (Company Documents Dataset) 

## Project Overview

This project implements an end-to-end RAG chatbot that utilizes CrewAI for agentic task execution, Pinecone as the vector database, and DeepEval for evaluation. This system enables efficient information retrieval from a large dataset using chunking strategies, metadata filtering and prompt engineering. 


### Why Multi-Agent for Semi-structured Data?

Semi-structured data like shipping orders presents unique challenges:
- Information is organized in a consistent format but not in rigid database-like structures
- Contextual understanding is required to extract meaningful insights
- Business analysis requires domain expertise and specialized reasoning

## Architecture

The system uses a three-tier architecture:
1. **Document Processing Layer**
    - Advanced chunking with Chonkie
    - Vector embedding and indexing with Pinecone

2. **Agent Orchestration Layer**
    - Multi-agent coordination with CrewAI
    - Specialized agents for query understanding, data retrieval and report generation

3. **Evaluation Layer**
    - Comprehensive testing with DeepEval
    - Metrics for both retrieval accuracy and generation quality

## Components

### Dataset

The system uses a semi-structured shipping orders dataset containing:
- Order details (ID, dates, shipping status)
- Customer information
- Product listings with pricing
- Regional shipping information

I chose this dataset specifically from HuggingFace because it represents a real-word business scenario where multiple specialized skills (query parsing, data retrieval, business analysis) are needed to extract actionable insights. 

### Chunking Strategy: Why Recursive?
I experimented with various chunking strategies including semantic, sentence, and recursive chunking to optimize records retrieval. While semantic chunking proved effective for structured content such as Plain text, recursive chunking offered more balanced chunk sizes and better meaning preservation for "Company Documents" dataset.

**Implementation**: The `ChunkingService` class in `chunking_service.py` supports multiple chunking strategies, with recursive chunking set as default. 

### Vector Storage with Pinecone
I chose Pinecone for storing the chunked document embeddings due to:
- Efficient vector search with metadata filtering
- Low-latency retrieval

**Implementation**: The `IndexingService` class in `indexing_service.py` indexes document chunks in Pinecone leveraging parallel execution for faster indexing. The `RetrievalService` retrieves context using vector similarity search and metadata filtering, ensuring precise query resolution.

During testing, we leveraged namespaces within our Pinecone index to enable logical separation of datasets per agent for better scalability and retrieval efficiency for future extensions. 

### Agentic Execution with CrewAI
I designed my CrewAI pipeline with specialized agents performing one role per agent:
- Query understanding specialist: Analyzes queries and refines search parameters
- Data Retrieval & Validation Specialist: Retrieves relevant records and verifies metadata accuracy.
- Report Generation Specialist: Generates concise and clear order insights.

**Implementation**: Agents and tasks are defined in agents.py and tasks.py, while crew.py orchestrates execution using sequential processing.
**Testing**: During testing, I experimented with an additional Manager Agent to oversee and coordinate the crews hierarchically. However, for my use case, a sequential execution approach proved to be more effective without unnecessary delegation overhead. 

### Evaluation with DeepEval
To measure the effectiveness of retrieval and generation, I integrated DeepEval with following metrics:
- Contextual Precision
- Contextual Recall
- Contectual Relevancy
- Answer Relevancy
- Faithfulness

### Evaluation Results

The evaluation results are stored in a JSON object named `evaluate_results.json`. The framework provided the following scores:
- Contextual Precision: 1.0
- Contextual Recall: 1.0
- Contextual Relevancy: 1.0
- Answer Relevancy: 0.6
- Faithfulness: 1.0

The lower score for Answer Relevancy was attributed to the format of the report generated by the language model.

## Getting Started

### Prerequisites
- Python >=3.10 and <3.13
- OpenAI API key
- Pinecone API key 

### Installation
1. Clone the repository
```bash
$ git clone https://github.com/ajaysingh-codes/aparavi_chatbot.git
$ cd aparavi_chatbot
```

2. Install the packages:
```bash
$ pip install -e .
```

3. Configure environment: Create a .env file in project root with:
```bash
$ OPENAI_API_KEY=your_openai_api_key
$ PINECONE_API_KEY=your_pinecone_api_key
```

### Running the application

To run the Streamlit web interface (from root dir):
```bash
$ python run_streamlit.py
```

To run from CLI:
```bash
$ cd src/aparavi_chatbot
$ python main.py
```

## Future Enhancements
I plan to:
- Integrate Apache Airflow for automated chunking and indexing (regular update/delete records in Pinecone is essential for this use case)
- Enhance Retrieval Filtering by improving metadata extraction
- Improve Response quality with better prompt engineering and model fine-tuning

## Contributors
- Ajay Singh


