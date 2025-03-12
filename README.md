# AparaviChatbot - Multi-Agent RAG System

An agentic Retrieval-Augmented Generation (RAG) system that generates business insights from order/invoice data using a collaborative multi-agent approach.

## Project Overview

This project demonstrates the power of multi-agent AI systems for handling complex business intelligence tasks. The system processes semi-structured order/invoice data, extracts meaningful patterns, and generates actionable business insights through a coordinated team of specialized AI agents.


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

## Getting Started
### Prerequisites
- Python >=3.10 and <3.13
- OpenAI API key
- Pinecone API key 

### Installation
1. Clone the repository
```bash
$ git clone https://github.com/yourusername/aparavi_chatbot.git
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

To run the Streamlit web interface:
```bash
$ python run_streamlit.py
```

To run from CLI:
```bash
$ cd src/aparavi_chatbot
$ python main.py
```

## Components

### Dataset

The system uses a semi-structured shipping orders dataset containing:
- Order details (ID, dates, shipping status)
- Customer information
- Product listings with pricing
- Regional shipping information

I chose this dataset specifically from HuggingFace because it represents a real-word business scenario where multiple specialized skills (query parsing, data retrieval, business analysis) are needed to extract actionable insights. 


## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
$ crewai run
```


