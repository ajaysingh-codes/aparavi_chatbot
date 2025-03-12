from crewai import Task
from aparavi_chatbot.agents import (
    query_understanding_specialist, 
    retrieval_validation_specialist, 
    report_generation_specialist)
from aparavi_chatbot.tools.validation_tool import RetrievalValidationTool

## Task: Understand user query and Detect missing info
query_analysis_task = Task(  
    description=(
        "Analyze the user query to extract key search parameters and return them as a structured dictionary. "
        "Parameters to extract should include (when present):\n"
        "- order_id: Any 4-5 digit number that appears to be an order identifier\n"
        "- customer_name: Name of any customer mentioned\n"
        "- product_name: Name of specific products mentioned\n"
        "- ship_region: Any geographical region, country, or area mentioned\n"
        "Return these in a JSON structure with these exact key names. If a parameter isn't present, include the key with a null value."
    ),
    expected_output='A structured dictionary of extracted search parameters.',
    agent=query_understanding_specialist,
    output_key="search_params"  
)

## Task: Retrieve and validate data
data_retrieval_task = Task(
    description=(
        "Using the extracted search parameters, construct an effective search strategy to retrieve the most relevant documents."
        "IMPORTANT: When a regional query (like 'Western Europe') is detected, ensure to retrieve "
        "documents from EXACTLY THE REGION specified. Do not substitute with other regions."
        "Review these steps:\n"
        "1. Review the search parameters provided\n"
        "2. Prioritize exact matches on Order ID when available\n"
        "3. For other parameters, use them to enhance the semantic search\n"
        "4. Look for region information in the text of each document and ensure it matches\n"
        "5. Return a list of most relevant documents, with their metadata\n\n"
        "The search system works best when you combine exact filters (like Order ID) with semantic search for other parameters. "

    ),
    expected_output='A list of Validated documents sorted by relevance, with full metadata and content.',
    tools=[RetrievalValidationTool()],
    async_execution=False,
    agent=retrieval_validation_specialist,
    input_keys=["search_params"],
    output_key="validated_data" 
)

## Task: Generate a Business report
generate_business_report = Task(
    description=(
        "Generate a concise business report for user query, focusing on key insights such as "
        "customer trends, product profitability, and operational efficiency. "
        "Summarize the most important takeaways in bullet points, followed by a short supporting analysis."
    ),
    expected_output='A concise business insights summary with key takeaways.',
    tools=[],
    async_execution=False,
    agent=report_generation_specialist,
)