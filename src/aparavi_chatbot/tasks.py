from crewai import Task
from .tools.custom_tool import QueryAnalysisTool
from .tools.validation_tool import RetrievalValidationTool
from .agents import query_understanding_specialist, retrieval_validation_specialist, report_generation_specialist

## Task: Understand user query and Detect missing info
query_analysis_task = Task(  
    description=(
        "Analyze the user query, extract key details (such as Order ID), "
        "and refine it to ensure accurate document retrieval."
        "If an Order ID is missing, infer it from the context."
    ),
    expected_output='A refined query with all necessary details extracted.',
    tools=[QueryAnalysisTool()],
    agent=query_understanding_specialist,
    output_key="order_id"  
)

## Task: Retrieve and validate data
data_retrieval_task = Task(
    description=(
        "Retrieve relevant order/invoice documents using the extracted Order ID. "
        "Validate metadata using Pinecone DB and cross-check against expected records."
    ),
    expected_output='Validated order/invoice records with correct metadata.',
    tools=[RetrievalValidationTool()],
    async_execution=False,
    agent=retrieval_validation_specialist,
    input_keys=["order_id"],
    output_key="validated_data" 
)

## Task: Generate a Business report
generate_business_report = Task(
    description=(
        "Generate a concise business report for an order, focusing on key insights such as "
        "customer trends, product profitability, and operational efficiency. "
        "Summarize the most important takeaways in bullet points, followed by a short supporting analysis."
    ),
    expected_output='A concise business insights summary with key takeaways.',
    tools=[],
    async_execution=False,
    agent=report_generation_specialist,
)