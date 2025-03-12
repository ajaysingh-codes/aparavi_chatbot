from crewai import Task
from .tools.custom_tool import QueryAnalysisTool
from .tools.validation_tool import RetrievalValidationTool
from .agents import query_understanding_specialist, retrieval_validation_specialist, report_generation_specialist

## Task: Understand user query and Detect missing info
query_analysis_task = Task(  
    description=(
        "Analyze the user query, extract key details (such as Order ID), "
        "and refine it to ensure accurate document retrieval."
    ),
    expected_output='A refined query with all necessary details extracted.',
    tools=[QueryAnalysisTool()],
    agent=query_understanding_specialist,
    output_key="order_id"  
)

## Task: Retrieve and validate data
data_retrieval_task = Task(
    description=(
        "Retrieve relevant order/invoice documents and validate metadata using Pinecone DB."
        "Cross-check retrieved records with metadata (e.g., Order ID, Document Type)."
    ),
    expected_output='Validated order/invoice records with accurate metadata.',
    tools=[RetrievalValidationTool()],
    async_execution=False,
    agent=retrieval_validation_specialist,
    output_key="validated_data" 
)

## Task: Generate a Business report
generate_business_report = Task(
    description=(
        "Generate a structured business report based on the validated order records. "
        "Include key insights, financial details, and relevant transaction information."
    ),
    expected_output='A professionally formatted business report.',
    tools=[],
    async_execution=False,
    agent=report_generation_specialist,
)