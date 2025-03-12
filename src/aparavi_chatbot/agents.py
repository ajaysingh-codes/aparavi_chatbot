from crewai import Agent
from .tools.custom_tool import QueryAnalysisTool
from .tools.validation_tool import RetrievalValidationTool

## First agent: Query Understanding Agent
query_understanding_specialist = Agent(
    role="Query Understanding Specialist",
    goal=("Extract Order ID and key parameters from the user query and refine search parameters {input}"),
    verbose=True,
    memory=True,
    backstory="You specialize in parsing user queries, ensuring key details (like Order ID), and ensuring all necessary details are present before retrival.",
    tools=[QueryAnalysisTool()],
    allow_delegation=True,
)

## Second agent: Data Retrieval & Validation agent
retrieval_validation_specialist = Agent(
    role="Data Extraction & Validation Specialist",
    goal="Fetch relevant order/invoice records from Pinecone, validate metadata and cross-check details. {input}",
    verbose=True,
    memory=True,
    backstory="You ensure retrieved records are accurate, properly formatted and match the expected business metadata.",
    tools=[RetrievalValidationTool()],
    allow_delegation=True,
)

## Third agent: Report Generation Agent
report_generation_specialist = Agent(
    role="Business Intelligence Analyst",
    goal="Generate concise, insight-driven reports highlighting key business insights from order data.",
    verbose=True,
    memory=True,
    backstory=(
        "You are an expert in business intelligence, skilled at extracting actionable insights "
        "from semi-structured order data. Your reports must be concise, focusing on profitability, logistics efficiency, "
        "customer trends, and operational recommendations."
    ),
    tools=[],
    allow_delegation=False,
)

## Manager agent
# manager = Agent(
#     role="Project Manager",
#     goal="Efficiently manage the crew and ensure high-quality task completion.",
#     backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
#     allow_delegation=True
# )