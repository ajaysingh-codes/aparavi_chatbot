from crewai import Agent
from .tools.custom_tool import QueryAnalysisTool
from .tools.validation_tool import RetrievalValidationTool

## Manager agent
# manager = Agent(
#     role="Project Manager",
#     goal="Efficiently manage the crew and ensure high-quality task completion.",
#     backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
#     allow_delegation=True
# )

## First agent: Query Understanding Agent
query_understanding_specialist = Agent(
    role="Query Understanding Specialist",
    goal=("Analyze user queries, detect missing details, and refine search parameters {input}"),
    verbose=True,
    memory=True,
    backstory="You specialize in understanding user intent, ensuring all necessary details (such as Order ID) are present during retrival.",
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
    role="Business Report Generator Specialist",
    goal="Generate detailed reports based on validated order records.",
    verbose=True,
    memory=True,
    backstory="You transform structured order data into clear, concise, and professional business reports.",
    tools=[],
    allow_delegation=False,
)