from crewai import Agent
from aparavi_chatbot.tools.validation_tool import RetrievalValidationTool

## First agent: Query Understanding Agent
query_understanding_specialist = Agent(
    role="Query Understanding Specialist",
    goal=("Extract key search parameters from user queries and organize them for effective retrieval {input}"),
    verbose=True,
    memory=True,
    backstory=("You specialize in natural language understanding, extracting structured information from unstructured or semi-structured text"
               "Your expertise lies in identifying search intentions and parameters from conversational queries."
    ),
    tools=[],
    allow_delegation=True,
)

## Second agent: Data Retrieval & Validation agent
retrieval_validation_specialist = Agent(
    role="Data Extraction & Validation Specialist",
    goal="Retrieve relevant order/invoice records using search parameters from Pinecone, validate their accuracy and relevance. {input}",
    verbose=True,
    memory=True,
    backstory="You're an expert at information retrieval. Using provided search parameters, you can find, filter, and validate the most relevant documents from the database, ensuring they meet the user's information needs.",
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