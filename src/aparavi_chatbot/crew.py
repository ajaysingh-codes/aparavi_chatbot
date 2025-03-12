import os
from .services.retrieval_service import RetrievalService
from crewai import Crew, Process
from .agents import query_understanding_specialist, retrieval_validation_specialist, report_generation_specialist
from .tasks import query_analysis_task, data_retrieval_task, generate_business_report
from openai import OpenAI
from .config import OPENAI_API_KEY

retrieval_service = RetrievalService()

def run_crew_pipeline(user_query):
	"""
	Executes the Crew AI pipeline based on user input.

	Returns: A business report as a string
	"""

	retrieved_docs = retrieval_service.retrieve(query=user_query, top_k=3)

	processed_docs = [
		{
			"id": match["id"],
			"text": match["metadata"]["text"],
			"score": match["score"],
			"file_name": match["metadata"]["file_name"],
			"order_id": match["metadata"]["order_id"],
		}
		for match in retrieved_docs.get("matches", [])
	]

	crew = Crew(
		agents=[query_understanding_specialist, retrieval_validation_specialist, report_generation_specialist],
		tasks=[query_analysis_task, data_retrieval_task, generate_business_report],
		process=Process.sequential,
		memory=True,
		cache=True,
		max_rpm=100,
		share_crew=True,
	)

	# Start the task execution process with enhanced feedback
	validated_data = crew.kickoff(inputs={"input": processed_docs})

	return validated_data