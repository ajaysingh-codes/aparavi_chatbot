#!/usr/bin/env python
import warnings
from aparavi_chatbot.crew import run_crew_pipeline

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the entire AI workflow from user input to report generation.
    """
    user_query = input("Enter your business query: ")
    
    try:
        result = run_crew_pipeline(user_query)
        print("\nFinal Business Report:\n", result)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

if __name__ == "__main__":
    run()

