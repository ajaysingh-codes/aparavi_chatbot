from setuptools import setup, find_packages

setup(
    name="aparavi_chatbot",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "crewai>=0.105.0",
        "pinecone-client",
        "openai",
        "streamlit",
        "pydantic",
        "python-dotenv",
        "chonkie",
        "datasets",
        "tqdm",
        "deepeval",
    ],
    entry_points={
        "console_scripts": [
            "aparavi-chatbot=aparavi_chatbot.main:run",
        ],
    },
)