from setuptools import setup, find_packages

setup(
    name="aparavi_chatbot",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "crewai>=0.105.0",
        "chonkie>=0.5.1",
        "deepeval>=0.12.0",
        "pinecone>=6.0.1",
        "openai>=1.65.0",
        "python-dotenv>=1.0.0",
        "streamlit>=1.30.0",
        "datasets>=3.3.0",
        "pydantic>=2.10.0",
        "tqdm>=4.67.0",
    ],
    entry_points={
        "console_scripts": [
            "aparavi-chatbot=aparavi_chatbot.main:run",
        ],
    },
)