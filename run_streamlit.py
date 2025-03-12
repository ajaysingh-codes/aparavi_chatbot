import sys
import os
import subprocess

if __name__ == "__main__":
    try:
        # Try importing streamlit to check if it's installed
        import streamlit
        sys.path.append(os.path.abspath("src"))
        
        subprocess.run(["streamlit", "run", "src/aparavi_chatbot/app.py"])
    except ImportError:
        print("Streamlit is not installed. Installing now...")
        subprocess.run(["pip", "install", "streamlit"])
        print("Streamlit has been installed. Please run this script again.")