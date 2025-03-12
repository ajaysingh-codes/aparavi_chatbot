import streamlit as st
from aparavi_chatbot.crew import run_crew_pipeline

st.title("Multi-agent AI System to generate business reports")

query = st.text_input("Enter your query (e.g., 'Retrieve order details for Order ID 10560')")

if st.button("Generate Report"):
    with st.spinner("Generating insights..."):
        result = run_crew_pipeline(query)
        st.subheader("Business report")
        
        if isinstance(result, dict) and "raw" in result:
            raw_text = result["raw"]
            st.code(raw_text, language=None)
        else:
            st.text(str(result))

st.markdown("---")
st.write("Powered by CrewAI and DeepEval")