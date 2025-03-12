import streamlit as st
from crew import run_crew_pipeline

st.title("Multi-agent System to generate business reports")

query = st.text_input("Enter your query (e.g., 'Retrieve order details for Order ID 10560')")

if st.button("Generate Report"):
    with st.spinner("Generating insights..."):
        report = run_crew_pipeline(query)
        st.subheader("Business report")
        st.write(report)

        st.download_button("Download report", report, file_name="business_report.txt")

st.markdown("---")
st.write("Powered by CrewAI and DeepEval")