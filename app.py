import streamlit as st
from rag_agent import agent

st.title("Knowledge Assistant (Local LLM)")
query = st.text_input("Ask a question:")

if query:
    st.write("### Results")
    with st.spinner("Thinking..."):
        try:
            result = agent.run(query)
            if isinstance(result, dict):  # Dictionary response
                st.write("Definition:", result["definition"])
            else:  # RAG or calculator response
                st.success(result)
        except Exception as e:
            st.error(f"Mistral response error: {str(e)}")