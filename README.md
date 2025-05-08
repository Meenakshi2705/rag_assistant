# RAG Multi-Agent Assistant

## How to Run
1. Install dependencies: `pip install langchain faiss-cpu sentence-transformers streamlit PyDictionary`
2. Pull an Ollama model: `ollama pull llama3`
3. Add documents to `docs/` folder
4. Run: `streamlit run app.py`

## Usage
- Ask questions in the Streamlit interface.
- The system routes to RAG, Calculator, or Dictionary automatically.