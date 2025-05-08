from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain import LLMMathChain
from PyDictionary import PyDictionary

# 1. Load and chunk documents
loader = DirectoryLoader("docs/", glob="*.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 2. Create vector store (local embeddings)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 3. Define RAG function with Ollama
llm = Ollama(model="mistral")  # or "mistral"

def rag_pipeline(query):
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Answer based on: {context}\n\nQuestion: {query}"
    return llm(prompt)

# 4. Define tools
calculator = LLMMathChain.from_llm(llm)
dictionary = PyDictionary()

tools = [
    Tool(
        name="Calculator",
        func=calculator.run,
        description="Use for math calculations"
    ),
    Tool(
        name="Dictionary",
        func=lambda word: str(dictionary.meaning(word)),
        description="Use for word definitions"
    ),
    Tool(
        name="RAG Q&A",
        func=rag_pipeline,
        description="Use for general questions"
    )
]

# 5. Create agent
# Update the agent creation part:
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,  # Critical for Mistral
    max_iterations=3,  # Prevents infinite loops
    early_stopping_method="generate"  # Better for smaller models
)