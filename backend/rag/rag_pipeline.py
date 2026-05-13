import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Load FAISS index + metadata
print("📥 Loading FAISS index...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
import os
print(f"Checking directory: {os.path.abspath('data/index_5k')}")
print(f"Files in data/index_5k: {os.listdir('data/index_5k')}")
try:
    db = FAISS.load_local(
        "data/index_5k",
        embedding_model,
        allow_dangerous_deserialization=True
    )
except FileNotFoundError:
    print("Error: FAISS index not found in 'data/index_5k'. Please create the index first.")
    exit(1)
except Exception as e:
    print(f"Error loading FAISS index: {str(e)}")
    exit(1)

# Load Gemini Pro / Flash
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
except Exception as e:
    print(f"Error initializing Gemini model: {str(e)}")
    exit(1)

# Prompt template
template = """You are a helpful Recipe and Diet Assistant.
Use the context below to answer the user's query.
If the answer isn't found in the context, say "Sorry, I couldn't find a relevant recipe."

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Create RAG pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Chat loop
print("🤖 Gemini RAG Recipe Assistant ready!")
while True:
    try:
        query = input("\nAsk a recipe/diet question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        if not query.strip():
            print("Please enter a valid query.")
            continue
        start_time = time.time()
        result = qa_chain.invoke({"query": query})
        end_time = time.time()
        print(f"\n🧾 Answer:\n{result['result']}")
        print(f"Retrieved Documents: {result['source_documents']}")
        print(f"Retrieval time: {end_time - start_time:.2f} seconds")
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {str(e)}")