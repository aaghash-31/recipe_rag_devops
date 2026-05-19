import os
import time
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file. Get one from https://console.groq.com")

set_llm_cache(InMemoryCache())

DEFAULT_PROMPT_TEMPLATE = """You are a helpful Recipe and Diet Assistant.
Use the context below to answer the user's query to the best of your ability.
If the answer isn't clear in the context, provide a recipe or answer only based on LLM knowledge. DO NOT HALLUCINATE or say 'I have limited information, but here's a suggestion.'

Context:
{context}

Question:
{question}

Answer:"""

class RagService:
    def __init__(
        self,
        index_path: str = "vectorstore/index_5k",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant",
        temperature: float = 0.3,
        k: int = 3,
    ):
        self.index_path = index_path
        self.embedding_model_name = embedding_model_name
        self.llm_model = llm_model
        self.temperature = temperature
        self.k = k
        self.llm = None
        self.retriever = None
        self.prompt_template = None

    def initialize(self):
        logger.info("Initializing embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={
                "device": "cpu",
                "trust_remote_code": True,
                "local_files_only": True
            },
            cache_folder="/app/hf_cache"
        )

        logger.info("Loading FAISS index...")
        try:
            db = FAISS.load_local(
                self.index_path,
                embedding_model,
                allow_dangerous_deserialization=True,
            )
            logger.info("FAISS index loaded successfully")
        except FileNotFoundError:
            raise FileNotFoundError(f"FAISS index not found in '{self.index_path}'")
        except Exception as e:
            raise RuntimeError(f"Error loading FAISS index: {str(e)}")

        logger.info("Initializing Groq LLM...")
        try:
            self.llm = ChatGroq(
                model=self.llm_model,
                temperature=self.temperature,
                api_key=api_key,
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing Groq model: {str(e)}")

        self.prompt_template = PromptTemplate(
            template=DEFAULT_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )
        self.retriever = db
        logger.info("RAG service initialized")

    def query(self, query_text: str) -> dict:
        if self.llm is None or self.retriever is None or self.prompt_template is None:
            raise RuntimeError("RAG service has not been initialized")

        start_time = time.time()
        docs = self.retriever.similarity_search(query_text, k=self.k)
        context = "\n\n".join(
            f"Source: {doc.metadata.get('title', 'Unknown')}\n{doc.page_content}"
            for doc in docs
        )

        prompt_text = self.prompt_template.format(context=context, question=query_text)
        messages = [
            SystemMessage(content="You are a helpful Recipe and Diet Assistant."),
            HumanMessage(content=prompt_text),
        ]

        response = self.llm.invoke(messages)
        answer = getattr(response, "content", None)
        if answer is None:
            answer = response.generations[0][0].message.content

        retrieval_time = time.time() - start_time
        source_documents = [
            {
                "index": i + 1,
                "content": doc.page_content,
                "title": doc.metadata.get("title", "Unknown"),
            }
            for i, doc in enumerate(docs)
        ]

        return {
            "result": answer,
            "source_documents": source_documents,
            "retrieval_time": retrieval_time,
        }
