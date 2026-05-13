import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Recipe and Diet Assistant",
    page_icon="🍳",
    layout="wide"
)

# Load environment
load_dotenv()

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL")

# Streamlit UI
st.title("🍳 Recipe and Diet Assistant")
st.write("Ask me anything about recipes or dietary advice!")
st.info("💾 **Note:** Responses are cached at the backend. Repeated queries will respond instantly without using API quota.")

# Check backend health
try:
    response = requests.get(f"{BACKEND_URL}/health", timeout=5)
    if response.status_code != 200:
        st.error("⚠️ Backend is not responding. Please start the backend with: `python backend.py`")
        st.stop()
except requests.exceptions.ConnectionError:
    st.error("❌ Cannot connect to backend. Please start it with: `python backend.py`")
    st.stop()
except Exception as e:
    st.error(f"❌ Error connecting to backend: {str(e)}")
    st.stop()

# Input and chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if query := st.chat_input("What would you like to know?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.write(query)

    # Process query
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Call backend API
                response = requests.post(
                    f"{BACKEND_URL}/api/v1/query",
                    json={"query": query},
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    result = data["result"]
                    docs = data["source_documents"]
                    retrieval_time = data["retrieval_time"]
                    
                    # Display response
                    st.write(result)
                    
                    # Display source documents
                    with st.expander("📚 Retrieved Documents"):
                        for doc in docs:
                            st.markdown(f"**{doc['index']}. {doc['title']}**")
                            st.write(doc['content'])
                    
                    st.caption(f"⏱️ Retrieval Time: {retrieval_time:.2f} seconds")
                    
                    # Add to chat history
                    st.session_state.messages.append({"role": "assistant", "content": result})
                
                else:
                    error_msg = response.json().get("detail", "Unknown error")
                    st.error(f"Backend error: {error_msg}")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Is it running? Start it with: `python backend.py`")
            except requests.exceptions.Timeout:
                st.error("⏱️ Backend request timed out. Try again.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")