import streamlit as st
import os
import time
import functools
import uuid
from dotenv import load_dotenv
from typing import List, TypedDict, Optional, Dict, Any

# --- Core Logic Imports ---
import chromadb
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field # Using Pydantic V2

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# --- Imports needed for Ingestion (moved from ingest.py) ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. Load Environment Variables ---
# Load the .env file (which contains OPENAI_API_KEY)
# Streamlit Cloud will use its own Secrets Management, but this is good for local
load_dotenv()

# --- 2. Define Pydantic Model for LLM Output ---
class BotResponse(BaseModel):
    """The structured response from the Policy Bot."""
    answer: str = Field(description="The final answer to the user's question.")
    source_document: str = Field(
        description="The 'source' filename from the metadata (e.g., 'hr_policy.txt')."
    )

# --- 3. Define Langgraph State ---
class GraphState(TypedDict):
    question: str
    documents: List[str]
    response: Optional[BotResponse]
    metrics: Dict[str, Any] # For our profiling
    
# --- 4. Global Setup (Models & DB) ---

@st.cache_resource
def get_llm():
    """Cached function to get the LLM."""
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

@st.cache_resource
def get_embedding_model():
    """Cached function to get the embedding model."""
    return OpenAIEmbeddings()

@st.cache_resource
def get_retriever():
    """
    Cached function to create and populate the in-memory ChromaDB.
    This combines ingest.py and the old get_retriever().
    This will run ONCE when the Streamlit app starts.
    """
    # --- Configuration ---
    DOCUMENTS_DIR = "./documents"
    COLLECTION_NAME = "policy_docs"

    try:
        # 1. Use an IN-MEMORY client, not PersistentClient
        client = chromadb.Client()
        
        # 2. Get or create the collection
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"} 
        )

        # 3. --- Run Ingestion *if* collection is empty ---
        # This makes it run only once
        if collection.count() == 0:
            st.toast("No documents found in DB. Running one-time ingestion...")
            print("--- RUNNING INGESTION (ONE-TIME) ---")
            
            # Load documents
            loader = DirectoryLoader(
                DOCUMENTS_DIR,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True
            )
            docs = loader.load()
            if not docs:
                st.error("No documents found in './documents' folder. App cannot start.")
                return None

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(docs)

            # Embed and store
            embeddings_model = get_embedding_model() # Get cached model
            
            for i, chunk in enumerate(chunks):
                embedding = embeddings_model.embed_query(chunk.page_content)
                chunk_id = f"{os.path.basename(chunk.metadata['source'])}_{i}"
                
                # Add to collection (using lists as required)
                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding], # Must be a list
                    documents=[chunk.page_content],
                    metadatas=[chunk.metadata]
                )
                print(f"Processed chunk {i+1}/{len(chunks)}")
            
            st.toast("Ingestion complete! The bot is ready.")
            print("--- INGESTION COMPLETE ---")
        
        else:
            st.toast("Loaded existing vector store from cache.")
            print("--- LOADED EXISTING VECTOR STORE FROM CACHE ---")

        return collection # Return the populated collection
    
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {e}")
        return None


# --- Get our global resources ---
llm = get_llm()
embedding_model = get_embedding_model()
retriever = get_retriever() # This now runs the ingestion if needed

structured_llm = llm.with_structured_output(
    BotResponse,
    method="function_calling" 
)

# --- 5. Optimization: Cached Helper Function ---
@functools.lru_cache(maxsize=128)
def cached_llm_call(prompt_string: str) -> BotResponse:
    return structured_llm.invoke(prompt_string)

# --- 6. Langgraph Nodes ---

def retrieve_documents(state: GraphState) -> GraphState:
    print("--- 🔍 Retrieving Documents ---")
    metrics = state.get('metrics', {})
    start_time = time.time()
    
    question = state["question"]
    query_embedding = embedding_model.embed_query(question)
    
    results = retriever.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    
    state["documents"] = results["documents"][0]
    
    end_time = time.time()
    metrics["retrieval_time"] = end_time - start_time
    state["metrics"] = metrics
    print(f"--- Retrieved {len(state['documents'])} documents in {metrics['retrieval_time']:.2f}s ---")
    return state

def generate_answer(state: GraphState) -> GraphState:
    print("--- 🧠 Generating Answer ---")
    metrics = state.get('metrics', {})
    start_time = time.time()
    
    question = state["question"]
    documents = state["documents"]
    
    # --- FIX: Sort documents to ensure deterministic cache hits ---
    documents.sort()
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert HR & IT Policy assistant. "
         "Answer the user's question based *only* on the following context. "
         "You MUST provide the source document for your answer. "
         "If the context is not sufficient, say so."),
        ("human", 
         "Context:\n{context}\n\nQuestion:\n{question}")
    ])
    
    context = "\n\n---\n\n".join(documents)
    prompt_string = prompt_template.format(context=context, question=question)
    
    cache_info = cached_llm_call.cache_info()
    response_object = cached_llm_call(prompt_string)
    new_cache_info = cached_llm_call.cache_info()
    
    state["response"] = response_object
    end_time = time.time()
    metrics["llm_time"] = end_time - start_time
    metrics["cache_hit"] = "Yes" if new_cache_info.hits > cache_info.hits else "No"
    state["metrics"] = metrics
    print(f"--- Generated answer in {metrics['llm_time']:.2f}s (Cache: {metrics['cache_hit']}) ---")
    return state

# --- 7. Define and Compile the Graph ---
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app = workflow.compile(checkpointer=MemorySaver())

# --- 8. Streamlit UI (The Dashboard) ---
# (Your UI code is perfect, no changes needed here)

st.set_page_config(page_title="Policy Bot", layout="wide")
st.title("🤖 Company Policy Bot (RAG + Langgraph)")

st.sidebar.header("📊 Profiling & Metrics")
metrics_placeholder = st.sidebar.empty()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How can I help you with our company policies today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "source" in message:
            st.caption(f"Source: {message['source']}")

if prompt := st.chat_input("Ask about vacation, expenses, or IT..."):
    if not retriever:
        st.error("Retriever is not initialized. Please check logs.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        total_start_time = time.time()
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                inputs = {
                    "question": prompt, 
                    "metrics": {}
                }
                
                if "thread_id" not in st.session_state:
                    st.session_state.thread_id = str(uuid.uuid4())

                config = {
                    "configurable": {
                        "thread_id": st.session_state.thread_id,
                    },
                    "recursion_limit": 5
                }
                
                final_state = app.invoke(inputs, config)
                
                total_end_time = time.time()
                
                response = final_state["response"]
                metrics = final_state["metrics"]
                metrics["total_time"] = total_end_time - total_start_time
                
                st.markdown(response.answer)
                st.caption(f"Source: {response.source_document}")
                
                st.session_state.messages.append(
                    {"role": "assistant", 
                     "content": response.answer, 
                     "source": response.source_document}
                )
                
                metrics_placeholder.json(metrics)