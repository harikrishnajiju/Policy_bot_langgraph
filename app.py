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

# --- 1. Load Environment Variables ---
# Load the .env file (which contains OPENAI_API_KEY)
load_dotenv()

# --- 2. Define Pydantic Model for LLM Output ---
# This tells the LLM *exactly* how to format its response.
class BotResponse(BaseModel):
    """The structured response from the Policy Bot."""
    answer: str = Field(description="The final answer to the user's question.")
    source_document: str = Field(
        description="The 'source' filename from the metadata (e.g., 'hr_policy.txt')."
    )

# --- 3. Define Langgraph State ---
# This is the "memory" of our graph. It's what passes from one node to the next.
class GraphState(TypedDict):
    question: str
    documents: List[str]
    response: Optional[BotResponse]
    metrics: Dict[str, Any] # For our profiling

# --- 4. Global Setup (Models & DB) ---
# We initialize these once and cache them for all users/sessions.

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
    """Cached function to get the ChromaDB retriever."""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name="policy_docs")
        return collection
    except Exception as e:
        st.error(f"Error connecting to ChromaDB: {e}")
        return None

# Get our global resources
llm = get_llm()
embedding_model = get_embedding_model()
retriever = get_retriever()

structured_llm = llm.with_structured_output(
    BotResponse,
    # This fixes the UserWarning for gpt-3.5-turbo
    method="function_calling" 
)

# --- 5. Optimization: Cached Helper Function ---
# This is our "Optimization" step. We cache the *actual LLM call*.
@functools.lru_cache(maxsize=128)
def cached_llm_call(prompt_string: str) -> BotResponse:
    """
    A cached function to call the LLM.
    It takes a hashable string as input.
    It uses the *global* 'structured_llm' runnable.
    """
    # We use the globally defined structured_llm
    return structured_llm.invoke(prompt_string)

# --- 6. Langgraph Nodes ---
# These are the "steps" in our application's logic.

def retrieve_documents(state: GraphState) -> GraphState:
    """Node: Retrieves documents from ChromaDB."""
    print("--- 🔍 Retrieving Documents ---")
    metrics = state.get('metrics', {})
    start_time = time.time()
    
    question = state["question"]
    
    # 1. Embed the user's question
    query_embedding = embedding_model.embed_query(question)
    
    # 2. Query ChromaDB
    results = retriever.query(
        query_embeddings=[query_embedding],
        n_results=2  # Get the top 2 most relevant chunks
    )
    
    # 3. Store in state
    state["documents"] = results["documents"][0]
    
    end_time = time.time()
    metrics["retrieval_time"] = end_time - start_time
    state["metrics"] = metrics
    
    print(f"--- Retrieved {len(state['documents'])} documents in {metrics['retrieval_time']:.2f}s ---")
    return state

def generate_answer(state: GraphState) -> GraphState:
    """Node: Generates a structured answer using the LLM."""
    print("--- 🧠 Generating Answer ---")
    metrics = state.get('metrics', {})
    start_time = time.time()
    
    question = state["question"]
    documents = state["documents"]
    
    # Create the prompt for the LLM
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
    
    # --- FIX: The formatted prompt is *already* a string. This is our cache key. ---
    # We just rename the variable 'prompt_object' to 'prompt_string'.
    prompt_string = prompt_template.format(context=context, question=question)
    
    # We no longer define structured_llm here, we use the global one.
    
    # Check cache status *before* calling
    cache_info = cached_llm_call.cache_info()
    
    # --- This is our Optimization ---
    # Call the *cached* function with only the string
    response_object = cached_llm_call(prompt_string)
    
    # Check cache status *after* calling
    new_cache_info = cached_llm_call.cache_info()
    
    state["response"] = response_object
    
    end_time = time.time()
    metrics["llm_time"] = end_time - start_time
    metrics["cache_hit"] = "Yes" if new_cache_info.hits > cache_info.hits else "No"
    state["metrics"] = metrics
    
    print(f"--- Generated answer in {metrics['llm_time']:.2f}s (Cache: {metrics['cache_hit']}) ---")
    return state

# --- 7. Define and Compile the Graph ---

# This is where we wire our nodes together.
workflow = StateGraph(GraphState)

# Add the nodes
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)

# Define the flow
workflow.add_edge(START, "retrieve") # Start by retrieving
workflow.add_edge("retrieve", "generate") # Then generate
workflow.add_edge("generate", END) # Then end

# Compile the graph into a runnable object
app = workflow.compile(checkpointer=MemorySaver())

# --- 8. Streamlit UI (The Dashboard) ---

st.set_page_config(page_title="Policy Bot", layout="wide")
st.title("🤖 Company Policy Bot (RAG + Langgraph)")

# --- Dashboard Sidebar for Metrics ---
st.sidebar.header("📊 Profiling & Metrics")
metrics_placeholder = st.sidebar.empty()

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How can I help you with our company policies today?"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If it's a bot response, show the source
        if message["role"] == "assistant" and "source" in message:
            st.caption(f"Source: {message['source']}")

# Handle user input
if prompt := st.chat_input("Ask about vacation, expenses, or IT..."):
    if not retriever:
        st.error("Retriever is not initialized. Please check DB connection.")
    else:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Start timer for total response
        total_start_time = time.time()
        
        # Show a "thinking" spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Run the graph
                inputs = {
                    "question": prompt, 
                    "metrics": {} # Pass in a fresh metrics dict
                }
                
                # We use 'stream' to get updates, but 'invoke' is simpler
                # For this POC, 'invoke' shows the final state clearly

                if "thread_id" not in st.session_state:
                    st.session_state.thread_id = str(uuid.uuid4())

                # This is the config dict langgraph needs
                config = {
                    "configurable": {
                        # Pass the thread_id so the checkpointer knows which conversation
                        "thread_id": st.session_state.thread_id,
                    },
                    "recursion_limit": 5
                }


                final_state = app.invoke(
                    inputs,
                    config
                )
                
                total_end_time = time.time()
                
                # Get results from the final state
                response = final_state["response"]
                metrics = final_state["metrics"]
                metrics["total_time"] = total_end_time - total_start_time
                
                # Display the bot's answer and source
                st.markdown(response.answer)
                st.caption(f"Source: {response.source_document}")
                
                # Add bot response to session state
                st.session_state.messages.append(
                    {"role": "assistant", 
                     "content": response.answer, 
                     "source": response.source_document}
                )
                
                # --- Update Metrics Dashboard ---
                metrics_placeholder.json(metrics)