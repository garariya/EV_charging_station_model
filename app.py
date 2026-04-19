import streamlit as st
import os
import time
import joblib
import pickle
import pandas as pd
import numpy as np

# Fix for Protobuf TypeError on Streamlit Cloud
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from dotenv import load_dotenv
load_dotenv()

# LangChain / LangGraph Imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_chroma import Chroma
from langchain_core.tools import tool

# Set page config
st.set_page_config(page_title="EV Charging AI Agent", page_icon="⚡", layout="wide")

# --- UI Header ---
st.title("⚡ AI EV Charging Station Predictor")
st.markdown("---")

# --- API Key Management (Automated) ---
# Priority: 1. Streamlit Secrets (Cloud) 2. Environment Variables (Local)
api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("🔑 **Google API Key not found.**")
    st.info("Please add `GOOGLE_API_KEY` to your Streamlit Secrets (Cloud) or `.env` file (Local).")
    
    with st.expander("🛠️ Debug Information (Technical Details)"):
        import langchain_google_genai
        import google.generativeai as genai
        st.write(f"- `langchain-google-genai` version: {langchain_google_genai.__version__}")
        st.write(f"- `google-generativeai` version: {genai.__version__}")
        st.write("Available Keys in Secrets:", list(st.secrets.keys()))
        st.write("Available API-related Keys in Env:", [k for k in os.environ.keys() if "API" in k or "KEY" in k])
    st.stop()

# --- Rate Limiting Logic ---
RATE_LIMIT_SECONDS = 5

def check_rate_limit():
    if "last_request_time" not in st.session_state:
        st.session_state.last_request_time = 0
    
    current_time = time.time()
    elapsed = current_time - st.session_state.last_request_time
    
    if elapsed < RATE_LIMIT_SECONDS:
        remaining = int(RATE_LIMIT_SECONDS - elapsed)
        return False, remaining
    return True, 0

# --- Load ML Models ---
@st.cache_resource(show_spinner="Loading machine learning models...")
def load_models():
    try:
        rf_model = joblib.load('rf_balanced_retrained_fe.joblib')
        xgb_model = joblib.load('xgb_cs_retrained_fe.joblib')
        scaler = joblib.load('scaler.joblib')
        
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        with open('optimal_threshold.pkl', 'rb') as f:
            threshold = pickle.load(f)
            
        return rf_model, xgb_model, scaler, le, float(threshold)
    except Exception as e:
        st.error(f"Error loading ML models: {e}")
        return None, None, None, None, None

rf_model, xgb_model, scaler, le, threshold = load_models()

# --- Setup ChromaDB RAG Vector Store ---
from langchain_core.embeddings import Embeddings

class MiniLMEmbeddingsWrapper(Embeddings):
    def __init__(self):
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
        self.ef = ONNXMiniLM_L6_V2()
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.ef(texts)
        
    def embed_query(self, text: str) -> list[float]:
        return self.ef([text])[0]

@st.cache_resource(show_spinner="Initializing RAG Knowledge Base...")
def setup_rag():
    try:
        embeddings_wrapper = MiniLMEmbeddingsWrapper()
        db = Chroma(embedding_function=embeddings_wrapper, persist_directory="./chroma_db")
        
        # Seed facts if DB is empty
        if len(db.get()["ids"]) == 0:
            docs = [
                "Fast DC charging stations (Level 3) provide 50kW to 350kW, providing 80% charge in under 30 minutes.",
                "AC Charging (Level 2) typically provides 7kW to 22kW and is common for home and workplace charging.",
                "The CCS2 standard is the most common for Fast DC charging in Europe.",
                "In North America, Tesla's NACS and CCS1 are the dominant Fast DC standards.",
                "High-density urban areas and highway corridors are primary locations for Fast DC infrastructure."
            ]
            db.add_texts(
                texts=docs,
                metadatas=[{"source": "EV Facts"}] * len(docs),
                ids=[f"fact_{i}" for i in range(len(docs))]
            )
        return db
    except Exception as e:
        st.error(f"Failed to initialize RAG database: {e}")
        return None

db = setup_rag()

# --- Agent Tools ---
@tool
def search_ev_knowledge(query: str) -> str:
    """Search for factual information about EV charging infrastructure, standards, and speeds."""
    if db is None: return "Database offline."
    results = db.similarity_search(query, k=2)
    return "\n\n".join([doc.page_content for doc in results])

@tool
def predict_fast_dc(country_code: str, latitude: float, longitude: float, ports: int) -> str:
    """Predicts if a location likely has a Fast DC charging station based on its coordinates and port count."""
    if rf_model is None: return "ML Models not loaded."
    try:
        input_data = pd.DataFrame([[country_code, latitude, longitude, ports]], 
                                  columns=['country_code', 'latitude', 'longitude', 'ports'])
        
        # Preprocessing
        try:
            input_data['country_code'] = le.transform(input_data['country_code'])
        except:
            input_data['country_code'] = -1
            
        input_data[['latitude', 'longitude', 'ports']] = scaler.transform(input_data[['latitude', 'longitude', 'ports']])
        
        # Feature Engineering
        input_data['latitude_x_longitude'] = input_data['latitude'] * input_data['longitude']
        input_data['ports_x_latitude'] = input_data['ports'] * input_data['latitude']
        input_data['ports_x_longitude'] = input_data['ports'] * input_data['longitude']
        input_data['latitude_squared'] = input_data['latitude'] ** 2
        input_data['longitude_squared'] = input_data['longitude'] ** 2
        input_data['ports_squared'] = input_data['ports'] ** 2
        
        # Model Inference
        rf_prob = rf_model.predict_proba(input_data)[:, 1]
        xgb_prob = xgb_model.predict_proba(input_data)[:, 1]
        ensemble_prob = (rf_prob[0] + xgb_prob[0]) / 2
        
        result = "POSITIVE" if ensemble_prob >= threshold else "NEGATIVE"
        return f"Prediction: {result}. (Probability: {ensemble_prob:.2f})"
    except Exception as e:
        return f"Prediction Error: {str(e)}"

# --- Agent Initialization ---
try:
    llm = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-flash", temperature=0)
    agent_executor = create_react_agent(llm, [search_ev_knowledge, predict_fast_dc])
except Exception as e:
    st.error(f"Failed to initialize AI Agent: {e}")
    st.stop()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful EV charging assistant. Use tools to find facts or predict station types.")
    ]

# Display history
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage): continue
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content if isinstance(msg.content, str) else str(msg.content))

# Input
if prompt := st.chat_input("Ask about EV charging or request a prediction..."):
    # Check rate limit
    allowed, wait_time = check_rate_limit()
    if not allowed:
        st.warning(f"⏳ Please wait {wait_time} seconds before your next question.")
    else:
        st.session_state.last_request_time = time.time()
        
        # Add human message
        user_msg = HumanMessage(content=prompt)
        st.session_state.messages.append(user_msg)
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Agent Reasoning
        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                try:
                    response = agent_executor.invoke({"messages": st.session_state.messages})
                    # Add only new messages to session state
                    new_msgs = response["messages"][len(st.session_state.messages):]
                    for m in new_msgs:
                        st.session_state.messages.append(m)
                    
                    # Display final answer
                    final_text = response["messages"][-1].content
                    st.markdown(final_text)
                except Exception as e:
                    st.error(f"Agent Error: {str(e)}")

# Sidebar
with st.sidebar:
    st.header("App Management")
    if st.button("Clear Conversation"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()
    st.info("Rate limit: 1 request per 5 seconds.")
