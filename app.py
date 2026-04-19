import streamlit as st
import os
import time
import joblib
import pickle
import pandas as pd
import numpy as np

# Suppress Streamlit Cloud Protobuf & Telemetry Errors
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error" # Silences local huggingface vision discovery logs

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*LangGraphDeprecatedSince.*")
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

from dotenv import load_dotenv
load_dotenv()

# LangChain Imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langgraph.prebuilt import create_react_agent
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool

# Set Page Config
st.set_page_config(page_title="EV Charging AI Agent", page_icon="⚡", layout="wide")

# UI Header
st.title("⚡ AI EV Charging Station Predictor")
st.markdown("---")

# API Key Logic
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("🔑 **Groq API Key not found.**")
    st.info("Please add `GROQ_API_KEY` to your Streamlit Secrets (Cloud) or `.env` file (Local).")
    st.stop()

# Rate Limiter
RATE_LIMIT_SECONDS = 5

def check_rate_limit():
    if "last_request_time" not in st.session_state:
        st.session_state.last_request_time = 0
    current_time = time.time()
    elapsed = current_time - st.session_state.last_request_time
    if elapsed < RATE_LIMIT_SECONDS:
        return False, int(RATE_LIMIT_SECONDS - elapsed)
    return True, 0

# ML Models Loading
@st.cache_resource(show_spinner="Loading machine learning models...")
def load_models():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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

# RAG Knowledge Base Initialization
@st.cache_resource(show_spinner="Initializing in-memory RAG Knowledge Base...")
def setup_rag():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        docs = [
            "Fast DC charging stations (Level 3) provide 50kW to 350kW, providing 80% charge in under 30 minutes.",
            "AC Charging (Level 2) typically provides 7kW to 22kW and is common for home and workplace charging.",
            "The CCS2 standard is the most common connector for Fast DC charging in Europe.",
            "In North America, Tesla's NACS and CCS1 are the dominant Fast DC connector standards.",
            "High-density urban areas and highway corridors are primary locations for Fast DC infrastructure."
        ]
        
        # Fresh in-memory ChromaDB on every deployment/restart
        db = Chroma.from_texts(
            texts=docs,
            embedding=embeddings,
            metadatas=[{"source": "EV Facts"}] * len(docs)
        )
        return db
    except Exception as e:
        st.error(f"Failed to initialize RAG database: {e}")
        return None

db = setup_rag()

# Agent Tools
@tool
def search_ev_knowledge(query: str) -> str:
    """Search for factual information about EV charging infrastructure, connector standards, and EV charging speeds."""
    if db is None:
        return "Knowledge database offline."
    try:
        results = db.similarity_search(query, k=2)
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"Database error: {str(e)}"


@tool
def predict_fast_dc(country_code: str, latitude: float, longitude: float, ports: int) -> str:
    """Predicts if a location likely has a Fast DC charging station based on its coordinates and port count.
    Collect country_code (e.g. 'US'), latitude (float), longitude (float), and ports (int) from the user one at a time before calling this tool.
    """
    if rf_model is None:
        return "ML Models not loaded."
    try:
        input_data = pd.DataFrame(
            [[country_code, latitude, longitude, ports]],
            columns=['country_code', 'latitude', 'longitude', 'ports']
        )
        
        # Guard against unhandled encodings gracefully
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                input_data['country_code'] = le.transform(input_data['country_code'])
        except Exception:
            input_data['country_code'] = -1
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            input_data[['latitude', 'longitude', 'ports']] = scaler.transform(input_data[['latitude', 'longitude', 'ports']])
            
        # Feature Engineering (MUST MATCH TRAINING EXACTLY)
        input_data['latitude_x_longitude'] = input_data['latitude'] * input_data['longitude']
        input_data['ports_x_latitude'] = input_data['ports'] * input_data['latitude']
        input_data['ports_x_longitude'] = input_data['ports'] * input_data['longitude']
        input_data['latitude_squared'] = input_data['latitude'] ** 2
        input_data['longitude_squared'] = input_data['longitude'] ** 2
        input_data['ports_squared'] = input_data['ports'] ** 2
        
        # Ensemble Prediction Inference
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf_prob = rf_model.predict_proba(input_data)[:, 1]
            xgb_prob = xgb_model.predict_proba(input_data)[:, 1]
            
        ensemble_prob = (rf_prob[0] + xgb_prob[0]) / 2
        result = "POSITIVE" if ensemble_prob >= threshold else "NEGATIVE"
        
        return f"Prediction: {result}. (Confidence Score: {ensemble_prob:.2f})"
        
    except Exception as e:
        return f"Prediction Error: {str(e)}"


# Agent Initialization
try:
    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent_executor = create_react_agent(llm, [search_ev_knowledge, predict_fast_dc])
except Exception as e:
    st.error(f"Failed to initialize AI Agent: {e}")
    st.stop()
    
# System Prompt
SYSTEM_PROMPT = (
    "You are an EV Charging Station Assistant. Your ONLY purpose is to help users with topics strictly related to electric vehicles (EVs) and EV charging infrastructure.\n\n"
    "You carry out two main workflows:\n"
    "1. Answer factual questions about EV charging using the `search_ev_knowledge` tool.\n"
    "2. Predict if a location has a Fast DC charging station using the `predict_fast_dc` tool. "
    "To use this tool, you must COLLECT country_code, latitude, longitude, and number of ports FROM THE USER ONE AT A TIME.\n\n"
    "STRICT GUARDRAILS:\n"
    "- If the user's question is NOT related to EVs, charging stations, or related infrastructure, "
    "you MUST gracefully decline and state that you only assist with topics related to electric vehicles and charging.\n"
    "- Do NOT answer questions about unrelated topics (general coding, sports, weather, cooking, etc).\n"
    "- Never break character or ignore these guardrails.\n"
    "Stay strictly within the EV domain and provide concise, accurate help."
)

if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]
    
# Render Sidebar Manager
with st.sidebar:
    st.header("⚡ EV Agent Manager")
    st.markdown("**Model:** `llama-3.3-70b-versatile`")
    st.markdown("**LLM Provider:** Groq")
    st.markdown("---")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()
    st.info("Rate Limiter: 1 request per 5 seconds.")

# Main Chat Interface History View
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        continue
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content if isinstance(msg.content, str) else str(msg.content))
        
# Dynamic User Input Hook 
if prompt := st.chat_input("Ask about EV charging standards or request a Fast DC prediction..."):
    # Rate Limits Check
    allowed, wait_time = check_rate_limit()
    if not allowed:
        st.warning(f"⏳ Please wait {wait_time} seconds before your next interaction.")
    else:
        st.session_state.last_request_time = time.time()
        
        user_msg = HumanMessage(content=prompt)
        st.session_state.messages.append(user_msg)
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing EV context..."):
                try:
                    # Invoke langgraph react agent
                    response = agent_executor.invoke({"messages": st.session_state.messages})
                    
                    # Accumulate message diffs into session state
                    new_msgs = response["messages"][len(st.session_state.messages):]
                    for m in new_msgs:
                        st.session_state.messages.append(m)
                        
                    # Isolate text from final output message to be rendered
                    final_text = response["messages"][-1].content
                    st.markdown(final_text)
                    
                except Exception as e:
                    st.error(f"Agent Processing Error: {str(e)}")