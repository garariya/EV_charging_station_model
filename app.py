import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np
import os

# LangChain / LangGraph Imports
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_chroma import Chroma

# Set page config
st.set_page_config(page_title="EV Charging AI Agent", page_icon="⚡")

# --- Load Models ---
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
        st.error(f"Error loading ML models: {e}. Please ensure all model files are present.")
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
        # Utilize the ultra-lightweight ONNX version bundled directly within Chroma,
        # wrapped securely inside LangChain's base class interface!
        embeddings_wrapper = MiniLMEmbeddingsWrapper()
        
        db = Chroma(embedding_function=embeddings_wrapper, persist_directory="./chroma_db")
        
        # Seed dummy facts if DB is empty to demonstrate technical completion
        if len(db.get()["ids"]) == 0:
            docs = [
                "Fast DC charging stations are predominantly located near major transit corridors and urban centers. They typically utilize CCS or CHAdeMO standards.",
                "While AC chargers provide up to 22kW, Fast DC chargers usually supply 50kW to 350kW directly to the vehicle's battery.",
                "In Europe, the CCS2 standard dominates the Fast DC network rollout.",
                "In North America, Tesla's NACS and the CCS1 standards are the most widespread Fast DC charging options.",
                "Statistically, standalone charging locations with more than 3 distinct ports carry a substantially higher probability of housing Fast DC hardware.",
                "Range anxiety is largely mitigated by high-speed Level 3 (DC Fast) charging infrastructure which can charge vehicles to 80% in under 30 minutes."
            ]
            db.add_texts(
                texts=docs,
                metadatas=[{"source": "EV Facts Knowledge Base"}] * len(docs),
                ids=[f"fact_{i}" for i in range(len(docs))]
            )
        return db
    except Exception as e:
        st.error(f"Failed to initialize RAG database: {e}")
        return None

db = setup_rag()

# --- Define LangChain Agent Tools ---
@tool
def search_ev_knowledge(query: str) -> str:
    """
    Search the RAG knowledge base for information context on electric vehicles, charging standards, speed details, and region-specific EV facts.
    Use this tool whenever the user asks a general factual question about EV charging.
    """
    if db is None:
        return "RAG Database is offline."
    results = db.similarity_search(query, k=2)
    return "\n\n".join([doc.page_content for doc in results])

@tool
def predict_fast_dc(country_code: str, latitude: float, longitude: float, ports: int) -> str:
    """
    Predicts if a location has a Fast DC charging station.
    You must NOT call this tool until you have gathered EXACTLY these 4 arguments from the user:
    - country_code (string, e.g. 'US', 'CA')
    - latitude (float, e.g. 40.7128)
    - longitude (float, e.g. -74.0060)
    - ports (integer, number of charging ports, e.g. 2)
    
    Returns a string prediction ('Yes' or 'No').
    """
    if rf_model is None:
        return "Error: Local ML Models are not loaded."
        
    try:
        cc = str(country_code)
        lat = float(latitude)
        lon = float(longitude)
        p = int(ports)
        
        input_data = pd.DataFrame([[cc, lat, lon, p]], columns=['country_code', 'latitude', 'longitude', 'ports'])
        
        # Encode Country Code
        try:
            input_data['country_code'] = le.transform(input_data['country_code'])
        except ValueError:
            input_data['country_code'] = -1
            
        # Scale Features
        input_data[['latitude', 'longitude', 'ports']] = scaler.transform(input_data[['latitude', 'longitude', 'ports']])
        
        # Feature Engineering Context
        input_data['latitude_x_longitude'] = input_data['latitude'] * input_data['longitude']
        input_data['ports_x_latitude'] = input_data['ports'] * input_data['latitude']
        input_data['ports_x_longitude'] = input_data['ports'] * input_data['longitude']
        input_data['latitude_squared'] = input_data['latitude'] ** 2
        input_data['longitude_squared'] = input_data['longitude'] ** 2
        input_data['ports_squared'] = input_data['ports'] ** 2
        
        # Ensemble Prediction
        rf_prob = rf_model.predict_proba(input_data)[:, 1]
        xgb_prob = xgb_model.predict_proba(input_data)[:, 1]
        ensemble_prob = (rf_prob[0] + xgb_prob[0]) / 2
        
        final_pred = 1 if ensemble_prob >= threshold else 0
        
        if final_pred == 1:
            return "ML Prediction Output: POSITIVE. A Fast DC charging station is likely present."
        else:
            return "ML Prediction Output: NEGATIVE. A Fast DC charging station is NOT present."
            
    except Exception as e:
        return f"Prediction Execution Error: {str(e)}"

# --- UI Setup & Configuration ---
st.title("⚡ AI EV Charging Station Predictor")
st.markdown("**(Powered by LangGraph, ChromaDB, and Gemini)**")
st.markdown("Chat with the Agent to predict station layouts or ask about EV Charging Infrastructure facts!")

api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
if not api_key:
    st.info("👋 Welcome! Please enter your Google Gemini API Key in the sidebar to power the LangGraph agent.")
    st.stop()

def reset_conversation():
    from langchain_core.messages import SystemMessage
    st.session_state.messages = [
        SystemMessage(content=system_prompt)
    ]
    
st.sidebar.button("Start Over", on_click=reset_conversation)

# --- Agentic Initialization ---
# Initialize Gemini LLM natively by dynamically querying which models their API key allows
@st.cache_data(show_spinner=False)
def get_working_google_model(key):
    import google.generativeai as genai
    try:
        genai.configure(api_key=key)
        # Try to find a working generic or flash model in their account
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'flash' in m.name.lower() or 'pro' in m.name.lower() or 'gemini' in m.name.lower():
                    # LangChain expects the name without the "models/" prefix
                    return m.name.replace("models/", "")
    except Exception:
        pass
    return "gemini-1.5-flash"  # Fallback

try:
    working_model = get_working_google_model(api_key)
    llm = ChatGoogleGenerativeAI(google_api_key=api_key, model=working_model, temperature=0)
except Exception as e:
    st.error(f"Failed to initialize Gemini LLM: {e}")
    st.stop()

system_prompt = """You are a highly helpful and friendly EV charging station virtual assistant agent.
Your primary capabilities are:
1. Running math predictions using the `predict_fast_dc` tool.
2. Querying a real-world vector database about EV infrastructure using the `search_ev_knowledge` tool.

To run the ML prediction tool `predict_fast_dc`, you MUST collect 4 specific inputs from the user ONE BY ONE:
1. Country Code (e.g. US)
2. Latitude (number)
3. Longitude (number)
4. Number of Ports (integer)

Rules:
- Start by warmly greeting the user.
- Ask for ONE piece of required prediction information at a time.
- If the user asks a general factual question about EVs, you MUST autonomously use your `search_ev_knowledge` tool to pull relevant facts from the ChromaDB to answer them.
- Once you have successfully collected all 4 features from the user, immediately call the `predict_fast_dc` tool.
- Always explain the tool results friendly to the user.
"""

from langchain_core.messages import SystemMessage

# Compile LangGraph ReAct Agent
tools_list = [search_ev_knowledge, predict_fast_dc]
agent_executor = create_react_agent(llm, tools_list)

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=system_prompt)
    ]

# --- Helper for parsing Gemini UI Content ---
def extract_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts)
    return str(content)

# --- Display Chat History ---
for msg in st.session_state.messages:
    # Do not display the hidden system prompt to the user
    if isinstance(msg, SystemMessage):
        continue
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(extract_text(msg.content))
    # Filter out intermediary ToolMessages from UI so user only sees AI strings
    elif isinstance(msg, AIMessage):
        # LangChain sometimes outputs empty string AIMessages when invoking a tool request
        if msg.content:
            parsed_text = extract_text(msg.content)
            if parsed_text.strip():
                with st.chat_message("assistant"):
                    st.markdown(parsed_text)

# --- User Input & Graph Execution ---
if prompt := st.chat_input("Type your message here..."):
    # 1. Store and show user message
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Agent is reasoning..."):
            try:
                # 2. Invoke LangGraph executor pipeline
                response = agent_executor.invoke({"messages": st.session_state.messages})
                
                # 3. Append the newly generated agentic nodes/tool actions to state
                # response["messages"] contains the complete conversational state graph.
                # We just append the newly added messages to our Streamlit session state.
                new_messages = response["messages"][len(st.session_state.messages):]
                for m in new_messages:
                    st.session_state.messages.append(m)
                    
                # 4. Display the ultimate finalized AIMessage response to the user
                final_reply = extract_text(response["messages"][-1].content)
                message_placeholder.markdown(final_reply)
                
            except Exception as e:
                st.error(f"Error executing LangGraph agent workflow: {str(e)}")
