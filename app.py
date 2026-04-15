import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np
import google.generativeai as genai
import json
import re

# Set page config
st.set_page_config(page_title="EV Charging AI Agent", page_icon="⚡")

# --- Load Models ---
@st.cache_resource(show_spinner="Loading models into memory...")
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
        st.error(f"Error loading models: {e}. Please ensure all model files are present.")
        return None, None, None, None, None

rf_model, xgb_model, scaler, le, threshold = load_models()

# --- UI Setup ---
st.title("⚡ AI EV Charging Station Predictor")
st.markdown("Chat with our AI agent to check if a Fast DC charging station is likely present at a given location!")

api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.info("👋 Welcome! Please enter your Gemini API Key in the sidebar to start.")
    st.stop()

# System Prompt for Gemini
system_prompt = """You are a highly helpful and friendly EV charging station virtual assistant. 
Your task is to predict whether a Fast DC charging station is present based on user input.
You need to collect 4 specific inputs from the user ONE BY ONE:
1. Country Code (e.g., US, CA, NL)
2. Latitude (number, e.g., 40.7128)
3. Longitude (number, e.g., -74.0060)
4. Number of Ports (positive integer, e.g., 2)

Rules:
- Start by warmly greeting the user and offering to help them predict if a location has a Fast DC charging station.
- Ask for ONE piece of information at a time. Do not ask for all 4 features at once!
- Validate each input (e.g., latitude should be between -90 and 90, longitude between -180 and 180, ports should be > 0). If wrong, ask them to correct it.
- Once you have successfully collected and validated all 4 features, and ONLY THEN, output ONLY a JSON block like the following (and absolutely nothing else in that message):
```json
{
  "ready": true,
  "country_code": "US",
  "latitude": 40.7128,
  "longitude": -74.0060,
  "ports": 2
}
```
- If I (the system) provide you with a prediction result in brackets like "[SYSTEM: Prediction Result is: Yes/No...]", please explain the prediction to the user in natural language and ask if they want to check another location.
"""

def reset_conversation():
    st.session_state.messages = []
    
st.sidebar.button("Start Over", on_click=reset_conversation)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to parse JSON when all inputs are collected
def extract_json(text):
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            return None
    
    # Fallback to direct json parsing if no code blocks are present
    try:
        return json.loads(text)
    except:
        return None

# --- Display Chat ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display UI indicator if this message has a prediction stored
        if msg.get("prediction_ui"):
            if msg["prediction"] == 1:
                st.success("✅ Yes, a Fast DC charging station is likely present here.")
            else:
                st.error("❌ No Fast DC charging station found here.")

# --- User Input & Chat Flow ---
if prompt := st.chat_input("Type your message here..."):
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 2. Prepare history for Gemini (remember last 5 turns = 10 messages)
    history = []
    recent_msgs = st.session_state.messages[-10:]
    for m in recent_msgs:
        # Convert internal roles to ones generative AI expects
        role = "user" if m["role"] == "user" else "model"
        history.append({"role": role, "parts": [{"text": m["content"]}]})
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Auto-detect best model from user's quota
            model_name = "gemini-1.5-flash"
            try:
                available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                flash_models = [m for m in available if '1.5-flash' in m]
                pro_models = [m for m in available if '1.5' in m]
                
                if flash_models:
                    model_name = flash_models[0]
                elif pro_models:
                    model_name = pro_models[0]
                elif available:
                    model_name = available[0]
            except Exception:
                pass
                
            if "1.5" in model_name:
                model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
                response = model.generate_content(history)
            else:
                model = genai.GenerativeModel(model_name)
                # Inject system prompt into latest user message if 1.5 is unavailable
                if history:
                    history[-1]["parts"][0]["text"] = f"[SYSTEM INSTRUCTION:\n{system_prompt}]\n\nUser Input: {history[-1]['parts'][0]['text']}"
                response = model.generate_content(history)
            
            reply = response.text
            
            parsed_data = extract_json(reply)
            
            # 3. Check if JSON indicates readiness to predict
            if parsed_data and parsed_data.get("ready"):
                if rf_model is None:
                    err_reply = "I'm ready to predict, but the ML models are not loaded correctly. Please check the model files."
                    message_placeholder.markdown(err_reply)
                    st.session_state.messages.append({"role": "assistant", "content": err_reply})
                else:
                    # Run ML Prediction
                    cc = str(parsed_data.get("country_code", "US"))
                    lat = float(parsed_data.get("latitude", 0.0))
                    lon = float(parsed_data.get("longitude", 0.0))
                    ports = int(parsed_data.get("ports", 1))
                    
                    input_data = pd.DataFrame([[cc, lat, lon, ports]], columns=['country_code', 'latitude', 'longitude', 'ports'])
                    
                    # Encode Country Code
                    try:
                        input_data['country_code'] = le.transform(input_data['country_code'])
                    except ValueError:
                        input_data['country_code'] = -1
                        
                    # Scale Features
                    input_data[['latitude', 'longitude', 'ports']] = scaler.transform(input_data[['latitude', 'longitude', 'ports']])
                    
                    # Feature Engineering
                    input_data['latitude_x_longitude'] = input_data['latitude'] * input_data['longitude']
                    input_data['ports_x_latitude'] = input_data['ports'] * input_data['latitude']
                    input_data['ports_x_longitude'] = input_data['ports'] * input_data['longitude']
                    input_data['latitude_squared'] = input_data['latitude'] ** 2
                    input_data['longitude_squared'] = input_data['longitude'] ** 2
                    input_data['ports_squared'] = input_data['ports'] ** 2
                    
                    # Predict Ensemble
                    rf_prob = rf_model.predict_proba(input_data)[:, 1]
                    xgb_prob = xgb_model.predict_proba(input_data)[:, 1]
                    ensemble_prob = (rf_prob[0] + xgb_prob[0]) / 2
                    
                    final_pred = 1 if ensemble_prob >= threshold else 0
                    
                    # 4. Show Prediction Indicators
                    if final_pred == 1:
                        st.success("✅ Yes, a Fast DC charging station is likely present here.")
                    else:
                        st.error("❌ No Fast DC charging station found here.")
                        
                    # 5. Inject system message and ask Gemini to explain
                    pred_str = "Yes, Fast DC charging station is present" if final_pred == 1 else "No, Fast DC charging station is NOT present"
                    system_inject = f"[SYSTEM: Prediction Result is: {pred_str}. Briefly explain this friendly to the user, and ask if they want to check another location.]"
                    
                    # Add AI JSON & system instruction to history to maintain conversational flow context
                    history.append({"role": "model", "parts": [{"text": reply}]})
                    history.append({"role": "user", "parts": [{"text": system_inject}]})
                    
                    explain_response = model.generate_content(history)
                    final_reply = explain_response.text
                    
                    message_placeholder.markdown(final_reply)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": final_reply,
                        "prediction_ui": True,
                        "prediction": final_pred
                    })
            else:
                # Normal conversation turn
                message_placeholder.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                
        except Exception as e:
            st.error(f"Error communicating with Gemini: {str(e)}")
