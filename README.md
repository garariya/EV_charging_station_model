# ⚡ EV Charging Station AI Agent
### From Predictive Analytics to Agentic Intelligence

## 🚀 Project Overview
This project represents the evolution of an EV Charging Station Classifier into a sophisticated **Agentic AI System**. Originally designed as a machine learning pipeline to predict 'Fast DC Charger' status, it now features a fully autonomous conversational agent capable of reasoning, researching domain-specific facts, and executing complex predictive workflows.

The system uses a **ReAct (Reasoning and Acting)** pattern to interact with users, gathering necessary data points sequentially and providing insights backed by both statistical models and a Retrieval-Augmented Generation (RAG) knowledge base.

---

## 🌟 Key Features
- **Agentic Orchestration**: Built with **LangGraph**, the agent maintains conversation state and intelligently decides when to use tools.
- **RAG Knowledge Base**: Uses **ChromaDB** to store and retrieve domain-specific EV infrastructure facts, allowing the agent to answer technical questions about charging standards and corridors.
- **Ensemble ML Prediction**: Combines **RandomForest** and **XGBoost** models with advanced feature engineering to classify stations with high precision.
- **Adaptive Reasoning**: The agent gathers the 4 required prediction inputs (Country Code, Latitude, Longitude, Ports) through natural conversation before triggering the model.
- **Dynamic UI**: A modern **Streamlit** interface featuring a full chat experience.

---

## 🛠️ Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Agent Framework** | LangGraph, LangChain |
| **Core LLM** | Google Gemini (Gemini 1.5 Flash/Pro) |
| **Vector Database** | ChromaDB (RAG) |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Data Manipulation** | Pandas, Numpy |
| **Embeddings** | ONNX MiniLM-L6-V2 (Lightweight Local) |
| **Backend & UI** | Streamlit |
| **Model Persistence** | Joblib, Pickle |

---

## 📈 System Evolution

### Phase 1: Core ML Pipeline (Milestone 1)
- Developed baseline models using RandomForest and XGBoost.
- Implemented class imbalance handling (`class_weight='balanced'`) to address the rarity of Fast DC Chargers in the dataset.
- Established primary preprocessing and evaluation metrics (ROC AUC, F1-score).

### Phase 2: Optimized Solution (Milestone 2)
- **Advanced Feature Engineering**: Interaction terms (Lat x Lon) and squared terms (Ports²) to capture non-linear relationships.
- **Hyperparameter Tuning**: Optimized models via `RandomizedSearchCV`.
- **Ensemble Modeling**: Created a weighted probability-averaging ensemble for superior recall.

### Phase 3: Agentic Integration (Current)
- Wrapped the ML pipeline into a LangChain Tool.
- Integrated a RAG layer for "Knowledge-on-Demand".
- Implemented the LangGraph state machine to handle multi-turn information gathering.

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/garariya/EV_charging_station_model.git
cd EV_charging_station_model
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Model & Data Setup
Ensure the following artifacts are in the root directory:
- `rf_balanced_retrained_fe.joblib`
- `xgb_cs_retrained_fe.joblib`
- `scaler.joblib`
- `label_encoder.pkl`
- `optimal_threshold.pkl`

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 🤖 How to Interact with the Agent
1. **Provide a Gemini API Key**: In the sidebar, enter your Google Gemini API key to activate the agent.
2. **Chat Naturally**: You can ask questions like *"What is the difference between AC and DC charging?"* and the agent will use the RAG tool.
3. **Run a Prediction**: If you ask *"Can you predict a station for me?"*, the agent will begin a guided workflow, asking you for the location and port details one by one.
4. **Final Verdict**: Once all data is collected, the agent will run the ensemble model and explain the result.

---

## 👥 Contributors
Developed as part of a team effort focusing on Intelligent Intervention in EV infrastructure.

**Maintained by**: [garariya](https://github.com/garariya)
