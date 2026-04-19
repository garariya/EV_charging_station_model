# ⚡ EV Charging Station AI Agent
### From Predictive Analytics to Agentic Intelligence

## 🚀 Project Overview
This project represents the evolution of an EV Charging Station Classifier into a sophisticated **Agentic AI System**. Originally designed as a machine learning pipeline to predict 'Fast DC Charger' status, it now features a fully autonomous conversational agent capable of reasoning, researching domain-specific facts, and executing complex predictive workflows.

The system uses a **ReAct (Reasoning and Acting)** pattern to interact with users, gathering necessary data points sequentially and providing insights backed by both statistical models and a Retrieval-Augmented Generation (RAG) knowledge base.

---

## 🌟 Key Features
- **Agentic Orchestration**: Built with **LangGraph**, the agent maintains conversation state and intelligently decides when to use tools.
- **RAG Knowledge Base**: Uses **ChromaDB** to store and retrieve domain-specific EV infrastructure facts.
- **Ensemble ML Prediction**: Combines **RandomForest** and **XGBoost** models for high-precision station classification.
- **Rate Limiting**: Built-in 5-second cooldown to manage API usage and protect quotas.
- **Automated Key Management**: Automatically retrieves credentials from environment variables or Streamlit secrets.

---

## 🛠️ Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Agent Framework** | LangGraph, LangChain |
| **Core LLM** | Google Gemini (1.5-flash) |
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
1. **Automated Setup**: Once you have your `GOOGLE_API_KEY` in your `.env` or Streamlit Secrets, just start the app! No manual token entry required.
2. **Chat Naturally**: Ask questions like *"What is the difference between AC and DC charging?"* to trigger the RAG knowledge search.
3. **Run a Prediction**: Ask the agent to predict a station status, and it will guide you through the necessary inputs.
4. **Rate Limiting**: If you message too quickly, the system will ask you to wait a few seconds before your next interaction.

---

## 👥 Contributors
Developed as part of a team effort focusing on Intelligent Intervention in EV infrastructure.

**Maintained by**: [garariya](https://github.com/garariya)
