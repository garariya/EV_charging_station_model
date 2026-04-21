# EV Charging Station AI Agent

### From Predictive Analytics to Agentic Intelligence

## Project Overview

This project represents the evolution of an EV Charging Station Classifier into a fully agentic AI system. Initially developed as a machine learning pipeline to predict whether a station supports Fast DC Charging, the system has been extended into an autonomous conversational agent capable of reasoning, retrieving domain knowledge, and executing predictive workflows.

The architecture follows the ReAct (Reasoning and Acting) paradigm, enabling the agent to interact with users, gather required inputs iteratively, and deliver insights supported by both machine learning models and a retrieval-augmented knowledge base.

---

## Key Features

* **Agentic Orchestration**
  Built using LangGraph, the system maintains conversational state and dynamically decides when to invoke tools.

* **Retrieval-Augmented Generation (RAG)**
  ChromaDB is used to store and retrieve EV infrastructure knowledge for contextual responses.

* **Ensemble Machine Learning**
  Combines RandomForest and XGBoost models to improve classification accuracy and recall.

* **Rate Limiting Mechanism**
  A 5-second cooldown prevents excessive API usage and protects system quotas.

* **Automated Credential Management**
  API keys are securely loaded from environment variables or Streamlit secrets.

---

## Technology Stack

| Layer               | Technology                |
| ------------------- | ------------------------- |
| Agent Framework     | LangGraph, LangChain      |
| Core LLM            | Google Gemini (1.5-flash) |
| Vector Database     | ChromaDB                  |
| Machine Learning    | Scikit-learn, XGBoost     |
| Data Processing     | Pandas, NumPy             |
| Embeddings          | ONNX MiniLM-L6-V2         |
| Backend & Interface | Streamlit                 |
| Model Persistence   | Joblib, Pickle            |

---

## System Evolution

### Phase 1: Core ML Pipeline

* Developed baseline classification models using RandomForest and XGBoost.
* Addressed class imbalance using `class_weight='balanced'`.
* Established evaluation metrics including ROC AUC and F1-score.

### Phase 2: Optimization and Enhancement

* Introduced feature engineering techniques such as interaction terms (Latitude × Longitude) and polynomial transformations (Ports²).
* Applied hyperparameter tuning using `RandomizedSearchCV`.
* Built an ensemble model using weighted probability averaging to improve recall performance.

### Phase 3: Agentic Integration

* Encapsulated the ML pipeline as a LangChain tool.
* Integrated a RAG layer for dynamic knowledge retrieval.
* Implemented a LangGraph-based state machine to manage multi-turn interactions and decision-making.

---

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/garariya/EV_charging_station_model.git
cd EV_charging_station_model
```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Model and Data Setup

Ensure the following files are available in the root directory:

* `rf_balanced_retrained_fe.joblib`
* `xgb_cs_retrained_fe.joblib`
* `scaler.joblib`
* `label_encoder.pkl`
* `optimal_threshold.pkl`

### 4. Run the Application

```bash
streamlit run app.py
```

---

## Usage Guide

* **Initialization**
  Configure your `GOOGLE_API_KEY` via `.env` or Streamlit secrets before launching the application.

* **Knowledge Queries**
  Ask domain-related questions (e.g., differences between AC and DC charging) to utilize the RAG system.

* **Prediction Workflow**
  Request a prediction, and the agent will guide you through required inputs step-by-step.

* **Rate Limiting Behavior**
  If interactions occur too rapidly, the system will enforce a short delay before accepting the next request.

---

## Contributors

Developed as part of a collaborative effort focused on intelligent intervention in EV infrastructure systems.

**Maintained by**: garariya
GitHub: [https://github.com/garariya](https://github.com/garariya)