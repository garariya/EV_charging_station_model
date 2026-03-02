
# EV Charging Station Classification Using Machine Learning

## From Predictive Analytics to Intelligent Intervention

### Project Overview
This project focuses on the development of an **AI-driven Electric Vehicle (EV) Charging Station Classifier**. The system predicts whether a charging station is a 'Fast DC Charger' based on its characteristics, evolving through various machine learning and optimization stages to provide accurate and actionable insights.

- **Milestone 1: Core ML Pipeline**: Application of classical machine learning techniques for data preprocessing, feature engineering, and initial model training to predict Fast DC Charger status.
- **Milestone 2: Optimized & Deployable Solution**: Extension of the core ML pipeline with advanced optimization techniques, including class imbalance handling, hyperparameter tuning, ensemble modeling, and development of a user-friendly Streamlit application.

---

### Constraints & Requirements
- **Team Size:** A team of 4 students.
- **API Budget:** Free Tier Only (Open-source models / Free APIs).
- **Framework:** Scikit-learn, XGBoost, Pandas, Numpy, Matplotlib, Seaborn.
- **UI Framework:** Streamlit.
- **Hosting:** Mandatory (Streamlit Community Cloud).

---

### Technology Stack
| Component | Technology |
| :--- | :--- |
| **Data Manipulation** | Pandas, Numpy |
| **ML Models** | RandomForestClassifier (Scikit-learn), XGBClassifier (XGBoost) |
| **Preprocessing** | StandardScaler (Scikit-learn), LabelEncoder (Scikit-learn) |
| **Model Tuning** | RandomizedSearchCV (Scikit-learn) |
| **Evaluation & Visualization** | Matplotlib, Seaborn, Scikit-learn Metrics |
| **UI Framework** | Streamlit |
| **Deployment Utility** | joblib, pickle, Git LFS |

---

### Milestones & Deliverables

#### Milestone 1: Core ML Pipeline Development (Mid-Sem Focus)
**Objective:** To establish a robust machine learning pipeline for predicting 'Fast DC Charger' status, focusing on data preparation, initial model selection, and handling class imbalance.

**Key Deliverables:**
- **Problem Understanding & Business Context**: Defined the importance of classifying Fast DC Chargers for efficient EV infrastructure planning and user guidance.
- **Data Cleaning & Preprocessing**: Handled missing values, removed duplicates, encoded categorical features, and scaled numerical features.
- **Initial Model Training**: Developed baseline RandomForest and XGBoost models.
- **Class Imbalance Handling**: Implemented `class_weight='balanced'` for RandomForest and `scale_pos_weight` for XGBoost to address the minority class issue.
- **Local Application with UI**: (Awaiting Streamlit deployment success).
- **Model Performance Evaluation Report**: Initial reports for RandomForest (Initial, Balanced) and XGBoost models (Accuracy, ROC AUC, Precision, Recall, F1-score).

#### Milestone 2: Optimized & Deployable Solution (End-Sem Focus)
**Objective:** To enhance model accuracy, particularly for the minority class, through advanced feature engineering and ensemble methods, culminating in a publicly deployable Streamlit application.

**Key Deliverables:**
- **Advanced Feature Engineering**: Implemented interaction terms (`latitude_x_longitude`, `ports_x_latitude`, `ports_x_longitude`) and squared terms (`latitude_squared`, `longitude_squared`, `ports_squared`).
- **Hyperparameter Tuning**: Optimized the RandomForest model using `RandomizedSearchCV`.
- **Optimal Classification Thresholding**: Identified and applied optimal probability thresholds for both RandomForest and XGBoost models to maximize minority class F1-score.
- **Ensemble Model Creation**: Developed a simple probability-averaging ensemble model combining optimized RandomForest and XGBoost predictions.
- **Final Model Performance Report**: Comprehensive comparison of all models, highlighting the superior performance of the Ensemble Model in terms of ROC AUC and minority class recall.
- **Publicly Deployed Application**: A Streamlit application hosted on Streamlit Community Cloud for interactive predictions.
- **GitHub Repository & Complete Codebase**: The project's code, models, and necessary files committed to GitHub, including `requirements.txt` and Git LFS for large model artifacts.


---

### Evaluation Criteria

| Phase | Weight | Criteria |
| :--- | :--- | :--- |
| **Mid-Sem** | 25% | Data preprocessing quality, Initial ML model application, Effective class imbalance handling, Initial UI usability, Clear evaluation metrics. |
| **End-Sem** | 30% | Impact of advanced feature engineering, Effectiveness of hyperparameter tuning, Robustness of ensemble method, Clarity of Streamlit application, Successful deployment & accessibility, Comprehensive code documentation. |

---

### How to Run Locally

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/garariya/EV_charging_station_model.git
    cd EV_charging_station_model
    ```
2.  **Install Git LFS** (if not already installed, for model files):
    ```bash
    git lfs install
    git lfs pull
    ```
3.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```
4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```
    Your application will open in your web browser.

---

### Publicly Deployed Application

https://evchargingstationmodel-atxappj73smuikjcggqwnxg.streamlit.app/


---


