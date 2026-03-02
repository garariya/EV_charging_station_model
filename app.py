import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np

# Load the saved components
# Ensure these paths are correct relative to where app.py will be run
rf_balanced_retrained_fe = joblib.load('rf_balanced_retrained_fe.joblib')
xgb_cs_retrained_fe = joblib.load('xgb_cs_retrained_fe.joblib')
scaler = joblib.load('scaler.joblib')
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('optimal_threshold.pkl', 'rb') as f:
    optimal_threshold = pickle.load(f)

st.title('EV Charging Station Classifier')
st.markdown("""
This application predicts whether an electric vehicle charging station is a 'Fast DC Charger' based on its characteristics.

Enter the features of the charging station below to get a prediction.
""")

st.header('Enter Charging Station Details:')

# User input widgets
country_options = list(le.classes_)
# Set a default value that exists in the loaded le.classes_ to avoid potential errors
try:
    default_country_index = country_options.index('US') # Try to default to 'US'
except ValueError:
    default_country_index = 0 # Fallback to first option if 'US' not found

country_code_input = st.selectbox('Country Code', country_options, index=default_country_index)
latitude_input = st.number_input('Latitude', value=40.0, format="%.6f")
longitude_input = st.number_input('Longitude', value=-4.0, format="%.6f")
ports_input = st.number_input('Number of Ports', value=2, min_value=1, step=1)

st.subheader('Prediction:')

if st.button('Predict'):
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame([[country_code_input, latitude_input, longitude_input, ports_input]],
                                columns=['country_code', 'latitude', 'longitude', 'ports'])

    # Encode country_code
    try:
        input_data['country_code'] = le.transform(input_data['country_code'])
    except ValueError:
        st.warning(f"Country code '{country_code_input}' not recognized. Using a default encoded value for prediction.")
        # Fallback for unrecognized country codes: could use a common code, or average, etc.
        # A better approach might involve an 'Unknown' category during training and using that value here.
        # For this context, we will use a placeholder or handle explicitly if needed.
        # For now, let's just make it clear it's not recognized and will likely result in a default/arbitrary prediction.
        input_data['country_code'] = -1 # Or a specific encoded value for 'Unknown' from training if applicable

    # Scale numerical features
    num_cols = ['latitude', 'longitude', 'ports']
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # Engineer interaction features (must match training)
    input_data['latitude_x_longitude'] = input_data['latitude'] * input_data['longitude']
    input_data['ports_x_latitude'] = input_data['ports'] * input_data['latitude']
    input_data['ports_x_longitude'] = input_data['ports'] * input_data['longitude']

    # Engineer squared features (must match training)
    input_data['latitude_squared'] = input_data['latitude'] ** 2
    input_data['longitude_squared'] = input_data['longitude'] ** 2
    input_data['ports_squared'] = input_data['ports'] ** 2

    # Make predictions with both models
    rf_prob = rf_balanced_retrained_fe.predict_proba(input_data)[:, 1]
    xgb_prob = xgb_cs_retrained_fe.predict_proba(input_data)[:, 1]

    # Average probabilities for ensemble prediction
    ensemble_prob = (rf_prob + xgb_prob) / 2

    # Apply optimal threshold
    final_prediction = (ensemble_prob >= optimal_threshold).astype(int)

    st.write(f"Predicted Probability of Fast DC Charger: {ensemble_prob[0]:.4f}")
    if final_prediction[0] == 1:
        st.success('Prediction: This is likely a Fast DC Charger.')
    else:
        st.info('Prediction: This is likely NOT a Fast DC Charger.')
