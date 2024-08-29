import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
expected_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst'];
st.title('Breast Cancer Prediction')
inputs = {}
for feature in expected_features:
    inputs[feature] = st.number_input(feature, min_value=0.0)
def preprocess_and_predict(features_dict):
    features_df = pd.DataFrame([features_dict], columns=expected_features)
    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)
    return 'Malignant' if prediction[0] == 1 else 'Benign'
if st.button('Predict'):
    result = preprocess_and_predict(inputs)
    st.write(f'The prediction is: {result}')
