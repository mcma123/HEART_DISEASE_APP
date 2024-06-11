import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Function to predict heart disease
def predict_heart_disease(data):
    # Initialize the StandardScaler
    scaler = StandardScaler()
    # Scale the input data
    data_scaled = scaler.fit_transform(data)
    #
    return [0]  #

# Add a background image to the Streamlit app
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url(https://cdn.leonardo.ai/users/3f6ba5a1-8d42-491a-8e0d-2e7632a50c37/generations/59bac8d9-eaa1-4e38-99fc-570bad071e09/Default_dark_background_technology_heart_and_human_heart_analy_0.jpg);
    background-size: 100%;
    background-position: center;
    background-repeat: repeat;
    background-attachment: local;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

#  styles for headers
st.markdown("""
<style>
h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

#  The title of the Streamlit app
st.title("Heart Disease Prediction")

# Input fields for user to enter patient data
# Age: Numeric input for the patient's age
age = st.number_input('Age')

# Sex: Dropdown to select the patient's sex
sex = st.selectbox('Sex', ['Male', 'Female'])

# CP (Chest Pain Type): Dropdown to select the type of chest pain
cp = st.selectbox('CP', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])

# Resting Blood Pressure: Numeric input for resting blood pressure in mmHg
trestbps = st.number_input('Resting Blood Pressure (mmHg)')

# Cholesterol: Numeric input for cholesterol level in mg/dl
chol = st.number_input('Cholesterol in mg/dl')

# Fasting Blood Sugar: Dropdown to select if fasting blood sugar is > 120 mg/dl
fbs = st.selectbox('Fasting Blood Sugar', ['True', 'False'])

# Resting Electrocardiographic Results: Dropdown to select ECG results
restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'Having ST-T wave abnormality', 'Showing probable or definite left ventricular hypertrophy'])

# Maximum Heart Rate Achieved: Numeric input for maximum heart rate achieved
thalach = st.number_input('Maximum Heart Rate Achieved')

# Exercise Induced Angina: Dropdown to select if angina is induced by exercise
exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])

# ST Depression: Numeric input for ST depression induced by exercise relative to rest
oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest')

# Slope of the Peak Exercise ST Segment: Dropdown to select the slope type
slope = st.selectbox('The Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])

# Number of Major Vessels Colored by Flourosopy: Dropdown to select the number of major vessels
ca = st.selectbox('Number of major vessels colored by flourosopy', [0, 1, 2, 3])

# Thalassemia: Dropdown to select the type of thalassemia
thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversable Defect'])

# One-hot encoding for categorical variables
sex_ = 1 if sex == 'Male' else 0
cp_ = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp)
fbs_ = 1 if fbs == 'True' else 0
restecg_ = ['Normal', 'Having ST-T wave abnormality', 'Showing probable or definite left ventricular hypertrophy'].index(restecg)
exang_ = 1 if exang == 'Yes' else 0
slope_ = ['Upsloping', 'Flat', 'Downsloping'].index(slope)
thal_ = ['Normal', 'Fixed Defect', 'Reversable Defect'].index(thal)

# Create a DataFrame to store the input data
columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex_', 'cp_', 'fbs_', 'restecg_', 'exang_', 'slope_', 'thal_', 'ca']
input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
input_df.at[0, 'age'] = age
input_df.at[0, 'trestbps'] = trestbps
input_df.at[0, 'chol'] = chol
input_df.at[0, 'thalach'] = thalach
input_df.at[0, 'oldpeak'] = oldpeak
input_df.at[0, 'sex_'] = sex_
input_df.at[0, 'cp_'] = cp_
input_df.at[0, 'fbs_'] = fbs_
input_df.at[0, 'restecg_'] = restecg_
input_df.at[0, 'exang_'] = exang_
input_df.at[0, 'slope_'] = slope_
input_df.at[0, 'thal_'] = thal_
input_df.at[0, 'ca'] = ca

# Button to trigger prediction
if st.button('Predict'):
    # Predict heart disease using the input data
    result = predict_heart_disease(input_df)
    # Display the prediction result
    if result[0] == 1:
        st.error("The patient is likely to have heart disease.")
    else:
        st.success("The patient is unlikely to have heart disease.")
