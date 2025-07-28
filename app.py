import joblib
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

model = joblib.load("insurance_model_22July.pkl")

# Streamlit app
st.title("Insurance Response Prediction")

# Define input option
Gender = ['Male', 'Female']
Age = (0,100)
Driving_License = [0,1]
Region_Code = (0,53)
Previously_Insured = [0,1]
Vehicle_Age = ['> 2 Years', '1-2 Year', '< 1 Year']
Vehicle_Damage = ['Yes', 'No']
Annual_Premium = (0,550000)
Policy_Sales_Channel = (0,200)
Vintage = (0,300)

# user inputs
Gender_selected = st.selectbox("Select Gender", Gender)
Age_selected = st.slider("Select age", min_value=1, max_value=100, value=1)
Driving_License_selected = st.radio("Do you have a driving license?", Driving_License)
Region_Code_selected = st.slider("Select region code", min_value=1, max_value=100, value=1)
Previously_Insured_selected = st.radio("Were you previously insuranced?", Previously_Insured)
Vehicle_Age_selected = st.selectbox("Select vehicle age", Vehicle_Age)
Vehicle_Damage_selected = st.radio("Was vehicle damaged?", Vehicle_Damage)
Annual_Premium_selected = st.slider("Select annual premium", min_value=1, max_value=550000, value=1)
Policy_Sales_Channel_selected = st.slider("Select policy sales channal", min_value=1, max_value=200, value=1)
Vintage_selected = st.slider("Select vintage", min_value=1, max_value=300, value=1)



# Prediction button
if st.button("Predict"):
    # Create dict for input features
    input_data = {
        "Gender" : Gender_selected,
        "Age" : Age_selected,
        "Driving_License" : Driving_License_selected,
        "Region_Code" : Region_Code_selected,
        "Previously_Insured" : Previously_Insured_selected,
        "Vehicle_Age" : Vehicle_Age_selected,
        "Vehicle_Damage" : Vehicle_Damage_selected,
        "Annual_Premium" : Annual_Premium_selected,
        "Policy_Sales_Channel" : Policy_Sales_Channel_selected,
        "Vintage" : Vintage_selected,
    }

    # Convert input data to DataFrame
    df_input = pd.DataFrame({
        "Gender" : [Gender_selected],
        "Age" : [Age_selected],
        "Driving_License" : [Driving_License_selected],
        "Region_Code" : [Region_Code_selected],
        "Previously_Insured" : [Previously_Insured_selected],
        "Vehicle_Age" : [Vehicle_Age_selected],
        "Vehicle_Damage" : [Vehicle_Damage_selected],
        "Annual_Premium" : [Annual_Premium_selected],
        "Policy_Sales_Channel" : [Policy_Sales_Channel_selected],
        "Vintage" : [Vintage_selected],
    })

    # One-hot encoding
    df_input_ohe = pd.get_dummies(df_input, 
                                  columns = ['Gender','Vehicle_Age','Vehicle_Damage'])
    
    # df_input_ohe = df_input_ohe.to_numpy()

    feature_names = [
    'Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
    'Annual_Premium', 'Policy_Sales_Channel', 'Vintage',
    'Gender_Female', 'Gender_Male',
    'Vehicle_Age_1-2 Year', 'Vehicle_Age_< 1 Year', 'Vehicle_Age_> 2 Years',
    'Vehicle_Damage_No', 'Vehicle_Damage_Yes'
    ]

    df_input_reindex = df_input_ohe.reindex(columns = feature_names, 
                                            fill_value = 0)
    
    # Predict
    y_unseen_pred = model.predict(df_input_reindex)[0]
    st.success(f"Predicted Insurance Response : {y_unseen_pred:,.2f}%")

 

st.markdown(f''' <style> .stApp {{
    background-image: url(data:image/jpeg;base64,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);
    background-size: cover;}}</style>''', unsafe_allow_html=True)

# color: #46c928;
# font-color: #46c928;
# stslider{{background-color: #46c928;}}
