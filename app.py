# image: https://www.freepik.com/
import joblib
import streamlit as st
import numpy as np
import pandas as pd


model = joblib.load("insurance_model.pkl")

# Streamlit app
st.title("Insurance Response Prediction")
st.subheader("Our client is an Insurance company that has provided Health Insurance to its customers, now they need predict whether the policyholders (customers) from past year will also be interested in Vehicle Insurance provided by the company.")
st.subheader("Now let's put in customer's information for prediction")
# Define input option
Age = (0,100)
Region_Code = (0,53)
Previously_Insured = [0,1]
Vehicle_Age = ['1-2 Year', '< 1 Year']
Vehicle_Damage = ['Yes', 'No']
Annual_Premium = (0,550000)
Policy_Sales_Channel = (0,200)
Vintage = (0,300)

# user inputs
Age_selected = st.slider("Select age", min_value=1, max_value=100, value=1)
Region_Code_selected = st.slider("Select region code", min_value=1, max_value=100, value=1)
Previously_Insured_selected = st.radio("Were you previously insuranced?", Previously_Insured)
Vehicle_Age_selected = st.selectbox("Select vehicle age", Vehicle_Age)
Vehicle_Damage_selected = st.radio("Was vehicle damaged?", Vehicle_Damage)
Annual_Premium_selected = st.slider("Select annual premium", min_value=1, max_value=600000, value=1)
Policy_Sales_Channel_selected = st.slider("Select policy sales channal", min_value=1, max_value=200, value=1)
Vintage_selected = st.slider("Select vintage", min_value=1, max_value=300, value=1)



# Prediction button
if st.button("Predict"):
    # Create dict for input features
    input_data = {
        "Age" : Age_selected,
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
        "Age" : [Age_selected],
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
    'Age', 'Region_Code', 'Previously_Insured',
    'Annual_Premium', 'Policy_Sales_Channel', 'Vintage',
    'Vehicle_Age_1-2 Year', 'Vehicle_Age_< 1 Year', 
    'Vehicle_Damage_No', 'Vehicle_Damage_Yes'
    ]

    df_input_reindex = df_input_ohe.reindex(columns = feature_names, 
                                            fill_value = 0)
    
    # Predict
    y_unseen_pred = model.predict(df_input_reindex)[0]
    if y_unseen_pred == 1:
        result = "Yes" 
    else:
        result = "No"
    st.success(f"Will this client interested in Vehicle Insurance ? Prediction result: {result}")

    

st.markdown(f''' 
    <style> 
    .stApp {{   
    background-image: url("https://raw.githubusercontent.com/Genevieve-crypto/MLDP-Geneieve-Project/refs/heads/main/car-background.png");
    background-size: cover;
    }}

    st.markdown{{
    <h1 style='color:#002147;'>Insurance Response Prediction</h1>
    }}
            
    .stButton > button {{
        background-color: #002147;
        color: white;
        border-radius: 15px;
        padding: 10px 20px;
        font-size: 28px;
    }}

    </style>''', unsafe_allow_html=True)

