import joblib
import streamlit as st
import numpy as np
import pandas as pd

model = joblib.load("insurance_lr_22July.pkl")

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
    st.success(f"Predicted Insurance Response : ${y_unseen_pred:,.2f}")

st.markdown(f''' <style> .stApp {{
    background-image: url("https://media.istockphoto.com/id/1828732247/photo/different-umbrella.jpg?s=2048x2048&w=is&k=20&c=Sw2fqnolwOM-ZINwgbiaoS_aAY6Zuf3GgDckC1V2QSs=");
    background-size: cover;}}</style>''', unsafe_allow_html=True)

