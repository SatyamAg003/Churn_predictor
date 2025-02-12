#Gender 1 - female 0 male
#Churn 1 yes 0 no 
#scaler is imported as scaler.pkl 
#model is model.pkl using logisticreg..
# x order is->> Age,Geder,Tenure MonthlyCharges

import streamlit as st 
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Customer Churn Prediction App ")
st.divider()

st.write("Enter the values and click on predict to get a prediction of whether a customer will continue using the service or not!!")

st.divider()

age = st.number_input("Enter Age" , min_value=10,max_value=100,value=30)
tenure= st.number_input("Enter Tenure",min_value=0,max_value=130, value=10)
monthly_charges = st.number_input("Enter Monthly Charges",min_value=30,max_value =150,value =100)
gender =st.selectbox("Enter Gender",["Male","Female"])

st.divider()

predictbutton = st.button("Predict!")
st.divider()
if predictbutton:
    gender_selected =1 if gender =="Female" else 0
    X = [age,gender_selected,tenure,monthly_charges]
    x1 = np.array(X)
    X_array = scaler.transform([x1])
    prediction = model.predict(X_array)[0]
    predicted  = "Yes" if prediction==1 else "No"
    st.balloons()
    
    st.write(f"predicted: {predicted}")
else : 
    st.write("Please enter all the values and click on the predict button to get the prediction!!")