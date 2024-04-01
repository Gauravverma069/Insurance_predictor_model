import streamlit as st
import joblib
import numpy as np
# from PIL import Image


st.image("logo.png",width = 150)
st.write("# Acko Health Insurance Predictor App")

Age = st.number_input("Enter your Age",min_value = 10,max_value = 90,value = 30,step = 1)
Height = st.number_input("Enter your Height in Meters",min_value = 0.6,max_value = 2.7,value = 1.67)
Weight = st.number_input("Enter your Weight in kilogram",min_value = 25,max_value = 200,value = 80)
BMI = Weight/Height**2
BodyMassIndex = st.write(f" Your BMI is {round(BMI,1)}")

Children = st.number_input("Enter No. of Children",min_value = 0,max_value = 10,value = 0,step = 1)

Sex = st.selectbox("Enter your Gender",("Male","Female"))

Smoker = st.selectbox("Do you Smoke?",("Yes","No"))
if Smoker == "Yes":
    st.write("\u26A0 Smoking is injurious to health! \u26A0")


Smoker_num = 0 if Smoker == "No" else 1
test_data = [[Age,BMI,Children,Smoker_num]]

# Model Load
model = joblib.load("insurance_joblib")
poly = joblib.load("poly_obj")

if st.button("Get Quote"):
    
    test_poly = poly.transform(test_data)
    y_pred_log = model.predict(test_poly)
    premium = round(np.exp(y_pred_log)[0],2)
    st.write(f"## **Your Premium Amount is ${premium}**")
