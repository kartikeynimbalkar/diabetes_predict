import streamlit as st
import pickle
import pandas as pd
st.title("This is my first frontend project with Streamlit")
final_model=pickle.load(open("svc_diab.pkl","rb"))

# Create input fields for user to enter data
pregnancies = st.number_input("Number of Pregnancies", min_value=0,max_value=20, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=400, step=1)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, step=1)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, step=1.0)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=10.0, step=0.1)
age = st.number_input("Age", min_value=0, max_value=120, step=1)
# i want to save this input from the user in the form of dictionary

# Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
input_data = {
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": diabetes_pedigree_function,
    "Age": age
}
# so that i can convert dict into dataframe and then pass it to the model for prediction
input_df = pd.DataFrame([input_data])
# Create a button to trigger the prediction
if st.button("Predict"):
    # Make prediction using the loaded model
    prediction = final_model.predict(input_df)
    
    # Display the prediction result
    if prediction[0] == 1:
        st.write("The model predicts that the person has diabetes.")
    else:
        st.write("The model predicts that the person does not have diabetes.")