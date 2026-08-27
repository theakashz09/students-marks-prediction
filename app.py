import streamlit as st
import joblib

# Load trained model
model = joblib.load("model.pkl")

# App title
st.title(" Student Marks Prediction")
st.write(" Enter study hours and see predicted marks ")
 
#  user input
hours = st.number_input("How many hours you read?",
                        min_value=0.0,
                        max_value=24.0,
                        value=0.0,
                        step=0.5)

# Prediction button
if st.button("Predict Marks"):
    
    predictions = model.predict([[hours]])
    
    st.success(f"Predicted Marks:{predictions[0]:.2f}")