
import streamlit as st
from src.predict import run

st.title("Readmission Risk Orchestrator")

age = st.slider("Age",0,100)
visits = st.number_input("Emergency Visits",0,10)

if st.button("Predict"):
    prob, risk = run({"age":age,"number_of_emergency_visits":visits})
    st.metric("Risk",prob)
    st.write("Category:",risk)
