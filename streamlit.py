import pickle
import streamlit as st

global_model = pickle.load(open('Predict Global Air Pollution modelling.sav','rb'))

st.title('AQI Category Prediction using ML (Classification)')
