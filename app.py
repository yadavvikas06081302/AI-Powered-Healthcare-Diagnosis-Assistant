import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="AI Healthcare Diagnosis Assistant", page_icon="🩺")
@st.cache_resource
def load_model(): return joblib.load("healthcare_diagnosis_model.joblib")
model=load_model(); data=load_breast_cancer()
st.title("🩺 AI-Powered Healthcare Diagnosis Assistant")
st.caption("Developed by Vikas Yadav")
st.warning("Educational/demo project only. Not for real-world diagnosis or treatment decisions.")
values=[]; cols=st.columns(3)
for i,f in enumerate(data.feature_names):
    with cols[i%3]: values.append(st.number_input(f,value=float(data.data[:,i].mean()),format="%.6f"))
if st.button("🔍 Predict Diagnosis",type="primary"):
    x=np.array(values).reshape(1,-1); p=model.predict(x)[0]; prob=model.predict_proba(x)[0][p]
    st.subheader("Prediction"); st.success(f"Predicted class: {data.target_names[p].title()}"); st.metric("Model confidence",f"{prob*100:.2f}%")
    st.info("Consult a qualified healthcare professional for real medical concerns.")
