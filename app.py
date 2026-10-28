"""
app.py -- web app. Run with:  streamlit run app.py
Loads the trained model (model.pkl) and estimates a colic horse's survival chance.
Any categorical field left as "Unknown" is filled in by the SAME preprocessing used in training.
"""
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from model import sigmoid
from preprocess import Preprocessor  # needed to load the pickle  # noqa: F401

with open("model.pkl", "rb") as f:
    saved = pickle.load(f)
pre, w, b = saved["pre"], saved["w"], saved["b"]

LABELS = {
    "temp_extremities": {1: "Normal", 2: "Warm", 3: "Cool", 4: "Cold"},
    "peripheral_pulse": {1: "Normal", 2: "Increased", 3: "Reduced", 4: "Absent"},
    "mucous_membranes": {1: "Normal pink", 2: "Bright pink", 3: "Pale pink", 4: "Pale cyanotic",
                         5: "Bright red / injected", 6: "Dark cyanotic"},
    "pain": {1: "Alert, no pain", 2: "Depressed", 3: "Intermittent mild pain",
             4: "Intermittent severe pain", 5: "Continuous severe pain"},
    "peristalsis": {1: "Hypermotile", 2: "Normal", 3: "Hypomotile", 4: "Absent"},
    "abdominal_distension": {1: "None", 2: "Slight", 3: "Moderate", 4: "Severe"},
    "nasogastric_tube": {1: "None", 2: "Slight", 3: "Significant"},
    "nasogastric_reflux": {1: "None", 2: "More than 1 litre", 3: "Less than 1 litre"},
    "rectal_exam_feces": {1: "Normal", 2: "Increased", 3: "Decreased", 4: "Absent"},
    "abdomen": {1: "Normal", 2: "Other", 3: "Firm feces in large intestine",
                4: "Distended small intestine", 5: "Distended large intestine"},
    "capillary_refill": {1: "Less than 3 seconds", 2: "3 seconds or more"},
}

st.title("Horse Colic Survival Estimator")
st.caption("Student ML project trained on the UCI Horse Colic dataset (300 cases). "
           "Educational only, not veterinary advice.")

st.subheader("Measurements")
c1, c2 = st.columns(2)
temp = c1.slider("Rectal temperature (°C)", 35.0, 41.0, 38.2, 0.1)
pulse = c2.slider("Pulse (beats/min)", 30, 190, 64)
resp = c1.slider("Respiratory rate (breaths/min)", 8, 100, 24)
pcv = c2.slider("Packed cell volume (%)", 20, 80, 45)
protein = c1.slider("Total protein (g/dL)", 3.0, 14.0, 7.0, 0.1)
surgery = c2.checkbox("Horse had surgery")
young = c1.checkbox("Young horse (under 6 months)")

st.subheader("Clinical examination (choose 'Unknown' if not recorded)")
row = {"rectal_temp": temp, "pulse": pulse, "resp_rate": resp, "packed_cell_volume": pcv,
       "total_protein": protein, "surgery": int(surgery), "age": int(young)}
cols = st.columns(2)
for i, (name, options) in enumerate(LABELS.items()):
    choice = cols[i % 2].selectbox(name.replace("_", " ").capitalize(),
                                   ["Unknown"] + list(options.values()))
    code = next((k for k, v in options.items() if v == choice), np.nan)
    if name == "capillary_refill":
        code = np.nan if code is np.nan or pd.isna(code) else code - 1   # 1/2 -> 0/1, same as training
    row[name] = code

prob = float(sigmoid(pre.transform(pd.DataFrame([row])) @ w + b)[0])
st.metric("Estimated survival probability", f"{prob * 100:.1f}%")
if prob >= 0.5:
    st.success("The model leans towards survival.")
else:
    st.error("The model leans towards a poor outcome.")
