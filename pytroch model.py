import lime
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# 确保模型类 MLPModel 已经被导入（如果它存放在外部文件中）

import torch
import pickle

# 加载保存的模型
with open('MLP_model_default.pkl', 'rb') as file:
    model = pickle.load(file)

# 将模型切换到评估模式
model.eval()

print("模型加载完成，并切换到评估模式")



# Define feature names
feature_names = [
    "Symptom Duration",
    "Pain Catastrophizing Scale",
    "Deep Cervical Flexor Endurance Test",
    "Flexion/Extension ROM",
    "Rotation ROM",
    "spring test",
    "Pain Sensitivity Scale",
    "NDI",
    "BMI",
    "Presence of muscle tightness",
    "Exacerbation on Rotation",
    "Flexion",
    "Exacerbation on Extension",
    "Lifting Heavy Objects"
]

# Streamlit app title
st.title("MLP-Based Deep Learning Model for Predicting Patient Benefit from Manual Therapy")

# Create input fields for features
symptom_duration = st.number_input("Symptom Duration:", min_value=2, max_value=28, value=25, step=1)
pain_catastrophizing_scale = st.number_input("Pain Catastrophizing Scale:", min_value=13, max_value=35, value=25,
                                             step=1)
deep_cervical_flexor_endurance_test = st.number_input("Deep Cervical Flexor Endurance Test:", min_value=6, max_value=12,
                                                      value=10, step=1)
Flexion_Extension_ROM = st.number_input("Flexion/Extension ROM:", min_value=69, max_value=84, value=70, step=1)
rotation_rom = st.number_input("Rotation ROM:", min_value=100, max_value=140, value=120, step=1)
spring_test = st.selectbox("Spring Test (0=No, 1=middle, 2=sever):", options=[0, 1, 2])
pain_sensitivity_scale = st.number_input("Pain Sensitivity Scale:", min_value=13, max_value=37, value=25, step=1)
NDI = st.number_input("NDI:", min_value=26.66666667, max_value=37.77777778, value=30, step=1)
BMI = st.number_input("BMI:", min_value=24.17, max_value=25.02, value=25, step=1)
presence_of_muscle_tightness = st.selectbox("Presence of muscle tightness (0=No, 1=middle, 2=sever):",
                                            options=[0, 1, 2])
exacerbation_on_rotation = st.selectbox("Exacerbation on Rotation (0=No, 1=Yes):", options=[0, 1])
flexion = st.number_input("Flexion:", min_value=36, max_value=45, value=40, step=1)
exacerbation_on_extension = st.selectbox("Exacerbation on Extension (0=No, 1=Yes):", options=[0, 1])
lifting_heavy_objects = st.number_input("Lifting Heavy Objects:", min_value=0, max_value=5, value=2, step=1)

# Create feature values array
feature_values = [
    symptom_duration, pain_catastrophizing_scale, deep_cervical_flexor_endurance_test, Flexion_Extension_ROM, rotation_rom,
    spring_test, rotation_rom, pain_sensitivity_scale, NDI, BMI, presence_of_muscle_tightness, exacerbation_on_rotation, flexion,
    exacerbation_on_extension, lifting_heavy_objects]

# Create DataFrame
features = pd.DataFrame([feature_values], columns=feature_names)

# Add prediction button
if st.button("Predict"):
    # Perform prediction
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you tend to benefit from the manipulation of bone setting. "
            f"The model predicts that your probability of benefiting is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may benefit from Manipulation of bone setting."
        )
    else:
        advice = (
            f"According to our model, you tend not to benefit from the manipulation of bone setting. "
            f"The model predicts that your probability of not benefiting is {probability:.1f}%. "
        )

    st.write(advice)

    # SHAP explanation part
    explainer = shap.Explainer(model, features)  # Using the model directly for SHAP
    shap_values = explainer.shap_values(features)

    sample_shap_values = shap_values[0]
    expected_value = explainer.expected_value[0]

    # Create SHAP explanation object
    explanation = shap.Explanation(
        values=sample_shap_values[:, 0],  # SHAP values
        base_values=expected_value,  # Expected values
        data=features.iloc[0].values,  # Input data
        feature_names=feature_names  # Feature names
    )

    # Save SHAP plot as HTML
    shap.save_html("shap_force_plot.html", shap.plots.force(explanation, show=False))

    # Display SHAP force plot in Streamlit
    st.subheader("模型预测的力图")
    with open("shap_force_plot.html", "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=600)
