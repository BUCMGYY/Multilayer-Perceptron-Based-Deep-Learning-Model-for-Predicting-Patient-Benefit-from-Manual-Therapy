import lime
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO

from matplotlib.axes import Axes
from shap import TreeExplainer

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Define feature names
feature_names = [
    "Age", "Symptom Duration", "Flexion", "Extension", "Rotation ROM",
    "spring test", "Presence of muscle tightness", "Exacerbation on Flexion",
    "Exacerbation on Extension", "Pain Catastrophizing Scale", "Pain Sensitivity Scale"
]

# Streamlit app title
st.title("Multilayer Perceptron-Based Deep Learning Model for Predicting Patient Benefit from Manual Therapy")

# Create input fields for features
age = st.number_input("Age:", min_value=12, max_value=72, value=50, step=1)
symptom_duration = st.number_input("Symptom Duration (days):", min_value=1, max_value=360, value=25, step=1)
flexion = st.number_input("Flexion ROM:", min_value=15, max_value=60, value=40, step=1)
extension = st.number_input("Extension ROM:", min_value=15, max_value=50, value=36, step=1)
rotation_rom = st.number_input("Rotation ROM:", min_value=35, max_value=80, value=50, step=1)

# Classification variables (spring test, muscle tightness, exacerbation)
spring_test = st.selectbox("Spring Test (0=No, 1=middle, 2=severe):", options=[0, 1, 2])
presence_of_muscle_tightness = st.selectbox("Presence of muscle tightness (0=No, 1=middle, 2=severe):", options=[0, 1, 2])
exacerbation_on_flexion = st.selectbox("Exacerbation on Flexion (0=No, 1=Yes):", options=[0, 1])
exacerbation_on_extension = st.selectbox("Exacerbation on Extension (0=No, 1=Yes):", options=[0, 1])

# Continuous variables (pain scales)
pain_catastrophizing_scale = st.number_input("Pain Catastrophizing Scale:", min_value=2, max_value=51, value=25, step=1)
pain_sensitivity_scale = st.number_input("Pain Sensitivity Scale:", min_value=11, max_value=86, value=25, step=1)

# Create feature values array
feature_values = [
    age, symptom_duration, flexion, extension, rotation_rom,
    spring_test, presence_of_muscle_tightness, exacerbation_on_flexion, exacerbation_on_extension,
    pain_catastrophizing_scale, pain_sensitivity_scale
]


# features = np.array([feature_values])
    
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

import shap
import streamlit as st

# 使用树模型的 SHAP 解释器
explainer = shap.TreeExplainer(model)

# 计算给定实例的 SHAP 值
shap_values = explainer.shap_values(features)

# 获取期望值（模型的平均预测值）
expected_value = explainer.expected_value[0]  # 获取类别 0 的期望值（如果是多类别调整）

# 初始化 JavaScript 库，用于绘制交互式图形
shap.initjs()

# 生成 SHAP force plot
shap_html_str = shap.force_plot(expected_value, shap_values[0], features, plot_cmap="coolwarm", show=False).html()

# 在 Streamlit 中显示 SHAP force plot
st.subheader("模型预测的 SHAP Force Plot")
st.components.v1.html(shap_html_str, height=500)  # 显示交互式 force plot
    # # Generate explanation
    # exp = explainer.explain_instance(features[0], model.predict_proba)
    #
    # # Show LIME explanation as a table
    # st.write("**LIME Explanation of the Model Prediction:**")
    # exp.show_in_notebook()
    #
    # # Or visualize the explanation
    # fig = exp.as_pyplot_figure()
    # st.pyplot(fig)
