import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

# Cargar el modelo y el DictVectorizer
with open('churn-model.pck', 'rb') as f:
    dv, model = pickle.load(f)

# Título de la aplicación
st.title("Predicción de Churn de Clientes")

# Formulario para introducir datos del cliente
st.header("Introduce los datos del cliente:")
contract = st.selectbox("Tipo de contrato", ["month-to-month", "one_year", "two_year"])
dependents = st.selectbox("Dependientes", ["no", "yes"])
device_protection = st.selectbox("Protección de dispositivos", ["no", "yes", "no_internet_service"])
gender = st.selectbox("Género", ["female", "male"])
internet_service = st.selectbox("Tipo de Internet", ["dsl", "fiber_optic", "no"])
monthly_charges = st.number_input("Cargos mensuales", min_value=0.0, max_value=500.0, step=0.1)
tenure = st.number_input("Antigüedad (meses)", min_value=0, max_value=100, step=1)
total_charges = st.number_input("Cargos totales", min_value=0.0, max_value=10000.0, step=0.1)
payment_method = st.selectbox("Método de pago", ["bank_transfer_(automatic)", "credit_card_(automatic)", "electronic_check", "mailed_check"])

# Botón de predicción
if st.button("Predecir"):
    # Crear un diccionario con los datos del cliente
    client_data = {
        "contract": contract,
        "dependents": dependents,
        "deviceprotection": device_protection,
        "gender": gender,
        "internetservice": internet_service,
        "monthlycharges": monthly_charges,
        "tenure": tenure,
        "totalcharges": total_charges,
        "paymentmethod": payment_method
    }

    # Transformar los datos del cliente
    X_client = dv.transform([client_data])

    # Realizar la predicción
    y_pred_proba = model.predict_proba(X_client)[0][1]  # Probabilidad de churn

    # Mostrar resultado
    st.subheader("Resultado:")
    if y_pred_proba > 0.5:
        st.error(f"El cliente tiene una alta probabilidad de churn: {y_pred_proba:.2f}")
    else:
        st.success(f"El cliente tiene una baja probabilidad de churn: {y_pred_proba:.2f}")
