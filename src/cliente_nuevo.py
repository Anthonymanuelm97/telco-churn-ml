import pandas as pd
import joblib

#Cliente hipotetico 

cliente = pd.DataFrame([{
    "customerID": "NEW-0001",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.50
}])

print("\nCliente hipotetico: ")
print(cliente)

# Cargar Modelo de Regresion


modelo_reg = joblib.load(
    "models/modelo_regresion.pkl"
)

columnas_reg = joblib.load(
    "models/columnas_regresion.pkl"
)

#Eliminar identificador y encoding

cliente_reg = cliente.drop(columns=["customerID"])

cliente_reg = pd.get_dummies(
    cliente_reg,
    dtype=int
)
cliente_reg = cliente_reg.reindex(
    columns=columnas_reg,
    fill_value=0
)

tenure_predicho = modelo_reg.predict(
    cliente_reg
)[0]

print("\nTenure estimado:")
print(round(tenure_predicho, 2), "meses")


# ============================================================
# 3. PREPARAR CLIENTE PARA CLASIFICACIÓN
# ============================================================

modelo_clf = joblib.load(
    "models/modelo_clasificacion.pkl"
)

scaler = joblib.load(
    "models/scaler_clasificacion.pkl"
)

columnas_clf = joblib.load(
    "models/columnas_clasificacion.pkl"
)


# Preparacion de cliente para clasificacion

modelo_clf = joblib.load(
    "models/modelo_clasificacion.pkl"
)

scaler = joblib.load(
    "models/scaler_clasificacion.pkl"
)

columnas_clf = joblib.load(
    "models/columnas_clasificacion.pkl"
)


# Copiamos el cliente original
cliente_clf = cliente.copy()

# Agregamos tenure estimado
cliente_clf["tenure"] = tenure_predicho

# Estimamos TotalCharges
cliente_clf["TotalCharges"] = (
    cliente_clf["MonthlyCharges"] * tenure_predicho
)

# customerID no se usa para modelar
cliente_clf = cliente_clf.drop(
    columns=["customerID"]
)

#Hacer encoding
cliente_clf = pd.get_dummies(
    cliente_clf,
    dtype=int
)

#Alinear columnas 
cliente_clf = cliente_clf.reindex(
    columns=columnas_clf,
    fill_value=0
)

#Escalar variables numericas 

columnas_numericas = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

cliente_clf_scaled = cliente_clf.copy()

cliente_clf_scaled[columnas_numericas] = scaler.transform(
    cliente_clf[columnas_numericas]
)

#Predecir el churn

prediccion_churn = modelo_clf.predict(
    cliente_clf_scaled
)[0]

probabilidad_churn = modelo_clf.predict_proba(
    cliente_clf_scaled
)[0, 1]

print("\nPredicción de Churn:")

if prediccion_churn == 1:
    print("Sí, es probable que el cliente cancele.")
else:
    print("No, es probable que el cliente permanezca.")

print(
    "Probabilidad de cancelación:",
    round(probabilidad_churn * 100, 2),
    "%"
)


#Para el cliente hipotético analizado, el modelo de regresión estima una antigüedad aproximada de 19.75 meses. 
#El modelo de clasificación estima una probabilidad de abandono del 74.24%, por lo que el cliente sería considerado de alto riesgo
#y podría ser conveniente aplicar una estrategia de retención, como una oferta personalizada o una revisión de su plan.


#Resultados de cliente con todas las columnas

cliente_resultado = cliente.copy()

cliente_resultado["tenure"] = round(
    tenure_predicho,
    2
)

cliente_resultado["TotalCharges"] = round(
    cliente_resultado["MonthlyCharges"] * tenure_predicho,
    2
)

cliente_resultado["Churn"] = (
    "Yes" if prediccion_churn == 1 else "No"
)

print("\nCliente hipotético con resultados completos:")
print(cliente_resultado)