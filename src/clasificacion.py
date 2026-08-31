import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data/telco_clean.csv")

print("\nForma del data set: ")
print(df.shape)

print("\nValores de Churn:")
print(df["Churn"].value_counts())

#Convertir Churn a binario

df["Churn"] = df["Churn"].map({
    "Yes": 1, 
    "No": 0
})

print("\nValores de Churn después de convertir a binario:")
print(df["Churn"].value_counts())

#Diagnosticar desbalance

print("\nDistribucion porcentual de Churn:")
print(df["Churn"].value_counts(normalize=True) * 100)

#Preparacion de datos para modelado

X = df.drop(columns=["Churn"])
y = df["Churn"]

print("\nForma de X:")
print(X.shape)

print("\nForma de y:")
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

columnas_numericas = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"  
]

#Escalado de variables
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[columnas_numericas] = scaler.fit_transform(
    X_train[columnas_numericas]
)

X_test_scaled[columnas_numericas] = scaler.transform(
    X_test[columnas_numericas]
)

print("\nVariables numericas ANTES del escalado:")
print(X_train[columnas_numericas].head())
print("\nVariables numericas DESPUES del escalado:")
print(X_train_scaled[columnas_numericas].head())    

#Regresion logistica

modelo_log = LogisticRegression(max_iter=1000)

modelo_log.fit(X_train_scaled, y_train)

y_pred_log = modelo_log.predict(X_test_scaled)


print("\nPrimeras 20 predicciones del modelo de regresion logistica:")
print(y_pred_log[:20])

print("\nPrimeras 20 observaciones reales:")
print(y_test.iloc[:20].values)

#Arbol de decision

modelo_tree = DecisionTreeClassifier(
    random_state=42
)

modelo_tree.fit(X_train, y_train)

y_pred_tree = modelo_tree.predict(X_test)

print("\nPrimeras 20 predicciones del arbol de decision:")
print(y_pred_tree[:20])