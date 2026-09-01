import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score



df = pd.read_csv("data/telco_clean.csv")

print("\nForma del data set: ")
print(df.shape)

print("\nValores de Churn:")
print(df["Churn"].value_counts())

# Convertir Churn a binario

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0,
})

print("\nValores de Churn después de convertir a binario:")
print(df["Churn"].value_counts())

# Diagnosticar desbalance

print("\nDistribucion porcentual de Churn:")
print(df["Churn"].value_counts(normalize=True) * 100)

# Preparacion de datos para modelado

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
    stratify=y,
)

columnas_numericas = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# Escalado de variables
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

# Regresion logistica
modelo_log = LogisticRegression(max_iter=1000)

modelo_log.fit(X_train_scaled, y_train)

y_pred_log = modelo_log.predict(X_test_scaled)
y_prob_log = modelo_log.predict_proba(X_test_scaled)[:, 1]


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
y_prob_tree = modelo_tree.predict_proba(X_test)[:, 1]

print("\nPrimeras 20 predicciones del arbol de decision:")
print(y_pred_tree[:20])


#Metricas generales regresion logistica

accuracy_log = accuracy_score(
    y_test,
    y_pred_log
)

precision_log = precision_score(
    y_test,
    y_pred_log
)

recall_log = recall_score(
    y_test,
    y_pred_log
)

f1_log = f1_score(
    y_test,
    y_pred_log
)

auc_log = roc_auc_score(y_test, y_prob_log)

matriz_log = confusion_matrix(
    y_test,
    y_pred_log
)

#Resultados regresion logistica

print("\n========================================")
print("REGRESIÓN LOGÍSTICA")
print("========================================")

print("Accuracy:", round(accuracy_log, 4))
print("Precision:", round(precision_log, 4))
print("Recall:", round(recall_log, 4))
print("F1:", round(f1_log, 4))
print("AUC:", round(auc_log, 4))

print("\nMatriz de confusión:")
print(matriz_log)

#Metricas Arbol de decision 

accuracy_tree = accuracy_score(
    y_test,
    y_pred_tree
)

precision_tree = precision_score(
    y_test,
    y_pred_tree
)

recall_tree = recall_score(
    y_test,
    y_pred_tree
)

f1_tree = f1_score(
    y_test,
    y_pred_tree
)

auc_tree = roc_auc_score(
    y_test,
    y_prob_tree
)

matriz_tree = confusion_matrix(
    y_test,
    y_pred_tree
)

#Arbol de decision resultados
print("\n========================================")
print("ÁRBOL DE DECISIÓN")
print("========================================")

print("Accuracy:", round(accuracy_tree, 4))
print("Precision:", round(precision_tree, 4))
print("Recall:", round(recall_tree, 4))
print("F1:", round(f1_tree, 4))
print("AUC:", round(auc_tree, 4))

print("\nMatriz de confusión:")
print(matriz_tree)

#Validacion train vs test

# Regresion logistica 
y_pred_log_train = modelo_log.predict(X_train_scaled)

accuracy_log_train = accuracy_score(
    y_train,
    y_pred_log_train
)

# Árbol de decisión
y_pred_tree_train = modelo_tree.predict(X_train)

accuracy_tree_train = accuracy_score(
    y_train,
    y_pred_tree_train
)

print("\n========================================")
print("COMPARACIÓN TRAIN VS TEST")
print("========================================")

print("\nRegresión logística:")
print("Accuracy train:", round(accuracy_log_train, 4))
print("Accuracy test:", round(accuracy_log, 4))

print("\nÁrbol de decisión:")
print("Accuracy train:", round(accuracy_tree_train, 4))
print("Accuracy test:", round(accuracy_tree, 4))


#Cross validation regresion logistica

preprocesador = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            columnas_numericas
        )
    ],
    remainder="passthrough"
)

pipeline_log = Pipeline([
    ("preprocesamiento", preprocesador),
    ("modelo", LogisticRegression(max_iter=1000))
])

scores_cv = cross_val_score(
    pipeline_log,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\n========================================")
print("CROSS-VALIDATION - REGRESIÓN LOGÍSTICA")
print("========================================")

print("Accuracy de cada fold:")
print(scores_cv)

print("Accuracy media:", round(scores_cv.mean(), 4))
print("Desviación estándar:", round(scores_cv.std(), 4))


# AJUSTE DEL ÁRBOL PARA REDUCIR OVERFITTING

modelo_tree_ajustado = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

modelo_tree_ajustado.fit(
    X_train,
    y_train
)

# Predicciones en train
y_pred_tree_ajustado_train = modelo_tree_ajustado.predict(
    X_train
)

# Predicciones en test
y_pred_tree_ajustado_test = modelo_tree_ajustado.predict(
    X_test
)


# Accuracy train
accuracy_tree_ajustado_train = accuracy_score(
    y_train,
    y_pred_tree_ajustado_train
)

# Accuracy test
accuracy_tree_ajustado_test = accuracy_score(
    y_test,
    y_pred_tree_ajustado_test
)


print("\n========================================")
print("ÁRBOL: ANTES VS DESPUÉS")
print("========================================")

print("\nÁrbol original:")
print("Accuracy train:", round(accuracy_tree_train, 4))
print("Accuracy test:", round(accuracy_tree, 4))

print("\nÁrbol ajustado - max_depth=5:")
print(
    "Accuracy train:",
    round(accuracy_tree_ajustado_train, 4)
)
print(
    "Accuracy test:",
    round(accuracy_tree_ajustado_test, 4)
)