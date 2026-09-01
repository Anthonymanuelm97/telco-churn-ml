# telco-churn-ml
# Telco Customer Churn - Machine Learning

Proyecto final del Mes 5 de Machine Learning Supervisado.

El proyecto utiliza el dataset Telco Customer Churn para resolver dos problemas diferentes utilizando el mismo conjunto de datos:

1. Predecir la antigüedad (`tenure`) de un cliente mediante regresión.
2. Predecir si un cliente abandonará el servicio (`Churn`) mediante clasificación.

---

## Dataset

El dataset contiene información de 7,043 clientes de una empresa de telecomunicaciones, incluyendo:

- Perfil demográfico
- Servicios contratados
- Tipo de contrato
- Método de pago
- Cargos mensuales
- Cargos totales
- Antigüedad del cliente
- Abandono del servicio

---

## Preparación de datos

Durante la preparación se realizaron los siguientes pasos:

- Diagnóstico de forma, tipos de datos y valores nulos.
- Identificación de 11 valores vacíos en `TotalCharges`.
- Los 11 registros tenían `tenure = 0`, por lo que `TotalCharges` fue reemplazado por 0.
- Conversión de `TotalCharges` a tipo numérico.
- Eliminación de `customerID`, ya que es solamente un identificador.
- Codificación de variables categóricas mediante `pd.get_dummies()`.

---

## Modelo de regresión

El objetivo del modelo de regresión es predecir la antigüedad del cliente (`tenure`) utilizando su perfil y los servicios contratados.

Se utilizó `LinearRegression`.

`TotalCharges` fue excluido del modelo para evitar data leakage, ya que está fuertemente relacionado con `tenure`.

### Resultados

- MAE: 11.5685 meses
- RMSE: 14.5593 meses
- R²: 0.6633

El modelo se equivoca en promedio aproximadamente 11.57 meses y explica alrededor del 66.33% de la variación observada en la antigüedad de los clientes.

---

## Modelo de clasificación

El objetivo es predecir si un cliente abandonará el servicio.

La variable `Churn` fue convertida a:

- No = 0
- Yes = 1

La distribución encontrada fue:

- No: 73.46%
- Yes: 26.54%

Se compararon dos modelos:

- Logistic Regression
- Decision Tree Classifier

### Regresión logística

- Accuracy: 0.8055
- Precision: 0.6572
- Recall: 0.5588
- F1: 0.6040
- AUC: 0.8420

Matriz de confusión:

[[926, 109],
 [165, 209]]

### Árbol de decisión

- Accuracy: 0.7197
- Precision: 0.4730
- Recall: 0.4920
- F1: 0.4823
- AUC: 0.6467

Matriz de confusión:

[[830, 205],
 [190, 184]]

La regresión logística fue seleccionada como el mejor modelo porque obtuvo mejores resultados en todas las métricas evaluadas y produjo menos falsos negativos.

---

## Validación y Overfitting

### Regresión logística

Accuracy:

- Train: 0.8060
- Test: 0.8055

Cross-validation con 5 folds:

- Accuracy media: 0.8042
- Desviación estándar: 0.0143

Los resultados muestran un desempeño estable entre entrenamiento, prueba y los diferentes folds de validación.

### Árbol de decisión

El árbol original mostró señales claras de overfitting:

- Train: 0.9980
- Test: 0.7197

Después de limitar la profundidad:

`max_depth = 5`

los resultados fueron:

- Train: 0.8039
- Test: 0.7984

Limitar la profundidad redujo considerablemente el overfitting y mejoró el rendimiento sobre datos no vistos.

---

## Cliente hipotético

Se creó un cliente nuevo con las siguientes características principales:

- Género: Male
- Partner: No
- Dependents: No
- Internet: Fiber optic
- Online Security: No
- Tech Support: No
- Contract: Month-to-month
- Payment Method: Electronic check
- Monthly Charges: $95.50

### Predicciones

Antigüedad estimada:

19.75 meses

Probabilidad de abandono:

74.24%

Predicción:

Churn = Yes

Para un gerente de la empresa, este cliente podría considerarse de alto riesgo de abandono. Sería recomendable evaluar una estrategia de retención, como una oferta personalizada o una revisión de su plan.

---

## Tecnologías

- Python
- pandas
- scikit-learn
- matplotlib
- joblib

---

## Ejecución

Preparación de datos:

```bash
python src/data_prep.py