import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("data/telco_clean.csv")

X=df.drop(columns=["tenure", "TotalCharges", "Churn"])
Y = df["tenure"]

print("\nForma de X:")
print(X.shape)

print("\nForma de Y:")
print(Y.shape)

print("\nColumnas usadas en X:")
print(X.columns.tolist())

#Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print("\nForma de X_train:", X_train.shape)
print("\nForma de X_test:", X_test.shape)
print("\nForma de y_train:", y_train.shape)
print("\nForma de y_test:", y_test.shape)


#Entrenar el modelo
modelo_reg = LinearRegression()
modelo_reg.fit(X_train, y_train)

#Realizar predicciones
y_pred = modelo_reg.predict(X_test)
print("\nPrimeras 10 predicciones:")
print(y_pred[:10])

print("\nPrimeras 10 observaciones reales:")
print(y_test.iloc[:10].values)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

print("\nMetricas de evaluacion del modelo de regresion:")

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")