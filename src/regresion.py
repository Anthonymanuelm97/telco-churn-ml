import pandas as pd
from sklearn.model_selection import train_test_split


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

