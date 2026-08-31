import pandas as pd

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