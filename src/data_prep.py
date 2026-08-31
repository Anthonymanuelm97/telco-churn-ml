import pandas as pd

df = pd.read_csv("data/telco.csv")

#Exploración de datos
print("Forma del data set: ", df.shape)

print("\nTipos de datos: ", df.dtypes)

print("\nValores nulos: ", df.isnull().sum())

#Buscar espacios en blancos en columnas de tipo str e identificarlos

print("\nEspacios en blanco en TotalCharges:")
print(df["TotalCharges"].str.strip().eq("").sum())

filas_vacias = df[df["TotalCharges"].str.strip().eq("")]

print("\nFilas con TotalCharges vacío:")
print(filas_vacias)

#Los valores vacios de TotalCharges estan relacionados a ternure = 0.

#Conversion de TotalCharges a tipo numérico, reemplazando los valores vacíos por 0
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("\nTipo de TotalCharges después de limpiar:")
print(df["TotalCharges"].dtype)

print("\nNulos después de limpiar TotalCharges:")
print(df["TotalCharges"].isnull().sum())

print("\nClientes con tenure 0:")
print(df.loc[df["tenure"] == 0, ["tenure", "MonthlyCharges", "TotalCharges"]])

#Eliminacion de columnas no utiles para modelar

df = df.drop(columns=["customerID"])
print("\nColumnas después de eliminar customerID:")
print(df.shape)

#Verificar columnas categoricas 

print("\nColumnas de texto:")
print(df.select_dtypes(include=["object", "str"]).columns.tolist())

#Encoding

columnas_categoricas = df.select_dtypes(include=["object", "str"]).columns.tolist()

columnas_predictoras_categoricas = [
    columna for columna in columnas_categoricas
    if columna != "Churn"
]

df_encoded = pd.get_dummies(
    df,
    columns=columnas_predictoras_categoricas,
    drop_first=False,
    dtype=int
)

print("\nForma del data set después de encoding:")
print(df_encoded.shape)

print("\nTipos de datos despues del encoding:")
print(df_encoded.dtypes)

print("\nColumnas de texto restantes:")
print(
    df_encoded.select_dtypes(
    include=["object", "str"]
    ).columns.tolist()
)

df_encoded.to_csv("data/telco_clean.csv", index=False)
print("\nData set limpio guardado en 'data/telco_clean.csv'")