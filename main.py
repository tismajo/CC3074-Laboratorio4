"""
=============================================================
CC3074 - Minería de Datos 
Laboratorio 4: Árboles de Decisión
Integrantes:
  - Mia Alejandra Fuentes Merida, 23775
  - María José Girón Isidro, 23559
  - Leonardo Dufrey Mejía Mejía, 23648
=============================================================
"""

# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

try:
    import pyreadr
except ModuleNotFoundError:
    pyreadr = None

from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV
)
from sklearn.tree import (
    DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

plt.rcParams["figure.figsize"] = (12, 7)
plt.style.use("ggplot")

# 1 — Descarga / carga del conjunto de datos

RDATA_PATH = "./data/listings.RData"
CSV_PATH = "./data/collectedData.csv"


def load_csv_fallback(csv_path, required_col="price", chunksize=100_000):
    """Carga un CSV de forma segura validando el encabezado primero.

    Esto evita lecturas pesadas cuando el CSV de respaldo no coincide
    con el esquema esperado de Airbnb.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"No esta el archivo: {csv_path}")
        return None

    header = None
    used_encoding = None
    for enc in ("utf-8", "latin-1"):
        try:
            header = pd.read_csv(csv_file, nrows=0, encoding=enc)
            used_encoding = enc
            break
        except UnicodeDecodeError:
            continue

    if header is None:
        print("No pude leer el encabesado del CSV.")
        return None

    if required_col not in header.columns:
        print(
            f"Le falta '{required_col}' al CSV."
        )
        print(
            "Ese CSV no se ve del dataset q toca."
        )
        return None

    frames = []
    try:
        for chunk in pd.read_csv(
            csv_file,
            encoding=used_encoding,
            low_memory=True,
            chunksize=chunksize,
        ):
            frames.append(chunk)
    except pd.errors.ParserError as e:
        print(f"No pude parsear el CSV: {e}")
        return None

    if not frames:
        print("El CSV esta vacio.")
        return None

    return pd.concat(frames, ignore_index=True)


df_raw = None

if pyreadr is not None:
    try:
        resultado_r = pyreadr.read_r(RDATA_PATH)
        if resultado_r:
            df_raw = next(iter(resultado_r.values()))
            print(f"Datos cargados de {RDATA_PATH}")
        else:
            print(f"No salio tabla en {RDATA_PATH}")
    except Exception as e:
        print(f"No pude leer {RDATA_PATH} con pyreadr: {e}")
else:
    print("No esta instalado pyreadr.")

if df_raw is None:
    print(f"Probando CSV respaldo: {CSV_PATH}")
    df_raw = load_csv_fallback(CSV_PATH)
    if df_raw is not None:
        print(f"Datos cargados de {CSV_PATH}")

if df_raw is None:
    print("No pude cargar datos validos.")
    print(
        "Instala pyreadr o revisa q el CSV tenga 'price'."
    )
    sys.exit(1)

if "price" not in df_raw.columns:
    print("No viene la columna 'price'.")
    print("Revisa listings.RData ")
    sys.exit(1)

print("Datos listos.")
print(f"Tamaño orig: {df_raw.shape}")


#  2 — Análisis exploratorio + preprocesamiento

#  2.1 Vista general 
print(f"\nRegistros: {df_raw.shape[0]}")
print(f"Variables: {df_raw.shape[1]}")
print("\nTipos:")
print(df_raw.dtypes.value_counts())
print("\nNulos top15:")
print(df_raw.isna().sum().sort_values(ascending=False).head(15))

# 2.2 Conversión de variables 
df = df_raw.copy()

# precio: quitar "$" y "," y convertir a flotante
df["price"] = (
    df["price"]
    .astype(str)
    .str.replace(r"[$,]", "", regex=True)
    .str.strip()
    .replace("", np.nan)
)
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# Porcentajes a numérico
for col in ["host_response_rate", "host_acceptance_rate"]:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
            .replace("", np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Binarias t/f a 1/0  (compatible con pandas 2.x y 3.x)
bool_cols = [
    c for c in df.columns
    if str(df[c].dtype) in ("object", "string")
    and df[c].dropna().isin(["t", "f"]).all()
]
for col in bool_cols:
    df[col] = df[col].map({"t": 1, "f": 0})

#  2.3 Eliminar columnas irrelevantes 
drop_cols = [c for c in df.columns if any(
    kw in c.lower() for kw in [
        "url", "description", "name", "summary", "space",
        "neighborhood_overview", "notes", "transit", "access",
        "interaction", "house_rules", "thumbnail", "medium",
        "picture", "xl_picture", "host_about", "scrape_id",
        "last_scraped", "calendar_last_scraped", "license"
    ]
)]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

# 2.4 Imputación de nulos en numéricas clave  (evita ChainedAssignmentError)
for col in ["bathrooms", "bedrooms", "beds"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

#  2.5 Eliminar valores atípicos en 'price'
df = df[(df["price"] > 0) & (df["price"] <= 1000)].copy()
df.dropna(subset=["price"], inplace=True)

print(f"\nDatos limpios: {df.shape}")
#  2.6 Estadísticas descriptivas 
print("\nEstadisticas de price:")
print(df["price"].describe())

# Distribución del precio
plt.figure(figsize=(8, 4))
sns.histplot(df["price"], bins=60, kde=True, color="steelblue")
plt.title("Distribución del precio por noche (USD)")
plt.xlabel("Precio")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

# 2.7 Precio vs capacidad 
if "accommodates" in df.columns:
    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=df, x="accommodates", y="price", alpha=0.2)
    plt.title("Capacidad vs Precio")
    plt.tight_layout()
    plt.show()

#  2.8 Precio por tipo de habitación 
if "room_type" in df.columns:
    plt.figure(figsize=(8, 4))
    df.groupby("room_type")["price"].mean().sort_values().plot(kind="bar", color="steelblue")
    plt.title("Precio promedio por tipo de habitación")
    plt.ylabel("Precio promedio (USD)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

# 2.9 Precio vs número de reviews 
if "number_of_reviews" in df.columns:
    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=df, x="number_of_reviews", y="price", alpha=0.2, color="coral")
    plt.title("Precio vs Número de reviews")
    plt.tight_layout()
    plt.show()

#  2.10 Mapa de correlación 
df_num = df.select_dtypes(include=["int64", "float64"])
plt.figure(figsize=(14, 10))
sns.heatmap(df_num.corr(), cmap="RdBu_r", center=0,
            linewidths=0.3, annot=False)
plt.title("Mapa de correlación entre variables numéricas")
plt.tight_layout()
plt.show()


#  3 — Análisis de grupos

print("\n--- Grupos ---")

# Grupo 1: Tipo de habitación
if "room_type" in df.columns:
    g1 = df.groupby("room_type")["price"].mean().round(2).reset_index()
    g1.columns = ["Tipo habitación", "Precio promedio"]
    print("\nGrupo habitacion:")
    print(g1.to_string(index=False))

# Grupo 2: Capacidad
if "accommodates" in df.columns:
    df["capacity_group"] = pd.cut(
        df["accommodates"],
        bins=[0, 2, 5, 10, 100],
        labels=["1-2", "3-5", "6-10", "11+"]
    )
    g2 = df.groupby("capacity_group", observed=True)["price"].mean().round(2).reset_index()
    g2.columns = ["Capacidad", "Precio promedio"]
    print("\nGrupo capacidad:")
    print(g2.to_string(index=False))

    plt.figure(figsize=(7, 4))
    df.groupby("capacity_group", observed=True)["price"].mean().plot(
        kind="bar", color="steelblue")
    plt.title("Precio promedio por capacidad")
    plt.ylabel("Precio promedio (USD)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Grupo 3: Número de reviews
if "number_of_reviews" in df.columns:
    df["review_group"] = pd.cut(
        df["number_of_reviews"],
        bins=[-1, 10, 50, 200, 9999],
        labels=["0-10", "11-50", "51-200", "200+"]
    )
    g3 = df.groupby("review_group", observed=True)["price"].mean().round(2).reset_index()
    g3.columns = ["Reviews", "Precio promedio"]
    print("\nGrupo reviews:")
    print(g3.to_string(index=False))


#  4 — Split entrenamiento / prueba

# Seleccionar solo numéricas para los modelos (descartamos categóricas por ahora)
# Eliminamos columnas auxiliares que creamos en el EDA
drop_eda = ["capacity_group", "review_group"]
df_model = df.drop(columns=drop_eda, errors="ignore")

# Quedarnos con columnas numéricas (int32, int64, float64, etc.)
df_num_model = df_model.select_dtypes(include=[np.number]).copy()

# Evitar perder todas las filas por nulos dispersos: imputamos en X.
if "price" not in df_num_model.columns:
    print("No esta 'price' pa modelar.")
    sys.exit(1)

y = df_num_model["price"].copy()
X = df_num_model.drop(columns=["price"]).copy()

# Quitar columnas completamente vacías y luego imputar con mediana.
X = X.dropna(axis=1, how="all")
X = X.fillna(X.median(numeric_only=True))

if X.empty or y.empty:
    print("No hay datos sufisientes pa entrenar.")
    sys.exit(1)

print(
    f"Columnas numericas: {X.shape[1]} | Filas entreno: {X.shape[0]}"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nEntreno: {X_train.shape}")
print(f"Prueba: {X_test.shape}")


# 
# 5 — Árbol de regresión con todas las variables
arbol_reg_base = DecisionTreeRegressor(max_depth=5, random_state=42)
arbol_reg_base.fit(X_train, y_train)

# Visualización
plt.figure(figsize=(28, 10))
plot_tree(arbol_reg_base, feature_names=X.columns.tolist(),
          filled=True, rounded=True, fontsize=7, max_depth=3)
plt.title("Árbol de Regresión (profundidad=5, primeros 3 niveles)")
plt.tight_layout()
plt.show()

