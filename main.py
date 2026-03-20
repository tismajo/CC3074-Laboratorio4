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

#  6 — Predicción y evaluación del árbol base

y_pred_base = arbol_reg_base.predict(X_test)

mse_base  = mean_squared_error(y_test, y_pred_base)
rmse_base = np.sqrt(mse_base)
r2_base   = r2_score(y_test, y_pred_base)

print(f"\nArbol regresion base (d=5)")
print(f"MSE: {mse_base:.4f}")
print(f"RMSE: {rmse_base:.4f}")
print(f"R2: {r2_base:.4f}")

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_base, alpha=0.3, color="steelblue")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Precio real")
plt.ylabel("Precio predicho")
plt.title(f"Árbol Regresión Base | R²={r2_base:.3f}  RMSE={rmse_base:.2f}")
plt.tight_layout()
plt.show()


#  7 — 3 modelos adicionales cambiando profundidad

configs_reg = [
    {"max_depth": 3,  "label": "depth=3"},
    {"max_depth": 10, "label": "depth=10"},
    {"max_depth": 15, "label": "depth=15"},
]

resultados_reg = []

for cfg in configs_reg:
    m = DecisionTreeRegressor(max_depth=cfg["max_depth"], random_state=42)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    mse  = mean_squared_error(y_test, pred)
    r2   = r2_score(y_test, pred)
    resultados_reg.append({
        "Modelo": cfg["label"],
        "MSE":    round(mse, 4),
        "RMSE":   round(np.sqrt(mse), 4),
        "R²":     round(r2, 4)
    })
    print(f"{cfg['label']} -> ECM={mse:.2f} R2={r2:.4f}")

# Agregar modelo base a la tabla
resultados_reg.insert(0, {
    "Modelo": "depth=5 (base)",
    "MSE":    round(mse_base, 4),
    "RMSE":   round(rmse_base, 4),
    "R²":     round(r2_base, 4)
})

tabla_reg = pd.DataFrame(resultados_reg)
print("\nComparacion arboles regresion:")
print(tabla_reg.to_string(index=False))

# Guardar el mejor árbol de regresión (depth=10 según informe)
mejor_arbol_reg = DecisionTreeRegressor(max_depth=10, random_state=42)
mejor_arbol_reg.fit(X_train, y_train)
y_pred_mejor_reg = mejor_arbol_reg.predict(X_test)
mse_mejor_reg  = mean_squared_error(y_test, y_pred_mejor_reg)
rmse_mejor_reg = np.sqrt(mse_mejor_reg)
r2_mejor_reg   = r2_score(y_test, y_pred_mejor_reg)

#  8 — Regresión lineal Ridge con canal + RidgeCV

# Canal: StandardScaler → RidgeCV
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

pipeline_ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge",  RidgeCV(alphas=alphas, cv=5,
                       scoring="neg_root_mean_squared_error"))
])

pipeline_ridge.fit(X_train, y_train)

mejor_alpha = pipeline_ridge.named_steps["ridge"].alpha_
print(f"\nMejor alfa seleccionado por RidgeCV: {mejor_alpha}")

y_pred_ridge = pipeline_ridge.predict(X_test)
mse_ridge    = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge   = np.sqrt(mse_ridge)
r2_ridge     = r2_score(y_test, y_pred_ridge)

print(f"\nRidge Pipeline (alfa={mejor_alpha})")
print(f"MSE: {mse_ridge:.4f}")
print(f"RMSE: {rmse_ridge:.4f}")
print(f"R2: {r2_ridge:.4f}")

# VC sobre el canal completo
cv_rmse = -cross_val_score(
    pipeline_ridge, X_train, y_train,
    cv=5, scoring="neg_root_mean_squared_error"
)
print(f"\nVC5 RECM: {cv_rmse.mean():.4f} +- {cv_rmse.std():.4f}")

# Comparación Ridge vs Mejor Árbol
comp_reg = pd.DataFrame({
    "Modelo": [f"Árbol Regresión (depth=10)", f"Ridge Pipeline (α={mejor_alpha})"],
    "MSE":    [round(mse_mejor_reg, 4), round(mse_ridge, 4)],
    "RMSE":   [round(rmse_mejor_reg, 4), round(rmse_ridge, 4)],
    "R²":     [round(r2_mejor_reg, 4), round(r2_ridge, 4)]
})
print("\nComparacion final regresion:")
print(comp_reg.to_string(index=False))

# Gráfica comparativa
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
colores = ["#4C72B0", "#DD8452"]
axes[0].bar(comp_reg["Modelo"], comp_reg["RMSE"], color=colores)
axes[0].set_title("RMSE (menor = mejor)")
axes[0].set_ylabel("RMSE")
axes[0].tick_params(axis="x", rotation=12)
axes[1].bar(comp_reg["Modelo"], comp_reg["R²"], color=colores)
axes[1].set_title("R² (mayor = mejor)")
axes[1].set_ylabel("R²")
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis="x", rotation=12)
plt.suptitle("Árbol de Regresión vs Ridge Pipeline", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# 9 — Variable categórica: Económica / Intermedia / Cara

q33 = df["price"].quantile(0.33)
q66 = df["price"].quantile(0.66)

print(f"\nP33 (Economica): ${q33:.2f}")
print(f"P66 (Intermedia): ${q66:.2f}")

def clasificar_precio(p):
    if p <= q33:
        return "Economica"
    elif p <= q66:
        return "Intermedia"
    else:
        return "Cara"

df["precio_cat"] = df["price"].apply(clasificar_precio)

dist_cat = df["precio_cat"].value_counts().reset_index()
dist_cat.columns = ["Categoría", "Cantidad"]
dist_cat["Porcentaje"] = (dist_cat["Cantidad"] / len(df) * 100).round(2)
print("\nDistribucion de categorias:")
print(dist_cat.to_string(index=False))

plt.figure(figsize=(6, 4))
df["precio_cat"].value_counts().plot(kind="bar", color=["#2ecc71", "#f39c12", "#e74c3c"])
plt.title("Distribución de categorías de precio")
plt.ylabel("Cantidad")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# PREPARACIÓN PARA CLASIFICACIÓN
df_cls = df_num_model.copy()

# Agregar la variable categórica
df_cls["precio_cat"] = df["precio_cat"].values

# Quitar price (la variable de la que se derivó precio_cat)
df_cls.drop(columns=["price"], inplace=True, errors="ignore")

# Mantener filas e imputar nulos en predictores numéricos.
X_cls = df_cls.drop(columns=["precio_cat"]).copy()
X_cls = X_cls.dropna(axis=1, how="all")
X_cls = X_cls.fillna(X_cls.median(numeric_only=True))

y_cls = df_cls["precio_cat"]

if X_cls.empty or y_cls.empty:
    print("No hay datos pa clasificar.")
    sys.exit(1)

X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

print(f"\nClasificación — Train: {X_cls_train.shape} | Prueba: {X_cls_test.shape}")
print("Distribución en train: ")
print(y_cls_train.value_counts())


# 10 — Árbol de clasificación

arbol_cls_base = DecisionTreeClassifier(
    criterion="gini", max_depth=5, random_state=42
)
arbol_cls_base.fit(X_cls_train, y_cls_train)

# Visualización del árbol
plt.figure(figsize=(28, 10))
plot_tree(
    arbol_cls_base,
    feature_names=X_cls.columns.tolist(),
    class_names=["Cara", "Economica", "Intermedia"],
    filled=True, rounded=True, fontsize=7, max_depth=3
)
plt.title("Árbol de Clasificación (depth=5, primeros 3 niveles)")
plt.tight_layout()
plt.show()


