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





