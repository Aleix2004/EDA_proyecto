import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def ensure_output_dir(path):
    """Crea el directorio si no existe."""
    if not os.path.exists(path):
        os.makedirs(path)


def plot_missing_values_heatmap(df, output_path):
    """
    Genera un heatmap mostrando los valores nulos del dataset.
    """
    if df.isnull().sum().sum() == 0:
        print("No se generó heatmap: el dataset no contiene valores nulos.")
        return

    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Heatmap de Valores Nulos")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "heatmap_missing_values.png"))
    plt.close()
    print("Heatmap de nulos guardado.")


def plot_categorical_counts(df, col, output_path, top_n=None):
    """
    Grafica los conteos de una variable categórica.
    Limpia strings antes de contar para evitar barras iguales por diferencias de mayúsculas o espacios.
    """
    if col not in df.columns:
        print(f"Columna '{col}' no encontrada, se omite gráfico.")
        return

    # Limpiar la columna: convertir a string, quitar espacios y pasar a minúsculas
    df[col] = df[col].astype(str).str.strip().str.lower()

    counts = df[col].value_counts()

    if top_n is not None:
        counts = counts.head(top_n)

    plt.figure(figsize=(12, 6))
    counts.plot(kind="bar")
    plt.title(f"Frecuencia de valores en '{col}'")
    plt.ylabel("Conteo")
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = f"value_counts_{col}.png"
    plt.savefig(os.path.join(output_path, filename))
    plt.close()
    print(f"Gráfico de '{col}' guardado.")


def plot_time_series(df, date_col, value_col, freq='M', output_path="outputs/01_initial_exploration/"):
    """
    Grafica una serie temporal agrupada por frecuencia (mes, año, trimestre)
    """
    if date_col not in df.columns or value_col not in df.columns:
        print(f"Columnas '{date_col}' o '{value_col}' no encontradas, se omite gráfico.")
        return

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['period'] = df[date_col].dt.to_period(freq)
    grouped = df.groupby('period')[value_col].mean().reset_index()

    plt.figure(figsize=(12,6))
    plt.bar(grouped['period'].astype(str), grouped[value_col])
    plt.xticks(rotation=45)
    plt.title(f"{value_col} promedio por {freq}")
    plt.xlabel(date_col)
    plt.ylabel(value_col)
    plt.tight_layout()

    filename = os.path.join(output_path, f"{value_col}_{date_col}_{freq}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Gráfico temporal guardado en {filename}")


def plot_initial_exploration(df, categorical_vars, output_folder="outputs/01_initial_exploration/"):
    """
    Genera visualizaciones iniciales del dataset:
      - Heatmap de valores nulos
      - Gráficos de barras para variables categóricas
      - Top 10 regiones si existe la columna 'region'
      - Gráficos de series temporales para columnas de fecha + valor numérico
    """
    ensure_output_dir(output_folder)
    print(f"Guardando visualizaciones en: {output_folder}")

    # 1️⃣ Heatmap de nulos
    plot_missing_values_heatmap(df, output_folder)

    # 2️⃣ Gráficos de variables categóricas
    for col in categorical_vars:
        plot_categorical_counts(df, col, output_folder)

    # 3️⃣ Top 10 regiones (si existe)
    if "region" in df.columns:
        plot_categorical_counts(df, "region", output_folder, top_n=10)

    # 4️⃣ Gráficos de series temporales (para columnas de tipo fecha + valor numérico)
    # Aquí asumimos que si existe columna 'Date' y alguna numérica como 'AveragePrice'
    if "Date" in df.columns:
        numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
        for value_col in numeric_cols:
            plot_time_series(df, "Date", value_col, freq='M', output_path=output_folder)

    print("Visualizaciones generadas correctamente.")
