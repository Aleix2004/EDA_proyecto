import os
import pandas as pd
import matplotlib.pyplot as plt
from data_processing import clean_data


def plot_cleaning_graphics():
    # Crear carpeta de salida
    output_dir = "outputs/02_data_cleaning/"
    os.makedirs(output_dir, exist_ok=True)

    # Cargar datos originales
    df_raw = pd.read_csv("data/raw/avocado.csv")

    # Limpiar datos
    df_clean = clean_data()

    # --- BOXPLOT ANTES ---
    plt.figure(figsize=(8, 5))
    plt.boxplot(df_raw['AveragePrice'], vert=False)
    plt.title('AveragePrice - Antes de limpieza')
    plt.xlabel('AveragePrice')
    plt.savefig(os.path.join(output_dir, "boxplot_before.png"), bbox_inches='tight')
    plt.close()

    # --- BOXPLOT DESPUÉS ---
    plt.figure(figsize=(8, 5))
    plt.boxplot(df_clean['AveragePrice'], vert=False)
    plt.title('AveragePrice - Después de limpieza')
    plt.xlabel('AveragePrice')
    plt.savefig(os.path.join(output_dir, "boxplot_after.png"), bbox_inches='tight')
    plt.close()

    # --- DETECCIÓN DE OUTLIERS (SCATTER) ---
    Q1 = df_raw['AveragePrice'].quantile(0.25)
    Q3 = df_raw['AveragePrice'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_raw['is_outlier'] = (df_raw['AveragePrice'] < lower) | (df_raw['AveragePrice'] > upper)

    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(df_raw)), df_raw['AveragePrice'], s=10)
    plt.scatter(df_raw[df_raw['is_outlier']].index,
                df_raw[df_raw['is_outlier']]['AveragePrice'], s=10)
    plt.title('Detección de Outliers - AveragePrice (Raw)')
    plt.xlabel('Índice')
    plt.ylabel('AveragePrice')
    plt.savefig(os.path.join(output_dir, "outliers_scatter.png"), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    plot_cleaning_diagnostics()
    print("Visualizaciones generadas y guardadas en outputs/02_data_cleaning/")
