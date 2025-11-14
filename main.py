import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_and_inspect_data, get_date_range
from src.visualization import plot_initial_exploration

def main():
    filepath = "data/raw/avocado.csv"

    # 1️⃣ Cargar datos y obtener resumen
    df, info = load_and_inspect_data(filepath)

    # 2️⃣ Mostrar rango de fechas si existe columna 'Date'
    if "Date" in df.columns:
        get_date_range(df, "Date")

    # 3️⃣ Variables categóricas para gráficos
    categorical_vars = info["categorical_vars"]

    # 4️⃣ Generar todas las visualizaciones iniciales
    #    - Heatmap de nulos
    #    - Gráficos de barras para variables categóricas
    #    - Top 10 regiones
    #    - Gráficos de series temporales agrupadas por mes (o frecuencia deseada)
    plot_initial_exploration(df, categorical_vars, output_folder="outputs/01_initial_exploration/")

    print("\nPipeline completado correctamente. Todas las visualizaciones están en 'outputs/01_initial_exploration/'.")

if __name__ == "__main__":
    main()
