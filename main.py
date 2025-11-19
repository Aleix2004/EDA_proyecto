import sys
import os
import pandas as pd

# --- Asegurar que src esté en el path ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_and_inspect_data, get_date_range
from src.data_processing import add_features
from src.visualization import plot_temporal_features

def main():

    # --- 1. Crear carpeta de outputs si no existe ---
    output_folder = "outputs/04_feature_engineering/"
    os.makedirs(output_folder, exist_ok=True)

    # --- 2. Load raw data ---
    filepath = "data/raw/avocado.csv"
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

    df, info = load_and_inspect_data(filepath)

    # --- 3. Validar columna Date ---
    if "Date" not in df.columns:
        raise KeyError("El dataset no contiene la columna 'Date'. Verifica tu CSV.")

    # --- 4. Basic date range info ---
    get_date_range(df, "Date")

    # --- 5. Process data → new time-based features ---
    df = add_features(df)

    # --- 6. Visualizaciones de features temporales ---
    plot_temporal_features(df, output_path=output_folder)

    print("\nPipeline completed successfully.")
    print(f"All outputs saved to {output_folder}")


if __name__ == "__main__":
    main()
