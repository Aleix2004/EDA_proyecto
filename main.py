import sys
import os

# --- Asegurar que src esté en el path ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_and_inspect_data, get_date_range
from src.data_processing import clean_data, add_features, transform_data
from src.visualization import plot_temporal_features, plot_transformed_features

def main():
    # --- 1. Configuración ---
    raw_filepath = "data/raw/avocado.csv"
    output_folder_fe = "outputs/04_feature_engineering/"
    output_folder_trans = "outputs/03_transformation/"
    os.makedirs(output_folder_fe, exist_ok=True)
    os.makedirs(output_folder_trans, exist_ok=True)
    
    print("=" * 60)
    print("PIPELINE DE PROCESAMIENTO DE DATOS")
    print("=" * 60)
    
    # --- 2. Cargar y explorar datos RAW ---
    print("\n=== PASO 1: CARGA Y EXPLORACIÓN ===")
    df_raw, info = load_and_inspect_data(raw_filepath)
    
    if "Date" not in df_raw.columns:
        raise KeyError("No se encontró la columna 'Date'.")
    
    get_date_range(df_raw, "Date")
    
    # --- 3. Limpiar datos ---
    print("\n=== PASO 2: LIMPIEZA DE DATOS ===")
    df_clean = clean_data(save=True)
    print(f"✓ Dimensiones después de limpieza: {df_clean.shape}")
    
    # --- 4. Transformar datos (Feature Engineering) ---
    print("\n=== PASO 3: TRANSFORMACIÓN Y FEATURE ENGINEERING ===")
    df_transformed = add_features(df_clean, save=True)
    print(f"✓ Dimensiones después de transformación: {df_transformed.shape}")
    print(f"✓ Nuevas columnas agregadas: {[col for col in df_transformed.columns if col not in df_clean.columns]}")
    
    # --- 5. Aplicar transformaciones adicionales ---
    print("\n=== PASO 4: TRANSFORMACIONES ADICIONALES ===")
    df_final = transform_data(df_transformed)
    print(f"✓ Transformaciones aplicadas: type_encoded, Price_per_Volume, proporciones")
    
    # --- 6. Visualizar características transformadas ---
    print("\n=== PASO 5: VISUALIZACIÓN ===")
    plot_temporal_features(df_transformed, output_path=output_folder_fe)
    plot_transformed_features(df_final, output_dir=output_folder_trans)
    
    print("\n" + "=" * 60)
    print("✓ PIPELINE COMPLETADO EXITOSAMENTE")
    print(f"✓ Visualizaciones de feature engineering en: {output_folder_fe}")
    print(f"✓ Visualizaciones de transformaciones en: {output_folder_trans}")
    print("=" * 60)

if __name__ == "__main__":
    main()