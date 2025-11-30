import sys
import os

# Asegurar que src este en el path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_and_inspect_data, get_date_range
from src.data_processing import clean_data, add_features, transform_data
from src.visualization import plot_temporal_features, plot_transformed_features, plot_cleaning_graphics
from src.analysis import run_complete_eda

def main():
    # Configuracion inicial
    raw_filepath = "data/raw/avocado.csv"
    output_folder_fe = "outputs/04_feature_engineering/"
    output_folder_trans = "outputs/03_transformation/"
    os.makedirs(output_folder_fe, exist_ok=True)
    os.makedirs(output_folder_trans, exist_ok=True)
    
    print("=" * 60)
    print("PIPELINE DE PROCESAMIENTO Y ANALISIS DE DATOS")
    print("Fases: Carga, Limpieza, Transformacion y EDA")
    print("=" * 60)
    
    # Fase 1: Carga y exploracion inicial
    print("\n=== PASO 1: CARGA Y EXPLORACION ===")
    df_raw, info = load_and_inspect_data(raw_filepath)
    
    if "Date" not in df_raw.columns:
        raise KeyError("No se encontro la columna 'Date'.")
    
    get_date_range(df_raw, "Date")
    
    # Fase 2: Limpieza de datos
    print("\n=== PASO 2: LIMPIEZA DE DATOS ===")
    df_clean = clean_data(save=True)
    print(f"Dimensiones despues de limpieza: {df_clean.shape}")
    
    # Generar visualizaciones de limpieza
    plot_cleaning_graphics()
    
    # Fase 3: Transformacion y feature engineering
    print("\n=== PASO 3: TRANSFORMACION Y FEATURE ENGINEERING ===")
    df_transformed = add_features(df_clean, save=True)
    print(f"Dimensiones despues de transformacion: {df_transformed.shape}")
    print(f"Nuevas columnas agregadas: {[col for col in df_transformed.columns if col not in df_clean.columns]}")
    
    # Aplicar transformaciones adicionales
    print("\n=== PASO 4: TRANSFORMACIONES ADICIONALES ===")
    df_final = transform_data(df_transformed)
    print(f"Transformaciones aplicadas: type_encoded, Price_per_Volume, proporciones")
    
    # Visualizar caracteristicas transformadas
    print("\n=== PASO 5: VISUALIZACION DE FEATURE ENGINEERING ===")
    plot_temporal_features(df_transformed, output_path=output_folder_fe)
    plot_transformed_features(df_final, output_dir=output_folder_trans)
    
    # Fase 4: Analisis exploratorio completo
    print("\n=== PASO 6: ANALISIS EXPLORATORIO DE DATOS (EDA) ===")
    print("Ejecutando analisis exploratorio completo...")
    run_complete_eda(df_final)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print("\nSALIDAS GENERADAS:")
    print(f"  - Limpieza de datos: outputs/02_data_cleaning/")
    print(f"  - Transformaciones: {output_folder_trans}")
    print(f"  - Feature Engineering: {output_folder_fe}")
    print(f"  - EDA Univariante: outputs/05_EDA/univariate/")
    print(f"  - EDA Bivariante: outputs/05_EDA/bivariate/")
    print(f"  - EDA Correlaciones: outputs/05_EDA/correlation/")
    print("=" * 60)

if __name__ == "__main__":
    main()