import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_and_inspect_data, get_date_range
from src.data_processing import add_features
from src.visualization import plot_temporal_features

# Configuración
filepath = "data/raw/avocado.csv"
output_folder = "outputs/04_feature_engineering/"
os.makedirs(output_folder, exist_ok=True)

# Cargar y explorar datos
df, info = load_and_inspect_data(filepath)
if "Date" not in df.columns:
    raise KeyError("No se encontró la columna 'Date'.")
get_date_range(df, "Date")

# Transformar datos
df = add_features(df)

# Visualizar características transformadas
plot_temporal_features(df, output_path=output_folder)

print("\nPipeline completado. Visualizaciones guardadas en:", output_folder)
