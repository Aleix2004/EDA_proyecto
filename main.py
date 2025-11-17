from src.data_loader import load_and_inspect_data, get_date_range
from src.data_processing import transform_data
from src.visualization import plot_transformed_features

def main():
    filepath = "data/raw/avocado.csv"

    # Cargar y explorar datos
    df, info = load_and_inspect_data(filepath)

    # Transformar datos
    df = transform_data(df)

    # Visualizar características transformadas
    plot_transformed_features(df)

    # Obtener rango de fechas
    get_date_range(df, "Date")

if __name__ == "__main__":
    main()
