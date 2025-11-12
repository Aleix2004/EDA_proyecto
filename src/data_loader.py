import pandas as pd

def load_and_inspect_data(filepath):
    """
    Carga un dataset CSV, realiza inspección inicial y clasifica las variables.

    Parámetros
    ----------
    filepath : str
        Ruta del archivo CSV (por ejemplo: 'data/raw/avocado.csv')

    Retorna
    -------
    df : pandas.DataFrame
        DataFrame con los datos cargados
    info_dict : dict
        Información resumida del dataset (tamaño, tipos de variables, nulos, etc.)
    """

    # Cargar el dataset
    df = pd.read_csv(filepath)
    print(f"Archivo cargado correctamente: {filepath}")
    print("=" * 60)

    # Información básica
    print("\n=== INFORMACIÓN GENERAL ===")
    print(f"Dimensiones del dataset: {df.shape}")
    print("\n--- Tipos de datos ---")
    print(df.info())

    # Primeras filas
    print("\n=== VISTA PRELIMINAR ===")
    print(df.head())

    # Estadísticas descriptivas
    print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
    print(df.describe(include="all").T)

    # Valores nulos y duplicados
    print("\n=== REVISIÓN DE NULOS Y DUPLICADOS ===")
    print("Valores nulos por columna:\n", df.isnull().sum())
    print(f"\nTotal de filas duplicadas: {df.duplicated().sum()}")

    # Clasificación de variables
    print("\n=== CLASIFICACIÓN DE VARIABLES ===")
    numeric_vars = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_vars = df.select_dtypes(include=["object", "category"]).columns.tolist()
    temporal_vars = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

    print(f"Variables numéricas: {numeric_vars}")
    print(f"Variables categóricas: {categorical_vars}")
    print(f"Variables temporales: {temporal_vars}")

    info_dict = {
        "shape": df.shape,
        "numeric_vars": numeric_vars,
        "categorical_vars": categorical_vars,
        "temporal_vars": temporal_vars,
        "nulls": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum()
    }

    return df, info_dict


def get_date_range(df, date_column):
    """
    Obtiene el rango de fechas (mínimo y máximo) de una columna temporal.

    Parámetros
    ----------
    df : pandas.DataFrame
        Dataset con la columna temporal.
    date_column : str
        Nombre de la columna que contiene las fechas.

    Retorna
    -------
    tuple
        (fecha mínima, fecha máxima)
    """
    if date_column not in df.columns:
        raise ValueError(f"La columna '{date_column}' no existe en el DataFrame.")
    
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    min_date = df[date_column].min()
    max_date = df[date_column].max()

    print(f"\nRango de fechas en '{date_column}': {min_date.date()} → {max_date.date()}")
    return (min_date, max_date)
