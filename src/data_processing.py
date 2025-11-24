import pandas as pd
import os
from pandas.tseries.offsets import Week

def clean_data(save=True):
    """
    Limpia el dataset: elimina nulos, duplicados, normaliza datos y remueve outliers.
    
    Parámetros
    ----------
    save : bool
        Si True, guarda los datos limpios en data/processed/
    
    Retorna
    -------
    df : pandas.DataFrame
        DataFrame limpio
    """
    
    df = pd.read_csv("data/raw/avocado.csv")

    # Nulls and duplicates
    df = df.dropna()
    df = df.drop_duplicates()

    # Convertir Date a datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    # Convertir type y region a categóricos
    df['type'] = df['type'].astype('category')
    df['region'] = df['region'].astype('category')

    # Normalizar nombres de regiones
    df['region'] = df['region'].str.strip().str.capitalize().astype('category')

    # Detectar y remover outliers en AveragePrice usando IQR
    Q1 = df['AveragePrice'].quantile(0.25)
    Q3 = df['AveragePrice'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df['AveragePrice'] >= lower_bound) & (df['AveragePrice'] <= upper_bound)]

    # Guardar datos limpios
    if save:
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv("data/processed/avocado_clean.csv", index=False)
        print("✓ Datos limpios guardados en data/processed/avocado_clean.csv")

    return df


def add_features(df, save=True):
    """
    Añade características temporales y relacionadas con festividades (Feature Engineering).
    
    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame limpio con columna 'Date'
    save : bool
        Si True, guarda los datos transformados en data/processed/
    
    Retorna
    -------
    df : pandas.DataFrame
        DataFrame con nuevas características
    """

    df = df.copy()

    # --- Temporal Features ---
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    # Evitar fechas NaT
    df = df.dropna(subset=["Date"])

    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Quarter"] = df["Date"].dt.quarter
    df["MonthName"] = df["Date"].dt.month_name()

    # --- Season Mapping ---
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall"
    }
    df["Season"] = df["Month"].map(season_map)

    # --- Holiday Flags ---
    df["Is_SuperBowl"] = ((df["Month"] == 2) & (df["Week"].isin([5, 6]))).astype(int)
    df["Is_CincoDeMayo"] = ((df["Month"] == 5) & (df["Date"].dt.day.between(4, 6))).astype(int)

    # Thanksgiving = cuarto jueves de noviembre
    df["Is_Thanksgiving"] = 0
    nov = df[df["Month"] == 11].copy()
    for year in nov["Date"].dt.year.unique():
        # primer día de noviembre
        first_nov = pd.Timestamp(year=year, month=11, day=1)
        # cuarto jueves = primer jueves + 3 semanas
        thanksgiving = first_nov + Week(weekday=3) + pd.DateOffset(weeks=3)
        df.loc[df["Date"] == thanksgiving, "Is_Thanksgiving"] = 1

    # General holiday flag
    df["Is_Holiday"] = (
        df["Is_SuperBowl"] |
        df["Is_CincoDeMayo"] |
        df["Is_Thanksgiving"]
    ).astype(int)

    # Guardar datos transformados
    if save:
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv("data/processed/avocado_transformed.csv", index=False)
        print("✓ Datos transformados guardados en data/processed/avocado_transformed.csv")

    return df


if __name__ == "__main__":
    # Probar pipeline completo
    print("=== LIMPIEZA ===")
    df_limpio = clean_data(save=True)
    print(f"Filas después de limpieza: {df_limpio.shape}")
    
    print("\n=== TRANSFORMACIÓN ===")
    df_transformado = add_features(df_limpio, save=True)
    print(f"Filas después de transformación: {df_transformado.shape}")
    print(f"Nuevas columnas: {df_transformado.columns.tolist()}")