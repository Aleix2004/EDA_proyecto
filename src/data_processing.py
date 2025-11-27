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
    
    NOTA: Este dataset tiene datos SEMANALES (domingos), no diarios.
    Por eso detectamos festividades usando ventanas de ±3 días.
    
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

    # --- Holiday Flags (con ventana de ±3 días) ---
    df["Is_SuperBowl"] = 0
    df["Is_CincoDeMayo"] = 0
    df["Is_Thanksgiving"] = 0
    
    for year in df["Date"].dt.year.unique():
        # Super Bowl: primer domingo de febrero
        feb_dates = pd.date_range(start=f'{year}-02-01', end=f'{year}-02-28', freq='D')
        sundays = [d for d in feb_dates if d.weekday() == 6]  # 6 = domingo
        if len(sundays) >= 1:
            superbowl = sundays[0]  # primer domingo de febrero
            # Marcar registros dentro de ±3 días
            mask = (df["Date"] >= superbowl - pd.Timedelta(days=3)) & \
                   (df["Date"] <= superbowl + pd.Timedelta(days=3))
            df.loc[mask, "Is_SuperBowl"] = 1
        
        # Cinco de Mayo: 5 de mayo ±3 días
        cinco_mayo = pd.Timestamp(year=year, month=5, day=5)
        mask = (df["Date"] >= cinco_mayo - pd.Timedelta(days=3)) & \
               (df["Date"] <= cinco_mayo + pd.Timedelta(days=3))
        df.loc[mask, "Is_CincoDeMayo"] = 1
        
        # Thanksgiving: cuarto jueves de noviembre ±3 días
        nov_dates = pd.date_range(start=f'{year}-11-01', end=f'{year}-11-30', freq='D')
        thursdays = [d for d in nov_dates if d.weekday() == 3]  # 3 = jueves
        if len(thursdays) >= 4:
            thanksgiving = thursdays[3]  # cuarto jueves
            mask = (df["Date"] >= thanksgiving - pd.Timedelta(days=3)) & \
                   (df["Date"] <= thanksgiving + pd.Timedelta(days=3))
            df.loc[mask, "Is_Thanksgiving"] = 1

    # General holiday flag
    df["Is_Holiday"] = (
        (df["Is_SuperBowl"] == 1) |
        (df["Is_CincoDeMayo"] == 1) |
        (df["Is_Thanksgiving"] == 1)
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
    
    # Verificar detección de festividades
    print("\n=== VERIFICACIÓN DE FESTIVIDADES ===")
    print(f"Super Bowl detectado: {df_transformado['Is_SuperBowl'].sum()} veces")
    print(f"Cinco de Mayo detectado: {df_transformado['Is_CincoDeMayo'].sum()} veces")
    print(f"Thanksgiving detectado: {df_transformado['Is_Thanksgiving'].sum()} veces")
    print(f"Total holidays: {df_transformado['Is_Holiday'].sum()} registros")
    
    # Mostrar algunos ejemplos
    print("\n=== EJEMPLOS DE FECHAS DETECTADAS ===")
    holidays = df_transformado[df_transformado['Is_Holiday'] == 1][['Date', 'Is_SuperBowl', 'Is_CincoDeMayo', 'Is_Thanksgiving']].head(10)
    print(holidays)


def transform_data(df):
    """
    Realiza transformaciones en el dataset:
    - Codifica la variable 'type' a numérico (type_encoded: 0/1).
    - Crea la variable 'Price_per_Volume'.
    - Calcula proporciones: Prop_4046, Prop_4225, Prop_4770.
    - Crea la variable 'Prop_Bags'.

    Parámetros
    ----------
    df : pandas.DataFrame
        Dataset original.

    Retorna
    -------
    df : pandas.DataFrame
        Dataset transformado con nuevas variables.
    """
    # Codificar 'type' a numérico
    df['type_encoded'] = df['type'].map({'conventional': 0, 'organic': 1})
    
    # Crear variable 'Price_per_Volume'
    df['Price_per_Volume'] = df['AveragePrice'] / df['Total Volume']
    
    # Crear proporciones
    df['Prop_4046'] = df['4046'] / df['Total Volume']
    df['Prop_4225'] = df['4225'] / df['Total Volume']
    df['Prop_4770'] = df['4770'] / df['Total Volume']
    
    # Crear variable 'Prop_Bags'
    df['Prop_Bags'] = df['Total Bags'] / df['Total Volume']
    
    return df

