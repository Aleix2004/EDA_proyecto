import pandas as pd

def clean_data():
    
    df = pd.read_csv("data/raw/avocado.csv")

    # Nulls and duplciates
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

    return df

if __name__ == "__main__":
    df_limpio = clean_data()
    print("Filas después de limpieza:", df_limpio.shape)
    print(df_limpio.head())
