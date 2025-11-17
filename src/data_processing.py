import pandas as pd

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