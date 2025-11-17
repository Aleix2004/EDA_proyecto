import matplotlib.pyplot as plt
import os

def plot_transformed_features(df, output_dir="outputs/03_transformation"):
    """
    Genera histogramas y gráficos de comparación para las nuevas variables derivadas.

    Parámetros
    ----------
    df : pandas.DataFrame
        Dataset transformado con nuevas variables.
    output_dir : str
        Directorio donde se guardarán los gráficos.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Variables derivadas
    derived_vars = ['Price_per_Volume', 'Prop_4046', 'Prop_4225', 'Prop_4770', 'Prop_Bags']
    
    for var in derived_vars:
        # Histograma
        plt.figure(figsize=(8, 6))
        df[var].hist(bins=30, color='skyblue', edgecolor='black')
        plt.title(f"Distribución de {var}")
        plt.xlabel(var)
        plt.ylabel("Frecuencia")
        plt.savefig(os.path.join(output_dir, f"{var}_histogram.png"))
        plt.close()

        # Comparación con variable original (si aplica)
        if var == 'Price_per_Volume':
            plt.figure(figsize=(8, 6))
            plt.scatter(df['AveragePrice'], df[var], alpha=0.5)
            plt.title(f"Comparación: AveragePrice vs {var}")
            plt.xlabel("AveragePrice")
            plt.ylabel(var)
            plt.savefig(os.path.join(output_dir, f"AveragePrice_vs_{var}.png"))
            plt.close()