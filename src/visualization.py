import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_processing import clean_data


def plot_cleaning_graphics():
    """Genera graficos para mostrar el proceso de limpieza de datos."""
    output_dir = "outputs/02_data_cleaning/"
    os.makedirs(output_dir, exist_ok=True)

    # Cargar datos originales
    df_raw = pd.read_csv("data/raw/avocado.csv")

    # Aplicar limpieza
    df_clean = clean_data(save=False)

    # Boxplot antes de limpieza
    plt.figure(figsize=(8, 5))
    plt.boxplot(df_raw['AveragePrice'], vert=False)
    plt.title('AveragePrice - Antes de limpieza')
    plt.xlabel('AveragePrice')
    plt.savefig(os.path.join(output_dir, "boxplot_before.png"), bbox_inches='tight')
    plt.close()

    # Boxplot despues de limpieza
    plt.figure(figsize=(8, 5))
    plt.boxplot(df_clean['AveragePrice'], vert=False)
    plt.title('AveragePrice - Despues de limpieza')
    plt.xlabel('AveragePrice')
    plt.savefig(os.path.join(output_dir, "boxplot_after.png"), bbox_inches='tight')
    plt.close()

    # Deteccion de outliers
    Q1 = df_raw['AveragePrice'].quantile(0.25)
    Q3 = df_raw['AveragePrice'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_raw['is_outlier'] = (df_raw['AveragePrice'] < lower) | (df_raw['AveragePrice'] > upper)

    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(df_raw)), df_raw['AveragePrice'], s=10, label='Normal')
    plt.scatter(df_raw[df_raw['is_outlier']].index,
                df_raw[df_raw['is_outlier']]['AveragePrice'], s=10, c='red', label='Outliers')
    plt.title('Deteccion de Outliers - AveragePrice')
    plt.xlabel('Indice')
    plt.ylabel('AveragePrice')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "outliers_scatter.png"), bbox_inches='tight')
    plt.close()

    print("Visualizaciones de limpieza guardadas en:", output_dir)


def plot_temporal_features(df, output_path="outputs/04_feature_engineering/"):
    """
    Genera visualizaciones de las variables temporales creadas.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Distribucion por estaciones
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x="Season", order=["Winter", "Spring", "Summer", "Fall"])
    plt.title("Distribucion por Estacion")
    plt.ylabel("Count")
    plt.xlabel("Estacion")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "season_distribution.png"))
    plt.close()

    # Precio promedio por estacion
    plt.figure(figsize=(10, 6))
    season_order = ["Winter", "Spring", "Summer", "Fall"]
    season_prices = df.groupby("Season", observed=True)["AveragePrice"].mean().reindex(season_order)
    
    bars = plt.bar(season_order, season_prices, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    plt.title("Precio Promedio por Estacion", fontsize=14, fontweight='bold')
    plt.ylabel("Precio Promedio ($)")
    plt.xlabel("Estacion")
    
    # Añadir valores sobre las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "price_by_season.png"))
    plt.close()

    # Evolucion temporal del precio por tipo
    plt.figure(figsize=(14, 6))
    
    temporal_data = df.groupby(['Date', 'type'], observed=True)['AveragePrice'].mean().reset_index()
    
    for avocado_type in temporal_data['type'].unique():
        subset = temporal_data[temporal_data['type'] == avocado_type]
        plt.plot(subset['Date'], subset['AveragePrice'], 
                label=avocado_type.capitalize(), linewidth=2, alpha=0.7)
    
    plt.title("Evolucion Temporal del Precio por Tipo", fontsize=14, fontweight='bold')
    plt.xlabel("Fecha")
    plt.ylabel("Precio Promedio ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "temporal_price_evolution.png"))
    plt.close()

    print("Visualizaciones temporales guardadas en:", output_path)


def plot_transformed_features(df, output_dir="outputs/03_transformation"):
    """
    Genera histogramas de las variables derivadas.

    Parametros
    ----------
    df : pandas.DataFrame
        Dataset con las nuevas variables
    output_dir : str
        Carpeta donde se guardaran los graficos
    """
    os.makedirs(output_dir, exist_ok=True)

    # Variables derivadas principales
    derived_vars = ['Price_per_Volume', 'Prop_4046', 'Prop_4225', 'Prop_4770', 'Prop_Bags']
    
    for var in derived_vars:
        if var not in df.columns:
            continue
            
        # Histograma
        plt.figure(figsize=(8, 6))
        df[var].hist(bins=30, color='skyblue', edgecolor='black')
        plt.title(f"Distribucion de {var}")
        plt.xlabel(var)
        plt.ylabel("Frecuencia")
        plt.savefig(os.path.join(output_dir, f"{var}_histogram.png"))
        plt.close()
    
    print(f"Visualizaciones de transformaciones guardadas en: {output_dir}")


if __name__ == "__main__":
    # Ejecutar generacion de graficos
    plot_cleaning_graphics()
    
    from src.data_processing import add_features
    df_clean = clean_data(save=False)
    df_transformed = add_features(df_clean, save=False)
    plot_temporal_features(df_transformed)