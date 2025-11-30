import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def univariate_analysis(df, output_dir="outputs/05_EDA/univariate"):
    """
    Analiza la distribucion de cada variable de forma individual.
    
    Parametros
    ----------
    df : pandas.DataFrame
        Dataset con los datos transformados
    output_dir : str
        Carpeta donde se guardaran los graficos
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Variables numericas principales
    numeric_vars = ['AveragePrice', 'Total Volume', '4046', '4225', '4770', 'Total Bags']
    
    # Histogramas de variables numericas
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, var in enumerate(numeric_vars):
        df[var].hist(bins=30, ax=axes[i], alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Distribucion de {var}')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Frecuencia')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "numeric_distributions.png"), dpi=300)
    plt.close()
    
    # Distribucion de variables categoricas
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Tipo de aguacate
    df['type'].value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#3498db'])
    axes[0].set_title('Distribucion por Tipo')
    axes[0].set_xlabel('Tipo')
    axes[0].set_ylabel('Frecuencia')
    axes[0].tick_params(axis='x', rotation=0)
    
    # Top 10 regiones
    df['region'].value_counts().head(10).plot(kind='barh', ax=axes[1], color='coral')
    axes[1].set_title('Top 10 Regiones (mayor numero de registros)')
    axes[1].set_xlabel('Frecuencia')
    axes[1].set_ylabel('Region')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "categorical_distributions.png"), dpi=300)
    plt.close()
    
    print(f"Analisis univariante guardado en: {output_dir}")


def bivariate_analysis(df, output_dir="outputs/05_EDA/bivariate"):
    """
    Estudia las relaciones entre pares de variables.
    
    Parametros
    ----------
    df : pandas.DataFrame
        Dataset con los datos transformados
    output_dir : str
        Carpeta donde se guardaran los graficos
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Relaciones entre variables numericas
    numeric_pairs = [
        ('Total Volume', 'AveragePrice'),
        ('4046', 'AveragePrice'),
        ('Total Bags', 'AveragePrice'),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (x_var, y_var) in enumerate(numeric_pairs):
        sns.scatterplot(data=df, x=x_var, y=y_var, alpha=0.3, ax=axes[i])
        sns.regplot(data=df, x=x_var, y=y_var, scatter=False, 
                   color='red', ax=axes[i], line_kws={'linewidth': 2})
        axes[i].set_title(f'{x_var} vs {y_var}')
        axes[i].set_xlabel(x_var)
        axes[i].set_ylabel(y_var)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "numeric_vs_numeric.png"), dpi=300)
    plt.close()
    
    # Comparacion de precio por categorias
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Precio segun tipo de aguacate
    sns.boxplot(data=df, x='type', y='AveragePrice', ax=axes[0], hue='type', palette='Set2', legend=False)
    axes[0].set_title('Precio Promedio por Tipo')
    axes[0].set_xlabel('Tipo')
    axes[0].set_ylabel('Precio Promedio ($)')
    
    # Precio segun estacion del año
    if 'Season' in df.columns:
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        sns.boxplot(data=df, x='Season', y='AveragePrice', 
                   order=season_order, ax=axes[1], hue='Season', palette='coolwarm', legend=False)
        axes[1].set_title('Precio Promedio por Estacion')
        axes[1].set_xlabel('Estacion')
        axes[1].set_ylabel('Precio Promedio ($)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "categorical_vs_numeric.png"), dpi=300)
    plt.close()
    
    print(f"Analisis bivariante guardado en: {output_dir}")


def correlation_analysis(df, output_dir="outputs/05_EDA/correlation"):
    """
    Calcula y visualiza las correlaciones entre variables numericas.
    
    Parametros
    ----------
    df : pandas.DataFrame
        Dataset con los datos transformados
    output_dir : str
        Carpeta donde se guardaran los graficos
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Seleccionar variables numericas relevantes
    numeric_cols = ['AveragePrice', 'Total Volume', '4046', '4225', '4770', 
                   'Total Bags', 'type_encoded']
    
    # Filtrar solo las que existen en el dataset
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    # Calcular matriz de correlacion
    corr_matrix = df[available_cols].corr()
    
    # Visualizar con heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Matriz de Correlacion - Variables Numericas', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300)
    plt.close()
    
    print(f"Analisis de correlacion guardado en: {output_dir}")
    print("\nCorrelaciones con AveragePrice:")
    price_corr = corr_matrix['AveragePrice'].sort_values(ascending=False)
    print(price_corr)


def run_complete_eda(df):
    """
    Ejecuta el analisis exploratorio completo del dataset.
    
    Parametros
    ----------
    df : pandas.DataFrame
        Dataset preparado y transformado
    """
    print("\n" + "="*60)
    print("ANALISIS EXPLORATORIO DE DATOS (EDA) - COMPLETO")
    print("="*60)
    
    univariate_analysis(df)
    bivariate_analysis(df)
    correlation_analysis(df)
    
    print("\n" + "="*60)
    print("EDA COMPLETO FINALIZADO")
    print("="*60)


if __name__ == "__main__":
    # Cargar datos para ejecutar el analisis
    from src.data_processing import clean_data, add_features, transform_data
    
    print("Cargando y preparando datos...")
    df_clean = clean_data(save=False)
    df_transformed = add_features(df_clean, save=False)
    df_final = transform_data(df_transformed)
    
    # Ejecutar analisis completo
    run_complete_eda(df_final)