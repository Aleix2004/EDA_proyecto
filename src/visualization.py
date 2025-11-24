import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from data_processing import clean_data


def plot_cleaning_graphics():
    """Genera gráficos de diagnóstico del proceso de limpieza."""
    # Crear carpeta de salida
    output_dir = "outputs/02_data_cleaning/"
    os.makedirs(output_dir, exist_ok=True)

    # Cargar datos originales
    df_raw = pd.read_csv("data/raw/avocado.csv")

    # Limpiar datos
    df_clean = clean_data(save=False)  # No guardar aquí, solo para comparar

    # --- BOXPLOT ANTES ---
    plt.figure(figsize=(8, 5))
    plt.boxplot(df_raw['AveragePrice'], vert=False)
    plt.title('AveragePrice - Antes de limpieza')
    plt.xlabel('AveragePrice')
    plt.savefig(os.path.join(output_dir, "boxplot_before.png"), bbox_inches='tight')
    plt.close()

    # --- BOXPLOT DESPUÉS ---
    plt.figure(figsize=(8, 5))
    plt.boxplot(df_clean['AveragePrice'], vert=False)
    plt.title('AveragePrice - Después de limpieza')
    plt.xlabel('AveragePrice')
    plt.savefig(os.path.join(output_dir, "boxplot_after.png"), bbox_inches='tight')
    plt.close()

    # --- DETECCIÓN DE OUTLIERS (SCATTER) ---
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
    plt.title('Detección de Outliers - AveragePrice (Raw)')
    plt.xlabel('Índice')
    plt.ylabel('AveragePrice')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "outliers_scatter.png"), bbox_inches='tight')
    plt.close()

    print("✓ Visualizaciones de limpieza guardadas en:", output_dir)


def plot_temporal_features(df, output_path="outputs/04_feature_engineering/"):
    """Generate visualizations for temporal and holiday features."""

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # --- 1. Season Distribution ---
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x="Season", order=["Winter", "Spring", "Summer", "Fall"])
    plt.title("Season Distribution")
    plt.ylabel("Count")
    plt.xlabel("Season")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "season_distribution.png"))
    plt.close()

    # --- 2. Holiday Occurrences Over Time ---
    plt.figure(figsize=(14,6))
    plt.plot(df["Date"], df["Is_SuperBowl"], label="SuperBowl", marker="o", linestyle="None")
    plt.plot(df["Date"], df["Is_CincoDeMayo"], label="Cinco de Mayo", marker="o", linestyle="None")
    plt.plot(df["Date"], df["Is_Thanksgiving"], label="Thanksgiving", marker="o", linestyle="None")
    plt.legend()
    plt.title("Holiday Occurrences Over Time")
    plt.ylabel("Holiday Flag")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "holiday_occurrences.png"))
    plt.close()

    # --- 3. Holiday Calendar Heatmap ---
    df_calendar = df.copy()
    df_calendar["Year"] = df_calendar["Date"].dt.year
    df_calendar["Month"] = df_calendar["Date"].dt.month

    heatmap_data = df_calendar.pivot_table(
        index="Month",
        columns="Year",
        values="Is_Holiday",
        aggfunc="sum"
    ).reindex(index=range(1,13)).fillna(0)

    plt.figure(figsize=(12,8))
    sns.heatmap(heatmap_data, cmap="Reds", linewidths=0.5, annot=True, fmt=".0f",
                yticklabels=[calendar.month_name[m] for m in range(1,13)])
    plt.title("Holiday Frequency Calendar Heatmap")
    plt.ylabel("Month")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "holiday_calendar_heatmap.png"))
    plt.close()

    print("✓ Temporal feature visualizations saved in:", output_path)


if __name__ == "__main__":
    # Generar ambos tipos de visualizaciones
    plot_cleaning_graphics()
    
    # Para visualizaciones temporales necesitas datos transformados
    from data_processing import add_features
    df_clean = clean_data(save=False)
    df_transformed = add_features(df_clean, save=False)
    plot_temporal_features(df_transformed)