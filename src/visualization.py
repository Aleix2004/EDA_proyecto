import matplotlib.pyplot as plt
import seaborn as sns
import os
import calendar

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
    ).reindex(index=range(1,13))  # ordenar de enero a diciembre

    plt.figure(figsize=(12,8))
    sns.heatmap(heatmap_data, cmap="Reds", linewidths=0.5, annot=True, fmt="d",
                yticklabels=[calendar.month_name[m] for m in range(1,13)])
    plt.title("Holiday Frequency Calendar Heatmap")
    plt.ylabel("Month")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "holiday_calendar_heatmap.png"))
    plt.close()

    print("Temporal feature visualizations saved in:", output_path)
