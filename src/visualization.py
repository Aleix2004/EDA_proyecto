import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def ensure_output_dir(path):
    """Create the directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def plot_missing_values_heatmap(df, output_path):
    """Generate a heatmap showing missing values."""
    if df.isnull().sum().sum() == 0:
        print("No heatmap generated: dataset contains no missing values.")
        return

    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "heatmap_missing_values.png"))
    plt.close()
    print("Missing values heatmap saved.")


def plot_categorical_counts(df, col, output_path, top_n=None):
    """
    Plot bar charts of categorical variables.
    Replaces PLU values with descriptive names.
    """
    if col not in df.columns:
        print(f"Column '{col}' not found. Skipping.")
        return

    df[col] = df[col].astype(str).str.strip().str.lower()

    # PLU → descriptive name
    plu_map = {
        "4046": "PLU-4046 Hass",
        "4225": "PLU-4225 Fuerte",
        "4770": "PLU-4770 Bacon"
    }
    df[col] = df[col].replace(plu_map)

    counts = df[col].value_counts()

    if top_n is not None:
        counts = counts.head(top_n)

    plt.figure(figsize=(12, 6))
    counts.plot(kind="bar")
    plt.title(f"Value Counts for '{col}'")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = f"value_counts_{col}.png"
    plt.savefig(os.path.join(output_path, filename))
    plt.close()
    print(f"Categorical plot for '{col}' saved.")


def plot_time_series(df, date_col, value_col, freq='M', output_path="outputs/01_initial_exploration/"):
    """
    Plot time series grouped by frequency (Month, Quarter, Year).
    """

    if date_col not in df.columns or value_col not in df.columns:
        print(f"Columns '{date_col}' or '{value_col}' not found. Skipping time series.")
        return

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df["period"] = df[date_col].dt.to_period(freq)

    grouped = df.groupby("period")[value_col].mean().reset_index()

    # Mapping for clean month/quarter/year titles
    freq_map = {
        "M": "Month",
        "Q": "Quarter",
        "Y": "Year"
    }
    freq_title = freq_map.get(freq, freq)

    # PLU → descriptive title
    plu_titles = {
        "4046": "PLU-4046",
        "4225": "PLU-4225",
        "4770": "PLU-4770"
    }

    title_value = plu_titles.get(value_col, value_col)

    plt.figure(figsize=(12, 6))
    plt.bar(grouped["period"].astype(str), grouped[value_col])
    plt.xticks(rotation=45)
    plt.title(f"{title_value} average per {freq_title}")
    plt.xlabel(freq_title)
    plt.ylabel(value_col)

    # 🔥 Disable scientific notation (removes the "1e6" label)
    plt.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()

    filename = os.path.join(output_path, f"{value_col}_{date_col}_{freq}.png")
    plt.savefig(filename)
    plt.close()

    print(f"Time series plot saved: {filename}")


def plot_initial_exploration(df, categorical_vars, output_folder="outputs/01_initial_exploration/"):
    """
    Generate initial dataset visualizations:
      - Missing values heatmap
      - Categorical variable bar charts
      - Top 10 regions
      - Monthly time series for numeric columns
    """
    ensure_output_dir(output_folder)
    print(f"Saving visualizations in: {output_folder}")

    plot_missing_values_heatmap(df, output_folder)

    for col in categorical_vars:
        plot_categorical_counts(df, col, output_folder)

    if "region" in df.columns:
        plot_categorical_counts(df, "region", output_folder, top_n=10)

    if "Date" in df.columns:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        for value_col in numeric_cols:
            plot_time_series(df, "Date", value_col, freq="M", output_path=output_folder)

    print("All visualizations generated successfully.")
