import pandas as pd
from pandas.tseries.offsets import Week

def add_features(df):
    """Add temporal and holiday-related engineered features."""

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

    return df
