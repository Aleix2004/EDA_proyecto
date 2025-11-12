from src.data_loader import load_and_inspect_data, get_date_range

def main():
    filepath = "data/raw/avocado.csv"

    df, info = load_and_inspect_data(filepath)

    get_date_range(df, "Date")

if __name__ == "__main__":
    main()
