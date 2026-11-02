from pathlib import Path
import pandas as pd

from src.data_cleaning import clean_data, summarize_cleaning
from src.exploratory_analysis import generate_visualizations


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR / "data" / "Medical_Device_Failure_dataset.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    original_df = pd.read_csv(DATA_PATH)

    print("--- INITIAL DATASET INSPECTION ---")
    print("Shape:", original_df.shape)
    print("\nColumns:")
    print(original_df.columns.tolist())
    print("\nData types:")
    print(original_df.dtypes)
    print("\nMissing values:")
    print(original_df.isnull().sum())

    cleaned_df = clean_data(original_df)

    print("\n--- CLEANING SUMMARY ---")

    report = summarize_cleaning(
        original_df, cleaned_df
    )

    for item, value in report.items():
        print(f"{item}: {value}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_path = (
        OUTPUT_DIR / "Medical_Device_Failure_cleaned.csv"
    )

    cleaned_df.to_csv(
        cleaned_path,
        index=False,
        date_format="%Y-%m-%d"
    )

    print("\nCleaned dataset saved:", cleaned_path)

    print("\n--- GENERATING VISUALIZATIONS ---")

    generate_visualizations(cleaned_df, CHART_DIR)

    print("\nData cleaning and visualization completed.")


if __name__ == "__main__":
    main()