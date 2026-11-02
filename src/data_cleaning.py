import pandas as pd


NUMERIC_COLUMNS = [
    "Age",
    "Maintenance_Cost",
    "Downtime",
    "Maintenance_Frequency",
    "Failure_Event_Count",
    "Maintenance_Class"
]


def clean_data(df):
    cleaned = df.copy()

    cleaned.columns = cleaned.columns.str.strip()

    text_columns = cleaned.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        cleaned[column] = (
            cleaned[column].astype("string").str.strip()
        )

    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(
                cleaned[column], errors="coerce"
            )

    cleaned["Purchase_Date"] = pd.to_datetime(
        cleaned["Purchase_Date"], errors="coerce"
    )

    cleaned["Maintenance_Cost_Negative_Flag"] = (
        cleaned["Maintenance_Cost"] < 0
    )

    cleaned["Maintenance_Cost_For_Analysis"] = (
        cleaned["Maintenance_Cost"].where(
            cleaned["Maintenance_Cost"] >= 0
        )
    )

    return cleaned


def summarize_cleaning(original, cleaned):
    original_columns = [
        str(column).strip()
        for column in original.columns
    ]

    summary = {
        "Rows before cleaning": len(original),
        "Rows after cleaning": len(cleaned),
        "Missing cells before cleaning": int(
            original.isna().sum().sum()
        ),
        "Missing cells in source fields after conversion": int(
            cleaned[original_columns].isna().sum().sum()
        ),
        "Duplicate rows": int(
            original.duplicated().sum()
        ),
        "Duplicate device IDs": int(
            cleaned["Device_ID"].duplicated().sum()
        ),
        "Invalid purchase dates": int(
            cleaned["Purchase_Date"].isna().sum()
        ),
        "Negative costs flagged": int(
            cleaned["Maintenance_Cost_Negative_Flag"].sum()
        )
    }

    return summary