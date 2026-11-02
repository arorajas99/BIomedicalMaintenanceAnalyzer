
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid")


def save_plot(fig, output_dir, filename):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / filename

    fig.tight_layout()
    fig.savefig(file_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Chart saved:", file_path)


def plot_device_type_counts(df, output_dir):
    counts = df["Device_Type"].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="barh", ax=ax)

    ax.set_title("Number of Records by Device Type")
    ax.set_xlabel("Number of records")
    ax.set_ylabel("Device type")

    save_plot(fig, output_dir, "01_device_type_counts.png")


def plot_failure_distribution(df, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))

    order = sorted(df["Failure_Event_Count"].dropna().unique())

    sns.countplot(
        data=df,
        x="Failure_Event_Count",
        order=order,
        ax=ax
    )

    ax.set_title("Distribution of Recorded Failure Events")
    ax.set_xlabel("Recorded failure-event count")
    ax.set_ylabel("Number of devices")

    save_plot(
        fig, output_dir, "02_failure_event_distribution.png"
    )


def plot_average_failures_by_type(df, output_dir):
    averages = (
        df.groupby("Device_Type")["Failure_Event_Count"]
        .mean()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    averages.plot(kind="barh", ax=ax)

    ax.set_title("Average Recorded Failure Events by Device Type")
    ax.set_xlabel("Average failure-event count")
    ax.set_ylabel("Device type")

    save_plot(
        fig, output_dir, "03_average_failures_by_type.png"
    )


def plot_maintenance_cost(df, output_dir):
    cost_data = df.dropna(
        subset=["Maintenance_Cost_For_Analysis"]
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=cost_data,
        x="Device_Type",
        y="Maintenance_Cost_For_Analysis",
        ax=ax
    )

    ax.set_title("Maintenance Cost by Device Type")
    ax.set_xlabel("Device type")
    ax.set_ylabel("Recorded maintenance cost (units unspecified)")
    ax.tick_params(axis="x", rotation=25)

    save_plot(
        fig, output_dir, "04_maintenance_cost_by_type.png"
    )


def plot_downtime(df, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=df,
        x="Device_Type",
        y="Downtime",
        ax=ax
    )

    ax.set_title("Downtime Distribution by Device Type")
    ax.set_xlabel("Device type")
    ax.set_ylabel("Recorded downtime (units unspecified)")
    ax.tick_params(axis="x", rotation=25)

    save_plot(fig, output_dir, "05_downtime_by_type.png")


def plot_correlation_heatmap(df, output_dir):
    columns = [
        "Age",
        "Maintenance_Cost_For_Analysis",
        "Downtime",
        "Maintenance_Frequency",
        "Failure_Event_Count"
    ]

    correlations = df[columns].corr()

    labels = {
        "Age": "Age",
        "Maintenance_Cost_For_Analysis": "Maintenance Cost",
        "Downtime": "Downtime",
        "Maintenance_Frequency": "Maintenance Frequency",
        "Failure_Event_Count": "Failure Events"
    }

    correlations = correlations.rename(
        index=labels, columns=labels
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        correlations,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax
    )

    ax.set_title("Correlation Between Numerical Variables")

    save_plot(fig, output_dir, "06_correlation_heatmap.png")


def create_device_summary(df):
    summary = (
        df.groupby("Device_Type")
        .agg(
            Device_Count=("Device_ID", "size"),
            Average_Failure_Events=("Failure_Event_Count", "mean"),
            Average_Downtime=("Downtime", "mean"),
            Average_Valid_Maintenance_Cost=(
                "Maintenance_Cost_For_Analysis", "mean"
            )
        )
        .round(2)
        .reset_index()
    )

    return summary


def generate_visualizations(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_device_type_counts(df, output_dir)
    plot_failure_distribution(df, output_dir)
    plot_average_failures_by_type(df, output_dir)
    plot_maintenance_cost(df, output_dir)
    plot_downtime(df, output_dir)
    plot_correlation_heatmap(df, output_dir)

    summary = create_device_summary(df)

    summary_path = output_dir.parent / "device_type_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n--- DEVICE TYPE SUMMARY ---")
    print(summary.to_string(index=False))
    print("\nSummary saved:", summary_path)
