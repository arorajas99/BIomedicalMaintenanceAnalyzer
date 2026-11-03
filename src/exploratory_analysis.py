from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Consistent styling for all exported figures.
sns.set_theme(
    style="whitegrid",
    context="notebook",
    palette="deep",
)

REQUIRED_COLUMNS = [
    "Device_Type",
    "Age",
    "Maintenance_Cost_For_Analysis",
    "Downtime",
    "Maintenance_Frequency",
    "Failure_Event_Count",
]


def _validate_data(df):
    """Check that the cleaned DataFrame contains the fields used by EDA."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Cannot generate visualizations. Missing columns: "
            + ", ".join(missing)
        )


def save_plot(fig, output_dir, filename):
    """Save a figure at report-friendly resolution and close it."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Chart saved:", file_path)


def _annotate_barh(ax, values, labels=None, value_format="{:.0f}"):
    """Add readable value labels to horizontal bars."""
    values = list(values)
    labels = labels if labels is not None else [None] * len(values)

    max_value = max(values) if values else 0
    offset = max_value * 0.012 if max_value else 0.02

    for index, (value, label) in enumerate(zip(values, labels)):
        text = value_format.format(value)
        if label is not None:
            text = f"{text}  ({label})"
        ax.text(value + offset, index, text, va="center", fontsize=9)


def plot_device_type_counts(df, output_dir):
    """Chart 1: device record counts and percentage contribution."""
    counts = df["Device_Type"].value_counts().sort_values(ascending=True)
    percentages = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(counts.index, counts.values, color=sns.color_palette("deep")[0])

    max_value = counts.max()
    ax.set_xlim(0, max_value * 1.22)

    for bar, count, percentage in zip(bars, counts.values, percentages.values):
        ax.text(
            bar.get_width() + max_value * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({percentage:.1f}%)",
            va="center",
            fontsize=9,
        )

    ax.set_title("Device Records by Category", weight="bold")
    ax.set_xlabel("Number of device records")
    ax.set_ylabel("Device category")
    save_plot(fig, output_dir, "01_device_type_counts.png")


def plot_failure_distribution(df, output_dir):
    """Chart 2: discrete distribution of recorded failure-event counts."""
    values = df["Failure_Event_Count"].dropna()
    counts = values.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(
        counts.index.astype(str),
        counts.values,
        color=sns.color_palette("deep")[1],
        edgecolor="white",
    )

    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    mean_value = values.mean()
    median_value = values.median()
    ax.set_title("Distribution of Recorded Failure Events", weight="bold")
    ax.set_xlabel("Recorded failure-event count")
    ax.set_ylabel("Number of device records")
    ax.text(
        0.98,
        0.95,
        f"Mean: {mean_value:.2f}   |   Median: {median_value:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )
    save_plot(fig, output_dir, "02_failure_event_distribution.png")


def plot_average_failures_by_type(df, output_dir):
    """Chart 3: mean failure count with bootstrap confidence intervals."""
    plot_data = df.dropna(subset=["Device_Type", "Failure_Event_Count"])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=plot_data,
        y="Device_Type",
        x="Failure_Event_Count",
        estimator="mean",
        errorbar=("ci", 95),
        orient="h",
        order=(
            plot_data.groupby("Device_Type")["Failure_Event_Count"]
            .mean().sort_values().index
        ),
        color=sns.color_palette("deep")[2],
        capsize=0.12,
        ax=ax,
    )

    summary = (
        plot_data.groupby("Device_Type")["Failure_Event_Count"]
        .agg(["mean", "count", "median"])
        .sort_values("mean")
    )

    for index, (device_type, row) in enumerate(summary.iterrows()):
        ax.text(
            row["mean"] + 0.04,
            index,
            f"mean={row['mean']:.2f} | n={int(row['count'])}",
            va="center",
            fontsize=8,
        )

    ax.set_title("Average Recorded Failure Events by Device Category", weight="bold")
    ax.set_xlabel("Mean recorded failure-event count (95% CI)")
    ax.set_ylabel("Device category")
    ax.set_xlim(left=0, right=max(summary["mean"]) * 1.55)
    save_plot(fig, output_dir, "03_average_failures_by_type.png")


def plot_maintenance_cost(df, output_dir):
    """Chart 4: valid (non-negative) maintenance costs by device category."""
    cost_data = df.dropna(
        subset=["Device_Type", "Maintenance_Cost_For_Analysis"]
    ).copy()

    fig, ax = plt.subplots(figsize=(11, 6))
    order = (
        cost_data.groupby("Device_Type")["Maintenance_Cost_For_Analysis"]
        .median()
        .sort_values()
        .index
    )

    sns.boxplot(
        data=cost_data,
        x="Device_Type",
        y="Maintenance_Cost_For_Analysis",
        order=order,
        showfliers=True,
        color=sns.color_palette("deep")[0],
        ax=ax,
    )

    ax.set_title("Maintenance Cost Distribution by Device Category", weight="bold")
    ax.set_xlabel("Device category")
    ax.set_ylabel("Recorded maintenance cost (currency unit unspecified)")
    ax.tick_params(axis="x", rotation=25)

    excluded = int((df["Maintenance_Cost"] < 0).sum()) if "Maintenance_Cost" in df else 0
    ax.text(
        0.99,
        0.98,
        f"Negative cost records excluded: {excluded}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    save_plot(fig, output_dir, "04_maintenance_cost_by_type.png")


def plot_downtime(df, output_dir):
    """Chart 5: downtime distribution by device category."""
    plot_data = df.dropna(subset=["Device_Type", "Downtime"])
    order = (
        plot_data.groupby("Device_Type")["Downtime"]
        .median()
        .sort_values()
        .index
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=plot_data,
        x="Device_Type",
        y="Downtime",
        order=order,
        showfliers=True,
        color=sns.color_palette("deep")[1],
        ax=ax,
    )

    ax.set_title("Downtime Distribution by Device Category", weight="bold")
    ax.set_xlabel("Device category")
    ax.set_ylabel("Recorded downtime (time unit unspecified)")
    ax.tick_params(axis="x", rotation=25)
    save_plot(fig, output_dir, "05_downtime_by_type.png")


def plot_correlation_heatmap(df, output_dir):
    """Chart 6: lower-triangle Pearson correlation heatmap."""
    columns = [
        "Age",
        "Maintenance_Cost_For_Analysis",
        "Downtime",
        "Maintenance_Frequency",
        "Failure_Event_Count",
    ]
    labels = [
        "Device age",
        "Maintenance cost",
        "Downtime",
        "Maintenance frequency",
        "Failure events",
    ]

    corr = df[columns].corr(method="pearson")
    corr.index = labels
    corr.columns = labels

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.6,
        cbar_kws={"label": "Pearson correlation coefficient"},
        ax=ax,
    )

    ax.set_title("Correlation Between Numerical Variables", weight="bold")
    save_plot(fig, output_dir, "06_correlation_heatmap.png")


def plot_age_vs_failures(df, output_dir):
    """Chart 7: relationship between device age and recorded failures."""
    data = df.dropna(subset=["Age", "Failure_Event_Count"])

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.regplot(
        data=data,
        x="Age",
        y="Failure_Event_Count",
        x_jitter=0.12,
        y_jitter=0.08,
        scatter_kws={"alpha": 0.25, "s": 22},
        line_kws={"color": "darkred", "linewidth": 2},
        lowess=True,
        ax=ax,
    )

    correlation = data[["Age", "Failure_Event_Count"]].corr().iloc[0, 1]
    ax.set_title("Device Age vs Recorded Failure Events", weight="bold")
    ax.set_xlabel("Device age (dataset units)")
    ax.set_ylabel("Recorded failure-event count")
    ax.text(
        0.98, 0.95, f"Pearson r = {correlation:.2f}",
        transform=ax.transAxes, ha="right", va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    save_plot(fig, output_dir, "07_age_vs_failures.png")


def plot_maintenance_frequency_vs_failures(df, output_dir):
    """Chart 8: failure-count distribution across maintenance-frequency levels."""
    data = df.dropna(subset=["Maintenance_Frequency", "Failure_Event_Count"])

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(
        data=data,
        x="Maintenance_Frequency",
        y="Failure_Event_Count",
        color=sns.color_palette("deep")[3],
        showfliers=True,
        ax=ax,
    )

    sns.stripplot(
        data=data.sample(min(len(data), 900), random_state=42),
        x="Maintenance_Frequency",
        y="Failure_Event_Count",
        color="black",
        alpha=0.22,
        size=2.5,
        jitter=0.22,
        ax=ax,
    )

    ax.set_title("Failure Events Across Maintenance Frequency Levels", weight="bold")
    ax.set_xlabel("Recorded maintenance frequency")
    ax.set_ylabel("Recorded failure-event count")
    save_plot(fig, output_dir, "08_maintenance_frequency_vs_failures.png")


def plot_device_performance_summary(df, output_dir):
    """Chart 9: standardized category-level metrics in a heatmap."""
    summary = (
        df.groupby("Device_Type")
        .agg(
            Mean_Failure_Events=("Failure_Event_Count", "mean"),
            Median_Downtime=("Downtime", "median"),
            Mean_Valid_Maintenance_Cost=(
                "Maintenance_Cost_For_Analysis", "mean"
            ),
        )
    )

    # Standardize each metric across device categories so different units
    # and scales can be compared without implying they share a common unit.
    standardized = (summary - summary.mean()) / summary.std(ddof=0)
    standardized = standardized.rename(
        columns={
            "Mean_Failure_Events": "Mean failures",
            "Median_Downtime": "Median downtime",
            "Mean_Valid_Maintenance_Cost": "Mean valid maintenance cost",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        standardized,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.6,
        cbar_kws={"label": "Standard deviations from category average"},
        ax=ax,
    )
    ax.set_title("Relative Device Category Performance", weight="bold")
    ax.set_xlabel("Analytical metric (standardized within each metric)")
    ax.set_ylabel("Device category")
    save_plot(fig, output_dir, "09_device_performance_summary.png")


def plot_cost_vs_downtime(df, output_dir):
    """Chart 10: relationship between valid maintenance cost and downtime."""
    data = df.dropna(
        subset=[
            "Maintenance_Cost_For_Analysis",
            "Downtime",
            "Device_Type",
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=data,
        x="Maintenance_Cost_For_Analysis",
        y="Downtime",
        hue="Device_Type",
        alpha=0.65,
        s=35,
        ax=ax,
    )

    correlation = data[
        ["Maintenance_Cost_For_Analysis", "Downtime"]
    ].corr().iloc[0, 1]

    ax.set_title("Maintenance Cost vs Recorded Downtime", weight="bold")
    ax.set_xlabel("Recorded maintenance cost (currency unit unspecified)")
    ax.set_ylabel("Recorded downtime (time unit unspecified)")
    ax.legend(title="Device category", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.text(
        0.02, 0.98, f"Pearson r = {correlation:.2f}",
        transform=ax.transAxes, ha="left", va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    save_plot(fig, output_dir, "10_cost_vs_downtime.png")


def create_device_summary(df):
    """Create a category-level table used in the report and insights."""
    summary = (
        df.groupby("Device_Type")
        .agg(
            Device_Count=("Device_ID", "size"),
            Average_Failure_Events=("Failure_Event_Count", "mean"),
            Median_Failure_Events=("Failure_Event_Count", "median"),
            Failure_Events_Std=("Failure_Event_Count", "std"),
            Average_Downtime=("Downtime", "mean"),
            Median_Downtime=("Downtime", "median"),
            Average_Valid_Maintenance_Cost=(
                "Maintenance_Cost_For_Analysis", "mean"
            ),
            Median_Valid_Maintenance_Cost=(
                "Maintenance_Cost_For_Analysis", "median"
            ),
        )
        .round(2)
        .reset_index()
    )
    return summary


def generate_automatic_insights(df):
    """
    Generate concise, data-derived observations.

    These are descriptive findings, not causal conclusions or predictions.
    """
    insights = []

    device_counts = df["Device_Type"].value_counts()
    largest_category = device_counts.idxmax()
    insights.append(
        f"{largest_category} has the largest representation in the dataset "
        f"({device_counts.max():,} records, "
        f"{device_counts.max() / len(df) * 100:.1f}% of all records)."
    )

    failure_by_type = (
        df.groupby("Device_Type")["Failure_Event_Count"].mean().dropna()
    )
    if not failure_by_type.empty:
        highest_failure_type = failure_by_type.idxmax()
        insights.append(
            f"{highest_failure_type} has the highest average recorded failure "
            f"count ({failure_by_type.max():.2f} events per record)."
        )

    downtime_by_type = df.groupby("Device_Type")["Downtime"].mean().dropna()
    if not downtime_by_type.empty:
        highest_downtime_type = downtime_by_type.idxmax()
        insights.append(
            f"{highest_downtime_type} has the highest average recorded downtime "
            f"({downtime_by_type.max():.2f}; time unit unspecified)."
        )

    valid_cost = df["Maintenance_Cost_For_Analysis"].dropna()
    if not valid_cost.empty:
        insights.append(
            f"The average valid maintenance cost is {valid_cost.mean():,.2f} "
            f"(currency unit unspecified); {len(valid_cost):,} records are "
            f"included in this calculation."
        )

    if "Maintenance_Cost_Negative_Flag" in df.columns:
        negative_count = int(df["Maintenance_Cost_Negative_Flag"].sum())
    elif "Maintenance_Cost" in df.columns:
        negative_count = int((df["Maintenance_Cost"] < 0).sum())
    else:
        negative_count = 0

    insights.append(
        f"{negative_count:,} records have negative maintenance costs. "
        "They are flagged and excluded from maintenance-cost analysis; "
        "the original records are retained."
    )

    age_failure = df[["Age", "Failure_Event_Count"]].dropna()
    if len(age_failure) >= 2:
        corr = age_failure.corr().iloc[0, 1]
        if abs(corr) < 0.10:
            strength = "negligible"
        elif abs(corr) < 0.30:
            strength = "weak"
        elif abs(corr) < 0.50:
            strength = "moderate"
        else:
            strength = "strong"
        direction = "positive" if corr > 0 else "negative" if corr < 0 else "zero"
        insights.append(
            f"Device age and recorded failure count have a {strength} "
            f"{direction} Pearson correlation (r = {corr:.2f}). This is an "
            "association, not evidence that age causes failures."
        )

    frequency_failure = df[
        ["Maintenance_Frequency", "Failure_Event_Count"]
    ].dropna()
    if len(frequency_failure) >= 2:
        corr = frequency_failure.corr().iloc[0, 1]
        strength = (
            "negligible" if abs(corr) < 0.10
            else "weak" if abs(corr) < 0.30
            else "moderate" if abs(corr) < 0.50
            else "strong"
        )
        insights.append(
            f"Maintenance frequency and recorded failure count have a "
            f"{strength} Pearson correlation (r = {corr:.2f}). The relationship "
            "is descriptive and does not establish whether maintenance changes "
            "failure rates."
        )

    cost_downtime = df[
        ["Maintenance_Cost_For_Analysis", "Downtime"]
    ].dropna()
    if len(cost_downtime) >= 2:
        corr = cost_downtime.corr().iloc[0, 1]
        strength = (
            "negligible" if abs(corr) < 0.10
            else "weak" if abs(corr) < 0.30
            else "moderate" if abs(corr) < 0.50
            else "strong"
        )
        direction = "positive" if corr > 0 else "negative" if corr < 0 else "zero"
        insights.append(
            f"Valid maintenance cost and recorded downtime have a {strength} "
            f"{direction} Pearson correlation (r = {corr:.2f}). Currency and "
            "downtime units are unspecified in the source dataset."
        )

    zero_failure_share = (
        (df["Failure_Event_Count"] == 0).mean() * 100
    )
    insights.append(
        f"{zero_failure_share:.1f}% of records have zero recorded failure events."
    )

    return insights


def generate_visualizations(df, output_dir):
    """Generate ten charts, summary CSV and automatic textual insights."""
    _validate_data(df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_functions = [
        plot_device_type_counts,
        plot_failure_distribution,
        plot_average_failures_by_type,
        plot_maintenance_cost,
        plot_downtime,
        plot_correlation_heatmap,
        plot_age_vs_failures,
        plot_maintenance_frequency_vs_failures,
        plot_device_performance_summary,
        plot_cost_vs_downtime,
    ]

    for chart_function in chart_functions:
        chart_function(df, output_dir)

    summary = create_device_summary(df)
    summary_path = output_dir.parent / "device_type_summary.csv"
    summary.to_csv(summary_path, index=False)

    insights = generate_automatic_insights(df)
    insights_path = output_dir.parent / "analysis_insights.txt"
    with insights_path.open("w", encoding="utf-8") as file:
        file.write("BIOMEDICAL DEVICE ANALYZER - AUTOMATIC OBSERVATIONS\n")
        file.write("=" * 58 + "\n\n")
        file.write(
            "These observations are descriptive. They do not establish "
            "causation or provide forecasts.\n\n"
        )
        for number, insight in enumerate(insights, start=1):
            file.write(f"{number}. {insight}\n\n")

    print("\n--- DEVICE TYPE SUMMARY ---")
    print(summary.to_string(index=False))
    print("\nSummary saved:", summary_path)

    print("\n--- AUTOMATIC OBSERVATIONS ---")
    for number, insight in enumerate(insights, start=1):
        print(f"{number}. {insight}")
    print("\nObservations saved:", insights_path)
