# Biomedical Device Analyzer

A Python-based data analysis project for exploring medical-device
records, maintenance costs, downtime, maintenance frequency, and
recorded failure events. The project cleans the source dataset,
generates statistical summaries, produces visualizations, and writes
automatically generated descriptive observations.

## Project Overview

Medical equipment maintenance and reliability can be examined using
historical device records. This project analyzes a dataset of medical
devices to identify patterns across device categories and numerical
variables.

The analysis is **descriptive and exploratory**. It does not currently
train a predictive model, forecast failures, or establish causal
relationships.

## Features

-   Loads and inspects a medical-device CSV dataset.
-   Standardizes column names and trims whitespace from text fields.
-   Converts numerical fields and purchase dates to appropriate data
    types.
-   Flags negative maintenance-cost records and excludes them from cost
    calculations without deleting the original records.
-   Generates ten analytical charts.
-   Produces a device-category summary CSV.
-   Generates a text file containing automatic, data-derived
    observations.
-   Includes an optional Tkinter desktop dashboard for interactive
    dataset inspection and filtering.

## Dataset

The project expects the dataset to be named:

`Medical_Device_Failure_dataset.csv`

Place it inside the `data/` directory.

The dataset fields used by the project include:

  Field                     Description
  ------------------------- ---------------------------------------------
  `Device_ID`               Device record identifier
  `Device_Type`             Medical-device category
  `Purchase_Date`           Recorded purchase date
  `Age`                     Device age value as supplied by the dataset
  `Manufacturer`            Device manufacturer
  `Model`                   Device model
  `Country`                 Country field in the source data
  `Maintenance_Cost`        Recorded maintenance cost
  `Downtime`                Recorded downtime
  `Maintenance_Frequency`   Recorded maintenance frequency
  `Failure_Event_Count`     Recorded number of failure events
  `Maintenance_Class`       Maintenance classification field
  `Maintenance_Report`      Maintenance report text

**Data interpretation note:** The source CSV does not specify the
currency for `Maintenance_Cost` or the time unit for `Downtime`. The
project therefore reports these values without assigning units. Negative
maintenance costs are flagged and excluded from maintenance-cost
analysis; source rows are retained.

## Technology Stack

-   **Python** --- project implementation
-   **Pandas** --- data loading, cleaning, transformation, and
    aggregation
-   **NumPy** --- numerical operations used in analysis and chart
    generation
-   **Matplotlib** --- chart creation
-   **Seaborn** --- statistical visualizations
-   **Tkinter** --- optional desktop dashboard interface

## Project Structure

Arrange the project files as follows:

``` text
Biomedical-Device-Analyzer/
│
├── data/
│   └── Medical_Device_Failure_dataset.csv
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py
│   └── exploratory_analysis.py
│
├── outputs/
│   ├── charts/
│   ├── Medical_Device_Failure_cleaned.csv
│   ├── device_type_summary.csv
│   └── analysis_insights.txt
│
├── main.py
├── dashboard.py
├── requirements.txt
└── README.md
```

`outputs/` and its contents are generated when the analysis is run. The
`__init__.py` file may be empty; it marks `src` as a Python package.

## Installation

### 1. Clone or download the project

``` bash
git clone <your-repository-url>
cd Biomedical-Device-Analyzer
```

If you downloaded the project as a ZIP file, extract it and open a
terminal in the project directory.

### 2. Create a virtual environment (recommended)

``` bash
python -m venv .venv
```

Activate it:

**Windows (Command Prompt):**

``` bat
.venv\Scripts\activate
```

**Windows (PowerShell):**

``` powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

Install the libraries used by the analysis and dashboard:

``` bash
pip install pandas numpy matplotlib seaborn
```

Alternatively, if a `requirements.txt` file is included:

``` bash
pip install -r requirements.txt
```

## Running the Analysis

1.  Confirm that the dataset is located at
    `data/Medical_Device_Failure_dataset.csv`.
2.  Confirm that `main.py`, `src/data_cleaning.py`, and
    `src/exploratory_analysis.py` are in the locations shown in the
    project structure.
3.  From the project root, run:

``` bash
python main.py
```

The program reads the dataset, prints an initial inspection and cleaning
summary, saves a cleaned CSV, and generates charts and analysis outputs.

## Generated Outputs

After a successful run, the `outputs/` directory contains:

  --------------------------------------------------------------------------
  Output                                 Description
  -------------------------------------- -----------------------------------
  `Medical_Device_Failure_cleaned.csv`   Cleaned dataset with additional
                                         maintenance-cost analysis fields

  `device_type_summary.csv`              Aggregated metrics for each device
                                         category

  `analysis_insights.txt`                Automatically generated descriptive
                                         observations

  `charts/`                              Ten PNG visualizations
  --------------------------------------------------------------------------

### Visualizations

1.  Device record counts by category, including percentage contribution.
2.  Distribution of recorded failure-event counts.
3.  Average failure events by device category, with 95% confidence
    intervals.
4.  Maintenance-cost distribution by device category.
5.  Downtime distribution by device category.
6.  Pearson correlation heatmap for selected numerical variables.
7.  Device age versus recorded failure events.
8.  Failure-event distribution across maintenance-frequency levels.
9.  Relative device-category performance heatmap using standardized
    metrics.
10. Maintenance cost versus downtime, grouped by device category.

## Optional Desktop Dashboard

The project also includes `dashboard.py`, a Tkinter-based interface for
interactively inspecting a CSV dataset.

Run it from the project root:

``` bash
python dashboard.py
```

The dashboard provides: - CSV file selection. - Device-category
filtering. - Summary indicators for record count, device categories,
average valid maintenance cost, average downtime, and recorded failure
events. - A visual-analysis tab and a dataset-preview tab.

The dashboard can load a CSV selected through the file picker. It first
attempts to load `data/Medical_Device_Failure_dataset.csv` automatically
if that file exists.

Tkinter is included with many standard Python installations. On some
Linux distributions, it may need to be installed separately through the
operating system package manager.

## Data Cleaning Method

The cleaning module performs the following operations:

1.  Creates a copy of the input DataFrame so the original object is not
    modified.
2.  Removes leading and trailing whitespace from column names and text
    values.
3.  Converts selected numerical columns using safe numeric conversion;
    unparseable values become missing values.
4.  Converts `Purchase_Date` to a datetime value; invalid dates become
    missing values.
5.  Creates `Maintenance_Cost_Negative_Flag` to identify negative
    maintenance-cost values.
6.  Creates `Maintenance_Cost_For_Analysis`, retaining non-negative
    costs and setting negative costs to missing for cost analysis.

The project retains the original rows. It does not silently replace
negative maintenance costs with zero.

## Statistical Interpretation and Limitations

-   Correlation measures association, not causation.
-   The automatic observations describe the supplied dataset; they
    should not be interpreted as clinical conclusions or
    equipment-safety assessments.
-   The reliability of the results depends on the completeness,
    accuracy, and representativeness of the source data.
-   The source dataset does not define the units for maintenance cost
    and downtime.
-   The current implementation is not a predictive maintenance system.
    Forecasting, failure prediction, and formal hypothesis testing would
    require additional methods and validation.

## Reproducibility

For consistent results: - Use the same source dataset version. - Keep
the expected dataset filename and folder structure. - Run `main.py` from
the project root. - Retain the generated outputs alongside the dataset
version used to produce them.

## Contributors

-   **Jas Arora** --- Roll No. `16014025003`
-   **Yash Garg** --- Roll No. `16014025021`

**Department:** Electronics and Computer Engineering (EXCP)\
**Institution:** K. J. Somaiya School of Engineering, Somaiya Vidyavihar
University\
**Academic Year:** 2026--27

## Repository

GitHub: `[Add the public repository URL here]`

## License

No license has been specified. Add a `LICENSE` file before presenting
this project as open source.
