# Dataset Instructions

This folder is reserved for the weather time-series dataset used by this project.

The dataset file is **not committed to GitHub**. To run the project locally, download the **Weather Long-term Time Series Forecasting** dataset from Kaggle:

https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting

After downloading the dataset, place the required CSV file in this folder and name it:

```text
cleaned_weather.csv
```

## Expected Local Structure

```text
data/
├── README.md
└── cleaned_weather.csv
```

Only `README.md` should be committed to GitHub. The CSV file should remain local.

## Expected Dataset Format

The project expects a CSV file with a datetime column named:

```text
date
```

and weather-related numerical variables such as:

```text
p, T, Tpot, Tdew, rh, VPmax, VPact, VPdef, sh, H2OC, rho,
wv, max. wv, wd, rain, raining, SWDR, PAR, max. PAR, Tlog
```

The main forecasting target used in the experiments is temperature:

```text
T
```

## How the Dataset Is Used

The pipeline reads the CSV file, parses the `date` column as a datetime index, performs preprocessing and exploratory analysis, and then evaluates forecasting models on hourly temperature prediction horizons.

The main script expects the dataset at:

```text
data/cleaned_weather.csv
```

when running the project from the repository root folder.

Example command:

```bash
python src/main.py --csv data/cleaned_weather.csv --plot-dir plots --report-dir results
```

By default, the implemented forecasting horizons are:

```text
1 hour, 6 hours, 24 hours
```

## Notes

- The dataset is intentionally excluded from this repository to keep the GitHub project lightweight and to avoid redistributing third-party data.
- If the downloaded Kaggle file has a different filename, rename the relevant CSV file to `cleaned_weather.csv`.
- Generated plots and comparison tables are included separately in the `plots/` and `results/` folders.
