# part1_exploratory_preprocessing.py

def run(context):
    """
    Exploratory Analysis & Preprocessing.

    - Loads the CSV (indexed by 'date'); uses context["csv_path"] if provided
    - Basic EDA (daily means, STL, correlations, distributions)
    - Cleans & feature-engineers (clip/interpolate/log/encode)
    - ACF/PACF diagnostics

    Returns: updates context with "df", "daily_df", "df_interpolated".
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    # ----- plotting style (no autosave patch here; main.py handles it) -----
    plt.style.use("ggplot")
    sns.set(rc={"figure.figsize": (15, 6)})

    # ----- load dataset -----
    csv_path = context.get("csv_path") 
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")

    # ----- info -----
    print("Shape:", df.shape)
    print("Time range:", df.index.min(), "to", df.index.max())
    try:
        print(df.head())
        df.info()
    except Exception:
        pass

    # ----- missing -----
    missing = df.isnull().sum()
    has_missing = missing[missing > 0]
    if not has_missing.empty:
        print("Columns with missing values:\n", has_missing)

    # ----- daily means -----
    daily_df = df.resample("D").mean(numeric_only=True)
    try:
        daily_df.plot(subplots=True, figsize=(15, 18), title="Daily Averages of Weather Variables")
        plt.tight_layout(); plt.show()
    except Exception as e:
        print("Daily averages plot skipped:", e)

    # ----- STL on daily T -----
    try:
        from statsmodels.tsa.seasonal import STL
        daily_temp = daily_df["T"].dropna()
        if not daily_temp.empty:
            stl = STL(daily_temp, period=365, robust=True)
            res = stl.fit()
            res.plot()
            plt.suptitle("STL Decomposition of Daily Temperature (Annual Seasonality)", fontsize=16)
            plt.show()
    except Exception as e:
        print("STL decomposition skipped:", e)

    # ----- correlation (hourly) -----
    try:
        corr_matrix = df.corr(numeric_only=True)
        import matplotlib.pyplot as _plt
        _plt.figure(figsize=(14, 10))
        import seaborn as _sns
        _sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        _plt.title("Correlation Matrix of Weather Variables (Hourly Data)", fontsize=16)
        _plt.xticks(rotation=45); _plt.yticks(rotation=0)
        _plt.tight_layout(); _plt.show()
    except Exception as e:
        print("Hourly correlation heatmap skipped:", e)

    # ----- correlations with wind dir sin/cos -----
    try:
        df_fe = df.copy()
        if "wd" in df_fe.columns:
            wd_rad = np.deg2rad(df_fe["wd"])
            df_fe["wd_sin"] = np.sin(wd_rad)
            df_fe["wd_cos"] = np.cos(wd_rad)
        cols = [c for c in df_fe.columns if c != "wd"]
        corr_fe = df_fe[cols].corr(numeric_only=True)
        import matplotlib.pyplot as _plt
        import seaborn as _sns
        _plt.figure(figsize=(14, 10))
        _sns.heatmap(corr_fe, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        _plt.title("Correlation Matrix (with wd_sin / wd_cos)", fontsize=16)
        _plt.xticks(rotation=45); _plt.yticks(rotation=0)
        _plt.tight_layout(); _plt.show()
    except Exception as e:
        print("wd_sin/wd_cos correlation heatmap skipped:", e)

    # ----- correlations on anomalies (x - daily mean) -----
    try:
        df_anom = df_fe.copy()
        daily_means = df_anom.resample("D").mean(numeric_only=True)
        daily_back = daily_means.reindex(df_anom.index, method="ffill")
        anoms = df_anom - daily_back
        if "wd" in anoms.columns:
            anoms = anoms.drop(columns=["wd"])
        corr_anoms = anoms.corr(numeric_only=True)
        import matplotlib.pyplot as _plt
        import seaborn as _sns
        _plt.figure(figsize=(14, 10))
        _sns.heatmap(corr_anoms, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        _plt.title("Correlation Matrix on Deseasonalized Anomalies (x - daily mean)", fontsize=16)
        _plt.xticks(rotation=45); _plt.yticks(rotation=0)
        _plt.tight_layout(); _plt.show()
    except Exception as e:
        print("Anomalies correlation heatmap skipped:", e)

    # ----- distributions (pre-cleaning) -----
    try:
        features_to_plot = ["T", "rho", "wv", "wd", "rain", "SWDR", "VPact"]
        for col in features_to_plot:
            if col in df.columns:
                import matplotlib.pyplot as _plt
                import seaborn as _sns
                _plt.figure(figsize=(10, 4))
                _sns.histplot(df[col].dropna(), kde=True, bins=50)
                _plt.title(f"Distribution of {col}")
                _plt.xlabel(col); _plt.ylabel("Frequency")
                _plt.tight_layout(); _plt.show()
    except Exception as e:
        print("Pre-cleaning distribution plots skipped:", e)

    # ----- cleaning & feature engineering -----
    df = df.sort_index()

    if "wv" in df.columns:
        df["wv"] = df["wv"].mask(df["wv"] < 0)
        df["wv"] = df["wv"].interpolate(method="time", limit_direction="both")

    if "rain" in df.columns:
        df["rain"] = df["rain"].clip(lower=0)

    if "SWDR" in df.columns:
        df["SWDR"] = df["SWDR"].clip(lower=0)
        df["SWDR_log"] = np.log1p(df["SWDR"])

    if "wd" in df.columns:
        wd_rad = np.deg2rad(df["wd"])
        df["wd_sin"] = np.sin(wd_rad)
        df["wd_cos"] = np.cos(wd_rad)

    if "wd_rad" in df.columns:
        df.drop(columns=["wd_rad"], inplace=True)

    df_interpolated = df.interpolate(method="time")
    print("Remaining missing values:", int(df_interpolated.isnull().sum().sum()))

    # ----- distributions (post-cleaning) -----
    try:
        features_to_plot = ["wv", "rain", "SWDR_log", "wd_sin", "wd_cos"]
        for col in features_to_plot:
            if col in df_interpolated.columns:
                import matplotlib.pyplot as _plt
                import seaborn as _sns
                _plt.figure(figsize=(10, 4))
                _sns.histplot(df_interpolated[col].dropna(), kde=True, bins=50)
                _plt.title(f"Distribution of {col} (post-cleaning)")
                _plt.xlabel(col); _plt.ylabel("Frequency")
                _plt.tight_layout(); _plt.show()

        if "wv" in df_interpolated.columns:
            print("wv min/max:", df_interpolated["wv"].min(), df_interpolated["wv"].max())
        if "rain" in df_interpolated.columns:
            print("rain min (should be >= 0):", df_interpolated["rain"].min())
        for c in ["wd_sin", "wd_cos"]:
            if c in df_interpolated.columns:
                print(f"{c} range: [{df_interpolated[c].min():.3f}, {df_interpolated[c].max():.3f}]")
    except Exception as e:
        print("Post-cleaning distribution plots skipped:", e)

    # ----- ACF / PACF diagnostics -----
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        if "T" in df_interpolated.columns:
            y_hourly = df_interpolated["T"].resample("h").mean()
            y_diag = y_hourly.dropna().values

            import matplotlib.pyplot as _plt
            _plt.figure(figsize=(14, 5))
            ax1 = _plt.subplot(1, 2, 1)
            plot_acf(y_diag, lags=72, ax=ax1, use_vlines=True)
            ax1.set_title("ACF (hourly T)")
            ax1.set_xlim(0, 72)

            ax2 = _plt.subplot(1, 2, 2)
            plot_pacf(y_diag, lags=72, ax=ax2, method="ywm", use_vlines=True)
            ax2.set_title("PACF (hourly T)")
            ax2.set_xlim(0, 72)

            _plt.tight_layout(); _plt.show()
    except Exception as e:
        print("ACF/PACF diagnostics skipped:", e)

    # ----- save to context -----
    context["df"] = df
    context["daily_df"] = daily_df
    context["df_interpolated"] = df_interpolated
    return context


if __name__ == "__main__":
    run({})
