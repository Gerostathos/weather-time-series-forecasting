# part2_ar_baseline.py

def run(context):
    """
    Baseline Forecast — Autoregressive (AR) Ridge ONLY.
    - Expects Part 1 to have run: context["df_interpolated"] preferred; will rebuild from context["df"] if needed.
    - Auto-saves all plots to plots/part2 before showing.
    - Stores metrics in context["results"].
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    # -------------------- autosave helper (patch plt.show) --------------------
    _plot_counter = {"i": 0}

    def _ensure_outdir(ctx):
        base = Path(ctx.get("plot_dir", "plots"))
        outdir = base / "part2"
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir

    def _patch_matplotlib_autosave(ctx, prefix):
        outdir = _ensure_outdir(ctx)
        _orig_show = plt.show
        def _auto_show(*args, **kwargs):
            for num in plt.get_fignums():
                fig = plt.figure(num)
                _plot_counter["i"] += 1
                fig.savefig(outdir / f"{prefix}_{_plot_counter['i']:03d}.png",
                            dpi=150, bbox_inches='tight')
            return _orig_show(*args, **kwargs)
        plt.show = _auto_show

    _patch_matplotlib_autosave(context, prefix="part2")

    # -------------------- sklearn + metrics --------------------
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    def rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    def mape(y_true, y_pred, eps=1e-3):
        y_true = np.asarray(y_true)
        denom = np.maximum(np.abs(y_true), eps)
        return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

    def smape(y_true, y_pred, eps=1e-3):
        y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
        denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
        return float(np.mean(np.abs(y_pred - y_true) / denom) * 100)

    def wmape(y_true, y_pred, eps=1e-3):
        y_true = np.asarray(y_true)
        num = np.sum(np.abs(y_true - y_pred))
        den = max(np.sum(np.abs(y_true)), eps)
        return float((num / den) * 100)

    # -------------------- supervised framing (DIRECT h-step) --------------------
    def make_supervised(y: pd.Series, p: int, h: int):
        """Predict y[t+h] from [y[t],...,y[t-p+1]]. Returns (X, target)."""
        df = pd.DataFrame({"y": y})
        for lag in range(p):
            df[f"lag_{lag}"] = df["y"].shift(lag)
        target = df["y"].shift(-h)
        X = df.drop(columns=["y"])
        out = pd.concat([X, target.rename("target")], axis=1).dropna()
        return out.drop(columns=["target"]), out["target"]

    def grid_search_p(y_train: pd.Series, y_val: pd.Series,
                      p_grid=(1,3,6,12,24,36,48,72), h=1, alpha=1.0):
        best = {"p": None, "mae": np.inf}
        scaler = StandardScaler()
        y_tr_s = pd.Series(scaler.fit_transform(y_train.to_frame())[:,0], index=y_train.index)
        y_va_s = pd.Series(scaler.transform(y_val.to_frame())[:,0], index=y_val.index)
        for p in p_grid:
            Xtr, ytr = make_supervised(y_tr_s, p=p, h=h)
            Xall, yall = make_supervised(pd.concat([y_tr_s, y_va_s]), p=p, h=h)
            mask = yall.index.isin(y_va_s.index)
            Xv, yv = Xall.loc[mask], yall.loc[mask]
            if len(Xtr) == 0 or len(Xv) == 0:
                continue
            model = Ridge(alpha=alpha)
            model.fit(Xtr, ytr)
            pred_val = model.predict(Xv)
            mae = mean_absolute_error(yv, pred_val)
            if mae < best["mae"]:
                best = {"p": p, "mae": mae}
        return best["p"], scaler

    # -------------------- bring/prepare data --------------------
    if "df_interpolated" in context:
        df_interpolated = context["df_interpolated"].copy()
    else:
        assert "df" in context, (
            "Neither 'df_interpolated' nor 'df' is defined in context. Run Part 1 first."
        )
        df_tmp = context["df"].copy()
        df_tmp['wv'] = df_tmp['wv'].mask(df_tmp['wv'] < 0)
        df_tmp['wv'] = df_tmp['wv'].interpolate(method='time')
        df_tmp['rain'] = df_tmp['rain'].clip(lower=0)
        wd_rad = np.deg2rad(df_tmp['wd'])
        df_tmp['wd_sin'] = np.sin(wd_rad)
        df_tmp['wd_cos'] = np.cos(wd_rad)
        df_tmp['SWDR_log'] = np.log1p(df_tmp['SWDR'])
        df_interpolated = df_tmp.interpolate(method='time')
        del df_tmp

    ts = df_interpolated['T'].copy()
    ts = ts[~ts.index.duplicated(keep='first')].sort_index().asfreq('h').dropna()

    TRAIN_RATIO = context.get('TRAIN_RATIO', 0.7)
    VAL_RATIO   = context.get('VAL_RATIO',   0.1)
    TEST_RATIO  = 1.0 - TRAIN_RATIO - VAL_RATIO
    assert 0 < TEST_RATIO < 0.5

    n = len(ts)
    i_train = int(n * TRAIN_RATIO)
    i_val   = int(n * (TRAIN_RATIO + VAL_RATIO))
    y_train = ts.iloc[:i_train]
    y_val   = ts.iloc[i_train:i_val]
    y_test  = ts.iloc[i_val:]

    HORIZONS = [1, 6, 24]
    P_GRID   = (1, 3, 6, 12, 24, 36, 48, 72)
    ALPHA    = 1.0

    # -------------------- AR baseline --------------------
    results = dict(context.get("results", {}))
    rows = []

    for h in HORIZONS:
        p_opt, scaler = grid_search_p(y_train, y_val, p_grid=P_GRID, h=h, alpha=ALPHA)
        y_trv = pd.concat([y_train, y_val])
        y_trv_s = pd.Series(scaler.fit_transform(y_trv.to_frame())[:,0], index=y_trv.index)
        Xtrv, ytrv = make_supervised(y_trv_s, p=p_opt, h=h)
        model = Ridge(alpha=ALPHA).fit(Xtrv, ytrv)

        y_test_s = pd.Series(scaler.transform(y_test.to_frame())[:,0], index=y_test.index)
        y_all_s  = pd.concat([y_trv_s, y_test_s])
        X_all, y_all_target = make_supervised(y_all_s, p=p_opt, h=h)
        mask = y_all_target.index.isin(y_test.index)
        X_test_h, y_test_h_s = X_all.loc[mask], y_all_target.loc[mask]

        y_pred_h_s = model.predict(X_test_h)
        y_true_h = pd.Series(
            scaler.inverse_transform(y_test_h_s.to_frame())[:, 0],
            index=y_test_h_s.index
        )
        y_pred_h = pd.Series(
            scaler.inverse_transform(y_pred_h_s.reshape(-1,1)).ravel(),
            index=y_test_h_s.index
        )

        rmse_v   = rmse(y_true_h, y_pred_h)
        mae_v    = float(mean_absolute_error(y_true_h, y_pred_h))
        mape_r   = mape(y_true_h, y_pred_h)
        smape_v  = smape(y_true_h, y_pred_h)
        wmape_v  = wmape(y_true_h, y_pred_h)

        rows.append({
            "Horizon": f"{h}-hour",
            "AR_p": p_opt,
            "AR_RMSE": rmse_v,
            "AR_MAE": mae_v,
            "AR_MAPE": mape_r,
            "AR_sMAPE": smape_v,
            "AR_WMAPE": wmape_v,
        })

        hk = f"{h}-hour"
        results[hk] = {
            "Model": "AR_Ridge",
            "p": int(p_opt),
            "RMSE": float(rmse_v),
            "MAE": float(mae_v),
            "MAPE_robust": float(mape_r),
            "SMAPE": float(smape_v),
            "WMAPE": float(wmape_v),
            "Actual": y_true_h.values.tolist(),
            "Forecast": y_pred_h.values.tolist(),
            "Index": y_true_h.index.astype(str).tolist(),
        }

    ar_results_df = pd.DataFrame(rows)
    try:
        from IPython.display import display
        display(ar_results_df)
    except Exception:
        print(ar_results_df)

    # aligned plots per horizon
    def plot_horizon_forecast(y_true, y_pred, title):
        plt.figure(figsize=(10,3))
        plt.plot(y_true.index, y_true.values, label="True")
        plt.plot(y_pred.index, y_pred.values, label="AR forecast")
        plt.title(title); plt.legend(); plt.tight_layout(); plt.show()

    for h in HORIZONS:
        hk = f"{h}-hour"
        yt = pd.Series(results[hk]["Actual"], index=pd.to_datetime(results[hk]["Index"]))
        yp = pd.Series(results[hk]["Forecast"], index=pd.to_datetime(results[hk]["Index"]))
        plot_horizon_forecast(yt, yp, f"AR direct — {h}-hour horizon (p={results[hk]['p']})")

    # -------------------- save back to context --------------------
    context["results"] = results
    context["ar_results_df"] = locals().get("ar_results_df")
    return context


if __name__ == "__main__":
    run({})
