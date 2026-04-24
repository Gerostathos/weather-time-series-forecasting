# part5_transformer.py
def run(context):
    """
    Part 5 — Transformer forecaster + (optional) Optuna tuning.
    - Reads cleaned hourly series from context["df_interpolated"].
    - Auto-saves ALL figures to plots/part5 as PNGs AND calls plt.show().
    - Writes results and best params back into context.
    """
    import math
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # -------------------- autosave helper --------------------
    _plot_counter = {"i": 0}

    def _ensure_outdir(ctx):
        base = Path(ctx.get("plot_dir", "plots"))
        outdir = base / "part5"
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir

    def _patch_matplotlib_autosave(ctx, prefix="part5"):
        outdir = _ensure_outdir(ctx)
        _orig_show = plt.show
        def _auto_show(*args, **kwargs):
            # save every open figure before showing
            for num in plt.get_fignums():
                fig = plt.figure(num)
                _plot_counter["i"] += 1
                fig.savefig(outdir / f"{prefix}_{_plot_counter['i']:03d}.png",
                            dpi=150, bbox_inches='tight')
            return _orig_show(*args, **kwargs)
        plt.show = _auto_show

    _patch_matplotlib_autosave(context, prefix="part5")

    # -------------------- inputs & defaults --------------------
    if "df_interpolated" not in context:
        raise RuntimeError("df_interpolated not found in context. Run Part 1 first.")
    df_interpolated = context["df_interpolated"].copy()

    ts = df_interpolated['T'].copy()
    ts = ts[~ts.index.duplicated(keep='first')].sort_index().asfreq('h').dropna()

    DEVICE = context.get("DEVICE") or (torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu")))
    print(f"[Transformer] Using DEVICE: {DEVICE}")

    SEQ_LENGTH = int(context.get("SEQ_LENGTH", 48))
    HORIZONS   = list(context.get("HORIZONS", [1, 6, 24]))
    TRAIN_RATIO = float(context.get("TRAIN_RATIO", 0.70))
    VAL_RATIO   = float(context.get("VAL_RATIO",   0.20))
    TEST_RATIO  = 1.0 - TRAIN_RATIO - VAL_RATIO
    assert 0 < TEST_RATIO < 0.5

    BATCH_SIZE = int(context.get("BATCH_SIZE", 128))
    EPOCHS = int(context.get("EPOCHS", 50))
    LEARNING_RATE = float(context.get("LEARNING_RATE", 1e-3))
    EARLY_STOPPING_PATIENCE = int(context.get("EARLY_STOPPING_PATIENCE", 5))

    # -------------------- helpers --------------------
    def create_supervised_sequences(series, seq_length: int, horizon: int,
                                    return_index: bool=False, add_channel: bool=True):
        s = pd.Series(series).astype(float).dropna()
        vals = s.to_numpy(); X, y, t_idx = [], [], []
        limit = len(vals) - seq_length - horizon + 1
        for i in range(limit):
            X.append(vals[i:i+seq_length])
            y.append(vals[i+seq_length+horizon-1])
            if return_index:
                t_idx.append(s.index[i+seq_length+horizon-1])
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if add_channel: X = X[..., None]
        if return_index: return X, y, np.asarray(t_idx, dtype=object)
        return X, y

    def train_model(model, train_loader, num_epochs=20, lr=1e-3, device=DEVICE):
        model = model.to(device)
        lossf = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(1, num_epochs+1):
            model.train(); run=0.0; n_s=0
            for xb, yb in train_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                opt.zero_grad(); preds = model(xb); loss = lossf(preds, yb); loss.backward(); opt.step()
                bs = xb.size(0); run += loss.item()*bs; n_s += bs
            print(f"Epoch {ep:02d}/{num_epochs} - Train MSE: {run/max(1,n_s):.6f}")
        return model, {}

    @torch.no_grad()
    def evaluate_model_robust(model, loader, device=DEVICE, mape_floor=1.0, eps=1e-6):
        model.eval().to(device); preds, trues = [], []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            yhat = model(xb)
            preds.append(yhat.detach().cpu().numpy()); trues.append(yb.detach().cpu().numpy())
        y_pred = np.concatenate(preds).astype('float64')
        y_true = np.concatenate(trues).astype('float64')
        abs_err = np.abs(y_pred - y_true)
        mae  = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
        denom = np.maximum(np.abs(y_true), float(mape_floor))
        mape_r = float(np.mean(abs_err/denom) * 100.0)
        smape  = float(np.mean(2.0*abs_err/(np.abs(y_true)+np.abs(y_pred)+eps)) * 100.0)
        wmape  = float(np.sum(abs_err)/(np.sum(np.abs(y_true))+eps) * 100.0)
        return {"MAE":mae, "RMSE":rmse, "MAPE_robust":mape_r, "SMAPE":smape, "WMAPE":wmape}, y_true, y_pred

    def plot_forecast_from_results(results_dict, horizon_key: str, model_label: str = "Transformer"):
        import matplotlib.pyplot as _plt, numpy as _np, pandas as _pd
        if horizon_key not in results_dict: return
        res = results_dict[horizon_key]
        y_true = _np.asarray(res["Actual"]); y_pred = _np.asarray(res["Forecast"]); t = res.get("Time", None)
        mae = res.get("MAE"); rmse = res.get("RMSE"); mape = res.get("MAPE"); maper = res.get("MAPE_robust")
        _plt.figure(figsize=(12,4))
        if t is not None:
            t = _pd.to_datetime(t)
            _plt.plot(t, y_true, label="Actual", linewidth=2)
            _plt.plot(t, y_pred, label=model_label, linestyle="--")
            _plt.axvline(t[0], ls="--", lw=1, alpha=0.6, label="Forecast start")
            _plt.xlabel("Time"); _plt.xticks(rotation=30)
        else:
            _plt.plot(y_true, label="Actual", linewidth=2)
            _plt.plot(y_pred, label=model_label, linestyle="--")
            _plt.xlabel("Sample index")
        title = [f"{horizon_key} — {model_label}"]
        if mae is not None and rmse is not None: title.append(f"MAE={mae:.3f}, RMSE={rmse:.3f}")
        if mape is not None: title.append(f"MAPE={mape:.2f}%")
        if maper is not None: title.append(f"MAPE₍robust₎={maper:.2f}%")
        _plt.title("  |  ".join(title)); _plt.ylabel("Temperature (°C)")
        _plt.grid(True); _plt.legend(); _plt.tight_layout(); _plt.show()

    # -------------------- Transformer components --------------------
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 1000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position*div_term)
            pe[:, 1::2] = torch.cos(position*div_term)
            self.register_buffer("pe", pe)
        def forward(self, x):
            T = x.size(1)
            return x + self.pe[:T].unsqueeze(0)

    class TimeSeriesTransformer(nn.Module):
        def __init__(self, input_size: int = 1, d_model: int = 128, nhead: int = 4,
                     num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1,
                     readout: str = "last"):
            super().__init__()
            assert d_model % nhead == 0, "d_model must be divisible by nhead"
            self.readout = readout
            self.in_proj = nn.Linear(input_size, d_model)
            self.pos_enc = PositionalEncoding(d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, activation="gelu"
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.out = nn.Linear(d_model, 1)
        def forward(self, x):
            h = self.in_proj(x)
            h = self.pos_enc(h)
            h = self.encoder(h)
            h_read = h.mean(dim=1) if self.readout == "mean" else h[:, -1, :]
            y = self.out(h_read)
            return y.squeeze(-1)

    # -------------------- Baseline Transformer runs --------------------
    from sklearn.preprocessing import StandardScaler
    results_tf = {}

    for h in HORIZONS:
        Xh, yh, t_idx_h = create_supervised_sequences(
            ts, seq_length=SEQ_LENGTH, horizon=h, add_channel=True, return_index=True
        )
        n_total = len(Xh); n_tr = int(n_total*TRAIN_RATIO); n_va = int(n_total*VAL_RATIO)
        X_tr, y_tr = Xh[:n_tr], yh[:n_tr]
        X_va, y_va = Xh[n_tr:n_tr+n_va], yh[n_tr:n_tr+n_va]
        X_te, y_te = Xh[n_tr+n_va:], yh[n_tr+n_va:]
        t_te       = t_idx_h[n_tr+n_va:]

        scaler = StandardScaler()
        Ntr, Nva, Nte = X_tr.shape[0], X_va.shape[0], X_te.shape[0]
        X_tr = scaler.fit_transform(X_tr.reshape(Ntr, -1)).reshape(Ntr, SEQ_LENGTH, 1).astype('float32')
        X_va = scaler.transform(   X_va.reshape(Nva, -1)).reshape(Nva, SEQ_LENGTH, 1).astype('float32')
        X_te = scaler.transform(   X_te.reshape(Nte, -1)).reshape(Nte, SEQ_LENGTH, 1).astype('float32')
        y_tr = y_tr.astype('float32'); y_va = y_va.astype('float32'); y_te = y_te.astype('float32')

        tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                               batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        te_loader = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
                               batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        model = TimeSeriesTransformer(
            input_size=1, d_model=128, nhead=4, num_layers=2,
            dim_feedforward=256, dropout=0.1, readout='last'
        )
        print(f"\n[Transformer] Training for {h}-hour horizon")
        model, _ = train_model(model, tr_loader, num_epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE)

        metrics_rb, y_true, y_pred = evaluate_model_robust(model, te_loader, device=DEVICE, mape_floor=1.0)
        classic_mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100.0)

        results_tf[f"{h}-hour"] = {
            "MAE":  metrics_rb["MAE"],
            "RMSE": metrics_rb["RMSE"],
            "MAPE": classic_mape,
            "MAPE_robust": metrics_rb["MAPE_robust"],
            "SMAPE": metrics_rb["SMAPE"],
            "WMAPE": metrics_rb["WMAPE"],
            "Forecast": y_pred,
            "Actual":   y_true,
            "Time":     t_te,
        }

    # show+save baseline plots (3)
    for h in HORIZONS:
        plot_forecast_from_results(results_tf, f"{h}-hour", model_label="Transformer")

    # -------------------- Optuna tuning (guarded) --------------------
    try:
        import optuna
        from optuna.samplers import TPESampler
        OPTUNA_OK = True
    except Exception as e:
        print("Optuna not available; tuning sections skipped:", e)
        OPTUNA_OK = False

    def _make_splits_tf(series, h, seq_len):
        Xh, yh, t_idx_h = create_supervised_sequences(
            series, seq_length=seq_len, horizon=h, add_channel=True, return_index=True
        )
        n_total = len(Xh); n_tr = int(n_total*TRAIN_RATIO); n_va = int(n_total*VAL_RATIO)
        X_tr, y_tr = Xh[:n_tr], yh[:n_tr]
        X_va, y_va = Xh[n_tr:n_tr+n_va], yh[n_tr:n_tr+n_va]
        X_te, y_te = Xh[n_tr+n_va:], yh[n_tr+n_va:]
        t_te       = t_idx_h[n_tr+n_va:]
        from sklearn.preprocessing import StandardScaler as _SS
        scaler = _SS()
        Ntr, Nva, Nte = X_tr.shape[0], X_va.shape[0], X_te.shape[0]
        X_tr = scaler.fit_transform(X_tr.reshape(Ntr, -1)).reshape(Ntr, seq_len, 1).astype('float32')
        X_va = scaler.transform(   X_va.reshape(Nva, -1)).reshape(Nva, seq_len, 1).astype('float32')
        X_te = scaler.transform(   X_te.reshape(Nte, -1)).reshape(Nte, seq_len, 1).astype('float32')
        return (X_tr, y_tr.astype('float32'),
                X_va, y_va.astype('float32'),
                X_te, y_te.astype('float32'),
                t_te)

    results_tf_tuned = {}
    best_tf_params = {}

    if OPTUNA_OK:
        def _objective_tf(trial, h):
            seq_len = trial.suggest_categorical(f"h{h}_seq_len", [24,36,48,60,72,96])
            d_model = trial.suggest_categorical(f"h{h}_d_model", [64,96,128,160])
            nhead   = trial.suggest_categorical(f"h{h}_nhead",   [2,4,8])
            num_layers = trial.suggest_int(f"h{h}_layers", 1, 3)
            dim_ff     = trial.suggest_categorical(f"h{h}_ff", [128,256,384,512])
            dropout    = trial.suggest_float(f"h{h}_drop", 0.0, 0.3)
            lr         = trial.suggest_float(f"h{h}_lr", 5e-4, 3e-3, log=True)
            batch_size = trial.suggest_categorical(f"h{h}_bs", [16,32,64,128])

            if d_model % nhead != 0:
                raise optuna.TrialPruned()

            X_tr, y_tr, X_va, y_va, X_te, y_te, _ = _make_splits_tf(ts, h, seq_len)
            tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                                   batch_size=batch_size, shuffle=True, drop_last=False)
            va_loader = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
                                   batch_size=batch_size, shuffle=False, drop_last=False)

            model = TimeSeriesTransformer(
                input_size=1, d_model=d_model, nhead=nhead, num_layers=num_layers,
                dim_feedforward=dim_ff, dropout=dropout, readout='last'
            ).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            lossf = nn.MSELoss()

            best = float('inf'); left = EARLY_STOPPING_PATIENCE; max_epochs = min(EPOCHS, 40)
            for ep in range(1, max_epochs+1):
                model.train()
                for xb, yb in tr_loader:
                    xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                    opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
                model.eval(); preds, trues = [], []
                with torch.no_grad():
                    for xb, yb in va_loader:
                        pr = model(xb.to(DEVICE, non_blocking=True)).cpu().numpy()
                        preds.append(pr); trues.append(yb.numpy())
                y_pred = np.concatenate(preds); y_true = np.concatenate(trues)
                rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))

                trial.report(rmse, ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if rmse < best:
                    best, left = rmse, EARLY_STOPPING_PATIENCE
                else:
                    left -= 1
                    if left == 0:
                        break
            return best

        # ---- tuning loop
        for h in HORIZONS:
            print(f"\n=== Tuning Transformer for {h}-hour ===")
            study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
            study.optimize(lambda t: _objective_tf(t, h), n_trials=25, show_progress_bar=False)
            best_tf_params[h] = {"val_rmse": study.best_value, **study.best_params}
            print("Best val RMSE:", study.best_value)
            print("Best params:", study.best_params)

            seq_len = study.best_params[f"h{h}_seq_len"]
            d_model = study.best_params[f"h{h}_d_model"]
            nhead   = study.best_params[f"h{h}_nhead"]
            layers  = study.best_params[f"h{h}_layers"]
            dim_ff  = study.best_params[f"h{h}_ff"]
            drop    = study.best_params[f"h{h}_drop"]
            lr      = study.best_params[f"h{h}_lr"]
            bs      = study.best_params[f"h{h}_bs"]

            X_tr, y_tr, X_va, y_va, X_te, y_te, t_te = _make_splits_tf(ts, h, seq_len)
            X_trva = np.concatenate([X_tr, X_va], axis=0)
            y_trva = np.concatenate([y_tr, y_va], axis=0)

            trva_loader = DataLoader(TensorDataset(torch.from_numpy(X_trva), torch.from_numpy(y_trva)),
                                     batch_size=bs, shuffle=True, drop_last=False)
            te_loader   = DataLoader(TensorDataset(torch.from_numpy(X_te),   torch.from_numpy(y_te)),
                                     batch_size=bs, shuffle=False, drop_last=False)

            model = TimeSeriesTransformer(
                input_size=1, d_model=d_model, nhead=nhead, num_layers=layers,
                dim_feedforward=dim_ff, dropout=drop, readout='last'
            )
            model, _ = train_model(model, trva_loader, num_epochs=EPOCHS, lr=lr, device=DEVICE)

            metrics, y_true, y_pred = evaluate_model_robust(model, te_loader, device=DEVICE, mape_floor=1.0)
            results_tf_tuned[f"{h}-hour"] = {
                "MAE":  metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE": float(np.mean(np.abs((y_true - y_pred)/np.maximum(np.abs(y_true),1e-6)))*100.0),
                "MAPE_robust": metrics["MAPE_robust"],
                "SMAPE": metrics["SMAPE"],
                "WMAPE": metrics["WMAPE"],
                "Forecast": y_pred, "Actual": y_true, "Time": t_te,
                "seq_len": seq_len, "d_model": d_model, "nhead": nhead,
                "layers": layers, "ff": dim_ff, "dropout": drop, "lr": lr, "batch": bs,
                "val_RMSE": best_tf_params[h].get('val_rmse', best_tf_params[h].get('val_RMSE'))
            }

        # show+save tuned plots (3) 
        if results_tf_tuned:
            for h in HORIZONS:
                if f"{h}-hour" in results_tf_tuned:
                    plot_forecast_from_results(results_tf_tuned, f"{h}-hour", model_label="Transformer (tuned)")

        # -------- Refined Optuna pass --------
        best_tf_params_refined = {}
        results_tf_refined = {}

        # static nhead space + prune invalid combos
        def _objective_tf_refined(trial, h):
            seq_len   = trial.suggest_categorical(f"ref_h{h}_seq_len", [36, 48, 60, 72])
            d_model   = trial.suggest_categorical(f"ref_h{h}_d_model", [96, 128, 160])
            nhead     = trial.suggest_categorical(f"ref_h{h}_nhead", [2, 4, 8])  # static space
            num_layers = trial.suggest_int(f"ref_h{h}_layers", 1, 3)
            dim_ff     = trial.suggest_categorical(f"ref_h{h}_ff", [256, 384, 512])
            dropout    = trial.suggest_float(f"ref_h{h}_drop", 0.05, 0.25)
            lr         = trial.suggest_float(f"ref_h{h}_lr", 5e-4, 2e-3, log=True)
            batch_size = trial.suggest_categorical(f"ref_h{h}_bs", [16, 32, 64])

            # prune invalid (d_model, nhead) pairs
            if d_model % nhead != 0:
                raise optuna.TrialPruned()

            X_tr, y_tr, X_va, y_va, X_te, y_te, _ = _make_splits_tf(ts, h, seq_len)
            tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                                   batch_size=batch_size, shuffle=True, drop_last=False)
            va_loader = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
                                   batch_size=batch_size, shuffle=False, drop_last=False)

            model = TimeSeriesTransformer(
                input_size=1, d_model=d_model, nhead=nhead, num_layers=num_layers,
                dim_feedforward=dim_ff, dropout=dropout
            ).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            lossf = nn.MSELoss()

            best = float('inf'); left = EARLY_STOPPING_PATIENCE; max_epochs = min(EPOCHS, 40)
            for ep in range(1, max_epochs+1):
                model.train()
                for xb, yb in tr_loader:
                    xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                    opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
                model.eval(); preds, trues = [], []
                with torch.no_grad():
                    for xb, yb in va_loader:
                        pr = model(xb.to(DEVICE, non_blocking=True)).cpu().numpy()
                        preds.append(pr); trues.append(yb.numpy())
                y_pred = np.concatenate(preds); y_true = np.concatenate(trues)
                rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))

                trial.report(rmse, ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if rmse < best:
                    best, left = rmse, EARLY_STOPPING_PATIENCE
                else:
                    left -= 1
                    if left == 0:
                        break
            return best
        

        if OPTUNA_OK:
            for h in HORIZONS:
                print(f"\n=== Refined tuning Transformer for {h}-hour ===")
                study_ref = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
                study_ref.optimize(lambda t: _objective_tf_refined(t, h), n_trials=25, show_progress_bar=False)
                best_tf_params_refined[h] = {"val_rmse": study_ref.best_value, **study_ref.best_params}
                print("Refined best val RMSE:", study_ref.best_value)
                print("Refined params:", study_ref.best_params)

                seq_len = study_ref.best_params[f"ref_h{h}_seq_len"]
                d_model = study_ref.best_params[f"ref_h{h}_d_model"]
                nhead   = study_ref.best_params[f"ref_h{h}_nhead"]
                layers  = study_ref.best_params[f"ref_h{h}_layers"]
                dim_ff  = study_ref.best_params[f"ref_h{h}_ff"]
                drop    = study_ref.best_params[f"ref_h{h}_drop"]
                lr      = study_ref.best_params[f"ref_h{h}_lr"]
                bs      = study_ref.best_params[f"ref_h{h}_bs"]

                X_tr, y_tr, X_va, y_va, X_te, y_te, t_te = _make_splits_tf(ts, h, seq_len)
                X_trva = np.concatenate([X_tr, X_va], axis=0)
                y_trva = np.concatenate([y_tr, y_va], axis=0)

                trva_loader = DataLoader(TensorDataset(torch.from_numpy(X_trva), torch.from_numpy(y_trva)),
                                         batch_size=bs, shuffle=True, drop_last=False)
                te_loader   = DataLoader(TensorDataset(torch.from_numpy(X_te),   torch.from_numpy(y_te)),
                                         batch_size=bs, shuffle=False, drop_last=False)

                model = TimeSeriesTransformer(
                    input_size=1, d_model=d_model, nhead=nhead, num_layers=layers,
                    dim_feedforward=dim_ff, dropout=drop
                )
                model, _ = train_model(model, trva_loader, num_epochs=EPOCHS, lr=lr, device=DEVICE)

                metrics, y_true, y_pred = evaluate_model_robust(model, te_loader, device=DEVICE, mape_floor=1.0)
                results_tf_refined[f"{h}-hour"] = {
                    "MAE":  metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "MAPE": float(np.mean(np.abs((y_true - y_pred)/np.maximum(np.abs(y_true),1e-6)))*100.0),
                    "MAPE_robust": metrics["MAPE_robust"],
                    "SMAPE": metrics["SMAPE"],
                    "WMAPE": metrics["WMAPE"],
                    "Forecast": y_pred, "Actual": y_true, "Time": t_te,
                    "seq_len": seq_len, "d_model": d_model, "nhead": nhead,
                    "layers": layers, "ff": dim_ff, "dropout": drop, "lr": lr, "batch": bs,
                    "val_RMSE": best_tf_params_refined[h]['val_rmse'],
                }

            # show+save refined plots (3)
            if results_tf_refined:
                for h in HORIZONS:
                    if f"{h}-hour" in results_tf_refined:
                        plot_forecast_from_results(results_tf_refined, f"{h}-hour", model_label="Transformer (refined tuned)")

            # persist tuned/refined in context
            context["best_tf_params"] = best_tf_params
            context["best_tf_params_refined"] = best_tf_params_refined
            context["results_tf_tuned"] = results_tf_tuned
            context["results_tf_refined"] = results_tf_refined

    # always save baseline results
    context["results_tf"] = results_tf

    # keep knobs in context
    context.update({
        "DEVICE": DEVICE,
        "SEQ_LENGTH": SEQ_LENGTH,
        "HORIZONS": HORIZONS,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VAL_RATIO": VAL_RATIO,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
    })

    return context


if __name__ == "__main__":
    run({})
