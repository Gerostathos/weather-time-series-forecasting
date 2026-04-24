# part3_vanilla_rnn.py

def run(context):
    """
    Vanilla RNN (PyTorch) ONLY + optional Optuna tuning and refined tuning.
    - Expects Part 1 to have run: context["df_interpolated"] preferred; will rebuild from context["df"] if needed.
    - Auto-saves all plots to plots/part3 before showing.
    - Stores results in context["results_rnn"], ["results_rnn_tuned"], ["results_rnn_ref_auto"].
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    # -------------------- autosave helper --------------------
    _plot_counter = {"i": 0}

    def _ensure_outdir(ctx):
        base = Path(ctx.get("plot_dir", "plots"))
        outdir = base / "part3"
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

    _patch_matplotlib_autosave(context, prefix="part3")

    # -------------------- tiny plot helper (used for baseline/tuned/refined) --------------------
    def _plot_forecast_from_results(results_dict, horizon_key: str, model_label: str):
        import numpy as _np
        import pandas as _pd
        import matplotlib.pyplot as _plt
        if horizon_key not in results_dict:
            return
        res = results_dict[horizon_key]
        y_true = _np.asarray(res["Actual"]); y_pred = _np.asarray(res["Forecast"])
        t = res.get("Time", None)
        mae  = res.get("MAE"); rmse = res.get("RMSE"); mape = res.get("MAPE"); maper = res.get("MAPE_robust")

        _plt.figure(figsize=(12, 4))
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
        if mae is not None and rmse is not None:
            title.append(f"MAE={mae:.3f}, RMSE={rmse:.3f}")
        if mape is not None:
            title.append(f"MAPE={mape:.2f}%")
        if maper is not None:
            title.append(f"MAPE₍robust₎={maper:.2f}%")
        _plt.title("  |  ".join(title))
        _plt.ylabel("Temperature (°C)")
        _plt.grid(True); _plt.legend(); _plt.tight_layout(); _plt.show()

    # -------------------- data (from Part 1) --------------------
    if "df_interpolated" in context:
        df_interpolated = context["df_interpolated"].copy()
    else:
        assert "df" in context, (
            "Neither 'df_interpolated' nor 'df' is defined in context. Run Part 1 first."
        )
        df = context["df"].copy()
        df['wv'] = df['wv'].mask(df['wv'] < 0)
        df['wv'] = df['wv'].interpolate(method='time')
        df['rain'] = df['rain'].clip(lower=0)
        wd_rad = np.deg2rad(df['wd'])
        df['wd_sin'] = np.sin(wd_rad); df['wd_cos'] = np.cos(wd_rad)
        df['SWDR_log'] = np.log1p(df['SWDR'])
        df_interpolated = df.interpolate(method='time')

    ts_nn = df_interpolated['T'].copy()
    ts_nn = ts_nn[~ts_nn.index.duplicated(keep='first')].sort_index().asfreq('h').dropna()

    # -------------------- Torch bits --------------------
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler
        TORCH_OK = True
    except Exception as e:
        print("PyTorch not available; RNN sections will be skipped:", e)
        TORCH_OK = False

    results_rnn = {}
    results_rnn_tuned = {}
    best_params_by_h = {}
    refined_spaces_rnn = {}
    results_rnn_ref_auto = {}

    if not TORCH_OK:
        context["results_rnn"] = results_rnn
        context["results_rnn_tuned"] = results_rnn_tuned
        context["best_params_by_h"] = best_params_by_h
        context["refined_spaces_rnn"] = refined_spaces_rnn
        context["results_rnn_ref_auto"] = results_rnn_ref_auto
        return context

    # ---- reproducibility ----
    import random
    SEED = int(context.get("SEED", 42))
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using DEVICE: {DEVICE}")

    # ---- hyperparams ----
    SEQ_LENGTH = int(context.get("SEQ_LENGTH", 48))
    HRS = list(context.get("HORIZONS", [1, 6, 24]))
    TRAIN_RATIO = float(context.get("TRAIN_RATIO", 0.70))
    VAL_RATIO   = float(context.get("VAL_RATIO",   0.20))
    TEST_RATIO  = 1.0 - TRAIN_RATIO - VAL_RATIO
    assert 0 < TEST_RATIO < 0.5, f"Bad TEST_RATIO={TEST_RATIO:.2f}"
    BATCH_SIZE = int(context.get("BATCH_SIZE", 128))
    EPOCHS = int(context.get("EPOCHS", 50))
    LEARNING_RATE = float(context.get("LEARNING_RATE", 1e-3))
    EARLY_STOPPING_PATIENCE = int(context.get("EARLY_STOPPING_PATIENCE", 5))

    # ---- helpers ----
    def create_supervised_sequences(series, seq_length: int, horizon: int,
                                    return_index: bool=False, add_channel: bool=True):
        s = pd.Series(series).astype(float).dropna()
        vals = s.to_numpy(); X, y, t_idx = [], [], []
        limit = len(vals) - seq_length - horizon + 1
        for i in range(limit):
            X.append(vals[i:i + seq_length])
            y.append(vals[i + seq_length + horizon - 1])
            if return_index:
                t_idx.append(s.index[i + seq_length + horizon - 1])
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if add_channel:
            X = X[..., None]
        if return_index:
            return X, y, np.asarray(t_idx, dtype=object)
        return X, y

    class VanillaRNN(nn.Module):
        def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1):
            super().__init__()
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.rnn(x); out = out[:, -1, :]; out = self.fc(out); return out.squeeze(-1)

    def train_model(model, train_loader, num_epochs=20, lr=1e-3, device=DEVICE):
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1, num_epochs+1):
            model.train(); running=0.0; n_s=0
            for xb, yb in train_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                optimizer.zero_grad(); preds = model(xb); loss = criterion(preds, yb)
                loss.backward(); optimizer.step()
                bs = xb.size(0); running += loss.item()*bs; n_s += bs
            print(f"[RNN] Epoch {epoch:02d}/{num_epochs} - Train MSE: {running/max(1,n_s):.6f}")
        return model, {}

    import math as _math
    @torch.no_grad()
    def evaluate_model(model, loader, device=DEVICE, eps=1e-6):
        model.eval().to(device); preds, trues = [], []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            yhat = model(xb)
            preds.append(yhat.detach().cpu().numpy()); trues.append(yb.detach().cpu().numpy())
        y_pred = np.concatenate(preds, axis=0).astype('float64')
        y_true = np.concatenate(trues, axis=0).astype('float64')
        mae  = float(np.mean(np.abs(y_pred - y_true)))
        rmse_ = float(_math.sqrt(np.mean((y_pred - y_true)**2)))
        mape_ = float(np.mean(np.abs((y_pred - y_true) / np.maximum(np.abs(y_true), eps))) * 100.0)
        return {"MAE": mae, "RMSE": rmse_, "MAPE": mape_}, y_true, y_pred

    @torch.no_grad()
    def evaluate_model_robust(model, loader, device=DEVICE, mape_floor=1.0, eps=1e-6):
        model.eval().to(device); preds, trues = [], []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            yhat = model(xb)
            preds.append(yhat.detach().cpu().numpy()); trues.append(yb.detach().cpu().numpy())
        y_pred = np.concatenate(preds, axis=0).astype("float64")
        y_true = np.concatenate(trues, axis=0).astype("float64")
        abs_err = np.abs(y_pred - y_true)
        mae = float(np.mean(abs_err))
        rmse = float(_math.sqrt(np.mean((y_pred - y_true)**2)))
        denom = np.maximum(np.abs(y_true), float(mape_floor))
        mape_robust = float(np.mean(abs_err/denom)*100.0)
        smape = float(np.mean(2.0*abs_err/(np.abs(y_true)+np.abs(y_pred)+eps))*100.0)
        wmape = float(np.sum(abs_err)/(np.sum(np.abs(y_true))+eps)*100.0)
        return {"MAE": mae, "RMSE": rmse, "MAPE_robust": mape_robust, "SMAPE": smape, "WMAPE": wmape}, y_true, y_pred

    # ---- base RNN per horizon ----
    for h in HRS:
        Xh, yh, t_idx_h = create_supervised_sequences(
            ts_nn, seq_length=SEQ_LENGTH, horizon=h, add_channel=True, return_index=True
        )
        n_total = len(Xh); n_tr = int(n_total*TRAIN_RATIO); n_va = int(n_total*VAL_RATIO)
        X_tr, y_tr, t_tr = Xh[:n_tr], yh[:n_tr], t_idx_h[:n_tr]
        X_va, y_va, t_va = Xh[n_tr:n_tr+n_va], yh[n_tr:n_tr+n_va], t_idx_h[n_tr:n_tr+n_va]
        X_te, y_te, t_te = Xh[n_tr+n_va:], yh[n_tr+n_va:], t_idx_h[n_tr+n_va:]

        from sklearn.preprocessing import StandardScaler
        scaler_in = StandardScaler()
        def _prep(X_):
            N = X_.shape[0]; return X_.reshape(N, -1)
        X_tr2d = scaler_in.fit_transform(_prep(X_tr))
        X_va2d = scaler_in.transform(_prep(X_va))
        X_te2d = scaler_in.transform(_prep(X_te))
        X_tr = X_tr2d.reshape(-1, SEQ_LENGTH, 1).astype('float32')
        X_va = X_va2d.reshape(-1, SEQ_LENGTH, 1).astype('float32')
        X_te = X_te2d.reshape(-1, SEQ_LENGTH, 1).astype('float32')
        y_tr = y_tr.astype('float32'); y_va = y_va.astype('float32'); y_te = y_te.astype('float32')

        tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                               batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        te_loader = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
                               batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        model = VanillaRNN(input_size=1, hidden_size=32, num_layers=1)
        print(f"\nTraining VanillaRNN for {h}-hour horizon:")
        _ = train_model(model, tr_loader, num_epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE)
        metrics, y_true, y_pred = evaluate_model(model, te_loader, device=DEVICE)

        results_rnn[f"{h}-hour"] = {
            "MAE":  metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "MAPE": metrics["MAPE"],
            "Forecast": y_pred,
            "Actual":   y_true,
            "Time":     t_te
        }

        # baseline plot (autosaved)
        _plot_forecast_from_results(results_rnn, f"{h}-hour", model_label="Vanilla RNN")

    # -------------------- Optuna tuning (guarded) --------------------
    try:
        import optuna
        from optuna.samplers import TPESampler
        OPTUNA_OK = True
    except Exception as e:
        print("Optuna not available; tuning sections skipped:", e)
        OPTUNA_OK = False

    def _make_splits_for_h(h):
        from sklearn.preprocessing import StandardScaler as _SS
        Xh, yh, t_idx_h = create_supervised_sequences(ts_nn, seq_length=SEQ_LENGTH, horizon=h, add_channel=True, return_index=True)
        n_total = len(Xh); n_tr = int(n_total*TRAIN_RATIO); n_va = int(n_total*VAL_RATIO)
        X_tr, y_tr = Xh[:n_tr], yh[:n_tr]
        X_va, y_va = Xh[n_tr:n_tr+n_va], yh[n_tr:n_tr+n_va]
        X_te, y_te = Xh[n_tr+n_va:], yh[n_tr+n_va:]
        t_te       = t_idx_h[n_tr+n_va:]
        scaler_in = _SS()
        def _prep(X_):
            N = X_.shape[0]; return X_.reshape(N, -1)
        X_tr2d = scaler_in.fit_transform(_prep(X_tr))
        X_va2d = scaler_in.transform(_prep(X_va))
        X_te2d = scaler_in.transform(_prep(X_te))
        X_tr = X_tr2d.reshape(-1, SEQ_LENGTH, 1).astype('float32')
        X_va = X_va2d.reshape(-1, SEQ_LENGTH, 1).astype('float32')
        X_te = X_te2d.reshape(-1, SEQ_LENGTH, 1).astype('float32')
        y_tr = y_tr.astype('float32'); y_va = y_va.astype('float32'); y_te = y_te.astype('float32')
        return X_tr, y_tr, X_va, y_va, X_te, y_te, t_te

    def _make_loaders(X_tr, y_tr, X_va, y_va, batch_size):
        tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        va_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
        va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        return tr_loader, va_loader

    if OPTUNA_OK:
        def _objective_for_h(trial, h, X_tr, y_tr, X_va, y_va):
            p = f"h{h}_"
            BATCH_CHOICES = [16, 32, 64, 128]
            if h == 24:
                hidden_size = trial.suggest_int(p+"hidden_size", 8, 32, step=8)
                num_layers  = 1
                lr          = trial.suggest_float(p+"lr", 5e-4, 1e-2, log=True)
                batch_size  = trial.suggest_categorical(p+"batch_size", BATCH_CHOICES)
            elif h == 6:
                hidden_size = trial.suggest_int(p+"hidden_size", 32, 80, step=16)
                num_layers  = trial.suggest_int(p+"num_layers", 1, 2)
                lr          = trial.suggest_float(p+"lr", 5e-5, 5e-3, log=True)
                batch_size  = trial.suggest_categorical(p+"batch_size", BATCH_CHOICES)
            else:
                hidden_size = trial.suggest_int(p+"hidden_size", 32, 128, step=16)
                num_layers  = trial.suggest_int(p+"num_layers", 1, 3)
                lr          = trial.suggest_float(p+"lr", 1e-4, 1e-2, log=True)
                batch_size  = trial.suggest_categorical(p+"batch_size", BATCH_CHOICES)

            tr_loader, va_loader = _make_loaders(X_tr, y_tr, X_va, y_va, batch_size)
            model = VanillaRNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(DEVICE)
            opt   = torch.optim.Adam(model.parameters(), lr=lr)
            lossf = nn.MSELoss()

            best = float('inf'); patience = EARLY_STOPPING_PATIENCE; left = patience; max_epochs = min(EPOCHS, 40)
            for ep in range(1, max_epochs+1):
                model.train()
                for xb, yb in tr_loader:
                    xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                    opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
                model.eval(); preds, trues = [], []
                with torch.no_grad():
                    for xb, yb in va_loader:
                        pr = model(xb.to(DEVICE, non_blocking=True))
                        preds.append(pr.cpu().numpy()); trues.append(yb.numpy())
                y_pred = np.concatenate(preds); y_true = np.concatenate(trues)
                rmse_ = float(np.sqrt(np.mean((y_pred - y_true)**2)))
                trial.report(rmse_, ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                if rmse_ < best:
                    best, left = rmse_, patience
                else:
                    left -= 1
                    if left == 0:
                        break
            return best

        # ---- tuning loop ----
        for h in [1, 6, 24]:
            print(f"\n=== Tuning VanillaRNN for horizon {h}h ===")
            X_tr, y_tr, X_va, y_va, X_te, y_te, t_te = _make_splits_for_h(h)
            study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED))
            study.optimize(lambda trial: _objective_for_h(trial, h, X_tr, y_tr, X_va, y_va),
                           n_trials=25, show_progress_bar=False)
            print(f"Best val RMSE: {study.best_value:.3f}\nBest params: {study.best_params}")
            best_params_by_h[h] = {"val_rmse": study.best_value, **study.best_params}

            # retrain on TRAIN+VAL
            bs = study.best_params[f"h{h}_batch_size"]
            hidden_size = study.best_params[f"h{h}_hidden_size"]
            lr = study.best_params[f"h{h}_lr"]
            num_layers = study.best_params.get(f"h{h}_num_layers", 1)

            X_trva = np.concatenate([X_tr, X_va], axis=0)
            y_trva = np.concatenate([y_tr, y_va], axis=0)
            trva_loader = DataLoader(TensorDataset(torch.from_numpy(X_trva), torch.from_numpy(y_trva)),
                                     batch_size=bs, shuffle=True, drop_last=False)
            te_loader   = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
                                     batch_size=bs, shuffle=False, drop_last=False)

            model = VanillaRNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
            model, _ = train_model(model, trva_loader, num_epochs=EPOCHS, lr=lr, device=DEVICE)
            metrics, y_true, y_pred = evaluate_model(model, te_loader, device=DEVICE)

            results_rnn_tuned[f"{h}-hour"] = {
                "MAE":  metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"],
                "Forecast": y_pred,
                "Actual":   y_true,
                "Time":     t_te
            }

        # plots for tuned results (3 figs)
        if len(results_rnn_tuned):
            for h in [1, 6, 24]:
                _plot_forecast_from_results(results_rnn_tuned, f"{h}-hour", model_label="Vanilla RNN (tuned)")

        # ---- refined ranges 
        def _mk_refined_space_rnn(h, winners):
            hs_win = winners.get(f"h{h}_hidden_size", 32 if h==24 else 64)
            nl_win = winners.get(f"h{h}_num_layers", 1)
            lr_win = winners.get(f"h{h}_lr", 1e-3)
            bs_win = winners.get(f"h{h}_batch_size", 64)
            hs_min = int(max(8, int(hs_win*0.5))); hs_max = int(min(256, int(hs_win*1.5)))
            hs_step = 8 if h==24 else 16
            nl_min = max(1, nl_win-1); nl_max = min(3, nl_win+1)
            lr_min = max(1e-5, lr_win/3.0); lr_max = min(1e-1, lr_win*3.0)
            bs_choices = sorted(set([16,32,64,128,bs_win]))
            return {"hidden_min":hs_min,"hidden_max":hs_max,"hidden_step":hs_step,
                    "layers_min":nl_min,"layers_max":nl_max,
                    "lr_min":lr_min,"lr_max":lr_max,
                    "batch_choices":bs_choices}

        refined_spaces_rnn = {h: _mk_refined_space_rnn(h, best_params_by_h.get(h, {})) for h in [1,6,24]}

        # ---- refined tuning
        best_params_by_h_refined_auto = {}
        results_rnn_ref_auto = {}

        def _objective_for_h_refined(trial, h, X_tr, y_tr, X_va, y_va, sp):
            hidden_size = trial.suggest_int(f"ref_h{h}_hidden", sp["hidden_min"], sp["hidden_max"], step=sp["hidden_step"])
            num_layers  = trial.suggest_int(f"ref_h{h}_layers", sp["layers_min"], sp["layers_max"])
            lr          = trial.suggest_float(f"ref_h{h}_lr", sp["lr_min"], sp["lr_max"], log=True)
            batch_size  = trial.suggest_categorical(f"ref_h{h}_bs", sp["batch_choices"])
            tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                                   batch_size=batch_size, shuffle=True, drop_last=False)
            va_loader = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
                                   batch_size=batch_size, shuffle=False, drop_last=False)
            model = VanillaRNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            lossf = nn.MSELoss()
            best = float('inf'); patience = EARLY_STOPPING_PATIENCE; left = patience; max_epochs = min(EPOCHS, 40)
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
                rmse_ = float(np.sqrt(np.mean((y_pred - y_true)**2)))
                trial.report(rmse_, ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                if rmse_ < best:
                    best, left = rmse_, patience
                else:
                    left -= 1
                    if left == 0:
                        break
            return best

        for h in [1,6,24]:
            print(f"\n=== Vanilla RNN — Auto-refined tuning for {h}-hour ===")
            X_tr, y_tr, X_va, y_va, X_te, y_te, t_te = _make_splits_for_h(h)
            sp = refined_spaces_rnn[h]
            study_ref = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED))
            study_ref.optimize(lambda t: _objective_for_h_refined(t, h, X_tr, y_tr, X_va, y_va, sp),
                               n_trials=25, show_progress_bar=False)
            best_params_by_h_refined_auto[h] = {"val_rmse": study_ref.best_value, **study_ref.best_params}
            print("Refined best val RMSE:", study_ref.best_value)
            print("Refined best params:", study_ref.best_params)

            bs   = study_ref.best_params[f"ref_h{h}_bs"]
            hid  = study_ref.best_params[f"ref_h{h}_hidden"]
            lyr  = study_ref.best_params[f"ref_h{h}_layers"]
            lr   = study_ref.best_params[f"ref_h{h}_lr"]

            X_trva = np.concatenate([X_tr, X_va], axis=0)
            y_trva = np.concatenate([y_tr, y_va], axis=0)
            trva_loader = DataLoader(TensorDataset(torch.from_numpy(X_trva), torch.from_numpy(y_trva)),
                                     batch_size=bs, shuffle=True, drop_last=False)
            te_loader   = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
                                     batch_size=bs, shuffle=False, drop_last=False)
            model = VanillaRNN(input_size=1, hidden_size=hid, num_layers=lyr)
            model, _ = train_model(model, trva_loader, num_epochs=EPOCHS, lr=lr, device=DEVICE)
            metrics, y_true, y_pred = evaluate_model_robust(model, te_loader, device=DEVICE, mape_floor=1.0)
            results_rnn_ref_auto[f"{h}-hour"] = {
                "MAE":  metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE": float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100.0),
                "MAPE_robust": metrics["MAPE_robust"],
                "SMAPE": metrics["SMAPE"],
                "WMAPE": metrics["WMAPE"],
                "Forecast": y_pred,
                "Actual":   y_true,
                "Time":     t_te,
                "hidden":   hid,
                "layers":   lyr,
                "batch":    bs,
                "lr":       lr,
                "val_RMSE": best_params_by_h_refined_auto[h].get("val_rmSE", best_params_by_h_refined_auto[h]["val_rmse"]),
            }

        # plots for refined tuned results (3 figs)
        if len(results_rnn_ref_auto):
            for h in [1, 6, 24]:
                _plot_forecast_from_results(results_rnn_ref_auto, f"{h}-hour", model_label="Vanilla RNN (refined tuned)")

    # -------------------- save back to context --------------------
    context["results_rnn"] = results_rnn
    context["best_params_by_h"] = best_params_by_h
    context["results_rnn_tuned"] = results_rnn_tuned
    context["refined_spaces_rnn"] = refined_spaces_rnn
    context["results_rnn_ref_auto"] = results_rnn_ref_auto
    return context


if __name__ == "__main__":
    run({})
