# part4_attention_rnn.py

def run(context):
    """
    Part 4 — Attention RNN (Encoder-Decoder with Luong attention) + optional Optuna tuning.
    - Uses cleaned series in context["df_interpolated"] (from Part 1).
    - Auto-saves ALL figures to plots/part4 as PNGs AND calls plt.show().
    - Stores: results_attn, results_attn_tuned, results_attn_ref_auto, best_attn_params, best_attn_params_refined_auto.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    # -------------------- autosave helper --------------------
    _plot_counter = {"i": 0}

    def _ensure_outdir(ctx):
        base = Path(ctx.get("plot_dir", "plots"))
        outdir = base / "part4"
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir

    def _patch_matplotlib_autosave(ctx, prefix="part4"):
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

    _patch_matplotlib_autosave(context, prefix="part4")

    # -------------------- torch setup --------------------
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    DEVICE = context.get("DEVICE")
    if DEVICE is None:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Attention block] Using DEVICE: {DEVICE}")

    # -------------------- series + hyperparams (defaults if missing) --------------------
    if "df_interpolated" not in context:
        raise RuntimeError("df_interpolated not found in context. Run Part 1 first.")
    df_interpolated = context["df_interpolated"].copy()

    ts = df_interpolated['T'].copy()
    ts = ts[~ts.index.duplicated(keep='first')].sort_index().asfreq('h').dropna()

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
    def create_supervised_sequences(series, seq_length: int, horizon: int, return_index: bool=False, add_channel: bool=True):
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
        y_pred = np.concatenate(preds).astype('float64'); y_true = np.concatenate(trues).astype('float64')
        abs_err = np.abs(y_pred - y_true)
        mae  = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
        denom = np.maximum(np.abs(y_true), float(mape_floor))
        mape_robust = float(np.mean(abs_err/denom) * 100.0)
        smape  = float(np.mean(2.0*abs_err/(np.abs(y_true)+np.abs(y_pred)+eps)) * 100.0)
        wmape  = float(np.sum(abs_err)/(np.sum(np.abs(y_true))+eps) * 100.0)
        return {"MAE":mae, "RMSE":rmse, "MAPE_robust":mape_robust, "SMAPE":smape, "WMAPE":wmape}, y_true, y_pred

    # -------------------- Attention model --------------------
    class EncoderRNN(nn.Module):
        def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 1):
            super().__init__(); self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        def forward(self, x):
            outputs, hidden = self.rnn(x); return outputs, hidden  # outputs: (B,T,H), hidden: (L,B,H)

    class LuongAttention(nn.Module):
        def __init__(self, hidden_size: int): super().__init__()
        def forward(self, decoder_hidden, encoder_outputs):
            dec_last = decoder_hidden[-1]                                                # (B,H)
            attn_scores = torch.bmm(encoder_outputs, dec_last.unsqueeze(2)).squeeze(2)  # (B,T)
            attn_weights = torch.softmax(attn_scores, dim=1)                             # (B,T)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)   # (B,H)
            return context, attn_weights

    class DecoderRNN(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
            super().__init__()
            self.rnn = nn.GRU(input_size=hidden_size + input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, y_prev, hidden, context):
            x_in = torch.cat([y_prev, context.unsqueeze(1)], dim=2)
            out, hidden = self.rnn(x_in, hidden)
            out = self.fc(out[:, -1, :])
            return out, hidden

    class Seq2SeqAttn(nn.Module):
        def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 1):
            super().__init__()
            self.input_size = input_size
            self.encoder = EncoderRNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
            self.attn    = LuongAttention(hidden_size)
            self.decoder = DecoderRNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        def forward(self, x):
            enc_out, hidden = self.encoder(x)
            context, _ = self.attn(hidden, enc_out)
            y_prev = x[:, -1:, :]
            out, _ = self.decoder(y_prev, hidden, context)
            return out.squeeze(-1)

    # -------------------- Train & evaluate (1h,6h,24h) --------------------
    from sklearn.preprocessing import StandardScaler
    results_attn = {}

    for h in HORIZONS:
        Xh, yh, t_idx_h = create_supervised_sequences(ts, seq_length=SEQ_LENGTH, horizon=h, add_channel=True, return_index=True)
        n_total = len(Xh); n_tr = int(n_total*TRAIN_RATIO); n_va = int(n_total*VAL_RATIO)
        X_tr, y_tr = Xh[:n_tr], yh[:n_tr]
        X_va, y_va = Xh[n_tr:n_tr+n_va], yh[n_tr:n_tr+n_va]
        X_te, y_te = Xh[n_tr+n_va:], yh[n_tr+n_va:]
        t_te       = t_idx_h[n_tr+n_va:]

        scaler_in = StandardScaler()
        Ntr, Nva, Nte = X_tr.shape[0], X_va.shape[0], X_te.shape[0]
        X_tr = scaler_in.fit_transform(X_tr.reshape(Ntr, -1)).reshape(Ntr, SEQ_LENGTH, 1).astype('float32')
        X_va = scaler_in.transform(   X_va.reshape(Nva, -1)).reshape(Nva, SEQ_LENGTH, 1).astype('float32')
        X_te = scaler_in.transform(   X_te.reshape(Nte, -1)).reshape(Nte, SEQ_LENGTH, 1).astype('float32')
        y_tr = y_tr.astype('float32'); y_va = y_va.astype('float32'); y_te = y_te.astype('float32')

        tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        te_loader = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        model = Seq2SeqAttn(input_size=1, hidden_size=64, num_layers=1)
        print(f"\n[Attention] Training for {h}-hour horizon")
        model, _ = train_model(model, tr_loader, num_epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE)

        metrics_rb, y_true, y_pred = evaluate_model_robust(model, te_loader, device=DEVICE, mape_floor=1.0)
        classic_mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100.0)

        results_attn[f"{h}-hour"] = {
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

    # Leaderboard 
    attn_df = (pd.DataFrame(results_attn).T.loc[[f"{h}-hour" for h in HORIZONS], ["MAE","RMSE","MAPE","MAPE_robust","SMAPE","WMAPE"]])
    try:
        from IPython.display import display
        display(attn_df.style.format({"MAE":"{:.3f}","RMSE":"{:.3f}","MAPE":"{:.2f}","MAPE_robust":"{:.2f}","SMAPE":"{:.2f}","WMAPE":"{:.2f}"}))
    except Exception:
        print(attn_df)

    # Plot helper — shows & autosaves
    def plot_forecast_from_results(results_dict, horizon_key: str, model_label: str = "Attention"):
        import numpy as _np, pandas as _pd, matplotlib.pyplot as _plt
        if horizon_key not in results_dict: return
        res = results_dict[horizon_key]
        y_true = _np.asarray(res["Actual"]); y_pred = _np.asarray(res["Forecast"]); t = res.get("Time", None)
        mae = res.get("MAE"); rmse = res.get("RMSE"); mape = res.get("MAPE"); maper = res.get("MAPE_robust")
        _plt.figure(figsize=(12,4))
        if t is not None:
            t = _pd.to_datetime(t); _plt.plot(t, y_true, label="Actual", linewidth=2); _plt.plot(t, y_pred, label=model_label, linestyle="--")
            _plt.axvline(t[0], ls="--", lw=1, alpha=0.6, label="Forecast start"); _plt.xlabel("Time"); _plt.xticks(rotation=30)
        else:
            _plt.plot(y_true, label="Actual", linewidth=2); _plt.plot(y_pred, label=model_label, linestyle="--"); _plt.xlabel("Sample index")
        title = [f"{horizon_key} — {model_label}"]
        if mae is not None and rmse is not None: title.append(f"MAE={mae:.3f}, RMSE={rmse:.3f}")
        if mape is not None: title.append(f"MAPE={mape:.2f}%")
        if maper is not None: title.append(f"MAPE₍robust₎={maper:.2f}%")
        _plt.title("  |  ".join(title)); _plt.ylabel("Temperature (°C)"); _plt.grid(True); _plt.legend(); _plt.tight_layout(); _plt.show()

    for h in HORIZONS:
        plot_forecast_from_results(results_attn, f"{h}-hour", model_label="Attention")

    # -------------------- Optuna tuning (guarded) --------------------
    try:
        import optuna
        from optuna.samplers import TPESampler
        OPTUNA_OK = True
    except Exception as e:
        print("Optuna not available; tuning sections skipped:", e)
        OPTUNA_OK = False

    best_attn_params = {}
    results_attn_tuned = {}

    def _make_splits_for_h_with_seq_len(series, h, seq_len):
        from sklearn.preprocessing import StandardScaler as _SS
        Xh, yh, t_idx_h = create_supervised_sequences(series, seq_length=seq_len, horizon=h, add_channel=True, return_index=True)
        n_total = len(Xh); n_tr = int(n_total*TRAIN_RATIO); n_va = int(n_total*VAL_RATIO)
        X_tr, y_tr = Xh[:n_tr], yh[:n_tr]
        X_va, y_va = Xh[n_tr:n_tr+n_va], yh[n_tr:n_tr+n_va]
        X_te, y_te = Xh[n_tr+n_va:], yh[n_tr+n_va:]
        t_te       = t_idx_h[n_tr+n_va:]
        scaler = _SS()
        Ntr, Nva, Nte = X_tr.shape[0], X_va.shape[0], X_te.shape[0]
        X_tr = scaler.fit_transform(X_tr.reshape(Ntr, -1)).reshape(Ntr, seq_len, 1).astype('float32')
        X_va = scaler.transform(   X_va.reshape(Nva, -1)).reshape(Nva, seq_len, 1).astype('float32')
        X_te = scaler.transform(   X_te.reshape(Nte, -1)).reshape(Nte, seq_len, 1).astype('float32')
        return (X_tr, y_tr.astype('float32'), X_va, y_va.astype('float32'), X_te, y_te.astype('float32'), t_te)

    if OPTUNA_OK:
        def _objective_attn(trial, h):
            seq_len     = trial.suggest_categorical(f"h{h}_seq_len", [24,36,42,48,60,68,72,96])
            hidden_size = trial.suggest_int(f"h{h}_hidden", 32, 128, step=16)
            num_layers  = trial.suggest_int(f"h{h}_layers", 1, 2)
            lr          = trial.suggest_float(f"h{h}_lr", 3e-4, 3e-3, log=True)
            batch_size  = trial.suggest_categorical(f"h{h}_bs", [16,32,64,128])
            X_tr, y_tr, X_va, y_va, X_te, y_te, _ = _make_splits_for_h_with_seq_len(ts, h, seq_len)
            tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=batch_size, shuffle=True, drop_last=False)
            va_loader = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)), batch_size=batch_size, shuffle=False, drop_last=False)
            model = Seq2SeqAttn(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(DEVICE)
            opt   = torch.optim.Adam(model.parameters(), lr=lr)
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
                        pr = model(xb.to(DEVICE, non_blocking=True)).cpu().numpy(); preds.append(pr); trues.append(yb.numpy())
                y_pred = np.concatenate(preds); y_true = np.concatenate(trues)
                rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
                trial.report(rmse, ep)
                if trial.should_prune(): raise optuna.TrialPruned()
                if rmse < best: best, left = rmse, EARLY_STOPPING_PATIENCE
                else:
                    left -= 1
                    if left == 0: break
            return best

        for h in HORIZONS:
            print(f"\n=== Tuning Attention model for {h}-hour ===")
            study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
            study.optimize(lambda trial: _objective_attn(trial, h), n_trials=25, show_progress_bar=False)
            print("Best value (val RMSE):", study.best_value)
            print("Best params:", study.best_params)
            best_attn_params[h] = {"val_rmse": study.best_value, **study.best_params}

            # retrain + test
            seq_len     = study.best_params[f"h{h}_seq_len"]
            hidden_size = study.best_params[f"h{h}_hidden"]
            num_layers  = study.best_params[f"h{h}_layers"]
            lr          = study.best_params[f"h{h}_lr"]
            bs          = study.best_params[f"h{h}_bs"]
            X_tr, y_tr, X_va, y_va, X_te, y_te, t_te = _make_splits_for_h_with_seq_len(ts, h, seq_len)
            X_trva = np.concatenate([X_tr, X_va], axis=0); y_trva = np.concatenate([y_tr, y_va], axis=0)
            trva_loader = DataLoader(TensorDataset(torch.from_numpy(X_trva), torch.from_numpy(y_trva)), batch_size=bs, shuffle=True, drop_last=False)
            te_loader   = DataLoader(TensorDataset(torch.from_numpy(X_te),   torch.from_numpy(y_te)),   batch_size=bs, shuffle=False, drop_last=False)
            model = Seq2SeqAttn(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
            model, _ = train_model(model, trva_loader, num_epochs=EPOCHS, lr=lr, device=DEVICE)
            metrics, y_true, y_pred = evaluate_model_robust(model, te_loader, device=DEVICE, mape_floor=1.0)
            classic_mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100.0)
            results_attn_tuned[f"{h}-hour"] = {
                "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "MAPE": classic_mape,
                "MAPE_robust": metrics["MAPE_robust"], "SMAPE": metrics["SMAPE"], "WMAPE": metrics["WMAPE"],
                "Forecast": y_pred, "Actual": y_true, "Time": t_te,
                "seq_len": seq_len, "hidden": hidden_size, "layers": num_layers, "batch": bs, "lr": lr, "val_RMSE": best_attn_params[h]["val_rmse"],
            }

        # ---------- tuned forecasts ----------
        if len(results_attn_tuned):
            tuned_df = (
                pd.DataFrame(results_attn_tuned)
                  .T.loc[[f"{h}-hour" for h in HORIZONS], ["MAE","RMSE","MAPE","MAPE_robust","SMAPE","WMAPE","val_RMSE"]]
            )
            try:
                from IPython.display import display
                display(tuned_df.style.format({
                    "MAE":"{:.3f}","RMSE":"{:.3f}","MAPE":"{:.2f}",
                    "MAPE_robust":"{:.2f}","SMAPE":"{:.2f}","WMAPE":"{:.2f}","val_RMSE":"{:.3f}"
                }).set_caption("Attention (Optuna tuned) — Test metrics"))
            except Exception:
                print(tuned_df)

            for h in HORIZONS:
                plot_forecast_from_results(
                    results_attn_tuned, f"{h}-hour", model_label="Attention (tuned)"
                )
        # -----------------------------------------------------------------------

    # -------------- Refined Optuna pass (auto) --------------
    refined_spaces_attn = {}
    results_attn_ref_auto = {}
    best_attn_params_refined_auto = {}

    if len(best_attn_params):
        ALLOWED_SEQ_LENS = [24,36,42,48,60,68,72,96]

        def _neighbors_from_allowed(center, allowed):
            if center not in allowed: center = min(allowed, key=lambda v: abs(v - center))
            idx = allowed.index(center); cand = {allowed[idx]}
            if idx-1 >= 0: cand.add(allowed[idx-1])
            if idx+1 < len(allowed): cand.add(allowed[idx+1])
            if idx-2 >= 0: cand.add(allowed[idx-2])
            if idx+2 < len(allowed): cand.add(allowed[idx+2])
            return sorted(cand)

        def _mk_refined_space_attn(h, winners):
            sl_win = winners.get(f"h{h}_seq_len", 48)
            hs_win = winners.get(f"h{h}_hidden", 64)
            nl_win = winners.get(f"h{h}_layers", 1)
            lr_win = winners.get(f"h{h}_lr", 1e-3)
            bs_win = winners.get(f"h{h}_bs", 64)
            seq_choices = _neighbors_from_allowed(sl_win, ALLOWED_SEQ_LENS)
            hs_min = int(max(16, int(hs_win*0.5))); hs_max = int(min(256, int(hs_win*1.5)))
            hs_min = (hs_min//16)*16; hs_max = ((hs_max+15)//16)*16
            nl_min = max(1, nl_win-1); nl_max = min(2, nl_win+1)
            lr_min = max(1e-5, lr_win/3.0); lr_max = min(1e-1, lr_win*3.0)
            bs_choices = sorted(set([16,32,64,128,bs_win]))
            return {"seq_choices": seq_choices, "hidden_min": hs_min, "hidden_max": hs_max, "hidden_step": 16,
                    "layers_min": nl_min, "layers_max": nl_max, "lr_min": lr_min, "lr_max": lr_max,
                    "batch_choices": bs_choices}

        refined_spaces_attn = {h: _mk_refined_space_attn(h, best_attn_params.get(h, {})) for h in HORIZONS}

        if OPTUNA_OK:
            def _objective_attn_refined(trial, h, sp):
                seq_len     = trial.suggest_categorical(f"ref_h{h}_seq_len", sp["seq_choices"])
                hidden_size = trial.suggest_int(       f"ref_h{h}_hidden", sp["hidden_min"], sp["hidden_max"], step=sp["hidden_step"])
                num_layers  = trial.suggest_int(       f"ref_h{h}_layers", sp["layers_min"], sp["layers_max"])
                lr          = trial.suggest_float(     f"ref_h{h}_lr",     sp["lr_min"],     sp["lr_max"],     log=True)
                batch_size  = trial.suggest_categorical(f"ref_h{h}_bs",    sp["batch_choices"])
                X_tr, y_tr, X_va, y_va, X_te, y_te, _ = _make_splits_for_h_with_seq_len(ts, h, seq_len)
                tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=batch_size, shuffle=True, drop_last=False)
                va_loader = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)), batch_size=batch_size, shuffle=False, drop_last=False)
                model = Seq2SeqAttn(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(DEVICE)
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                lossf = nn.MSELoss(); best = float('inf'); left = EARLY_STOPPING_PATIENCE; max_epochs = min(EPOCHS, 40)
                for ep in range(1, max_epochs+1):
                    model.train()
                    for xb, yb in tr_loader:
                        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                        opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
                    model.eval(); preds, trues = [], []
                    with torch.no_grad():
                        for xb, yb in va_loader:
                            pr = model(xb.to(DEVICE, non_blocking=True)).cpu().numpy(); preds.append(pr); trues.append(yb.numpy())
                    y_pred = np.concatenate(preds); y_true = np.concatenate(trues)
                    rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
                    trial.report(rmse, ep)
                    if rmse < best: best, left = rmse, EARLY_STOPPING_PATIENCE
                    else:
                        left -= 1
                        if left == 0: break
                return best

            best_attn_params_refined_auto = {}
            results_attn_ref_auto = {}

            for h in HORIZONS:
                print(f"\n=== Attention — Auto-refined tuning for {h}-hour ===")
                sp = refined_spaces_attn[h]
                study_ref = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
                study_ref.optimize(lambda t: _objective_attn_refined(t, h, sp), n_trials=25, show_progress_bar=False)
                best_attn_params_refined_auto[h] = {"val_rmse": study_ref.best_value, **study_ref.best_params}
                print("Refined best val RMSE:", study_ref.best_value)
                print("Refined best params:", study_ref.best_params)

                seq_len = study_ref.best_params[f"ref_h{h}_seq_len"]; hidden = study_ref.best_params[f"ref_h{h}_hidden"]
                layers = study_ref.best_params[f"ref_h{h}_layers"]; lr = study_ref.best_params[f"ref_h{h}_lr"]; bs = study_ref.best_params[f"ref_h{h}_bs"]

                X_tr, y_tr, X_va, y_va, X_te, y_te, t_te = _make_splits_for_h_with_seq_len(ts, h, seq_len)
                X_trva = np.concatenate([X_tr, X_va], axis=0); y_trva = np.concatenate([y_tr, y_va], axis=0)
                trva_loader = DataLoader(TensorDataset(torch.from_numpy(X_trva), torch.from_numpy(y_trva)), batch_size=bs, shuffle=True, drop_last=False)
                te_loader   = DataLoader(TensorDataset(torch.from_numpy(X_te),   torch.from_numpy(y_te)),   batch_size=bs, shuffle=False, drop_last=False)

                model = Seq2SeqAttn(input_size=1, hidden_size=hidden, num_layers=layers)
                model, _ = train_model(model, trva_loader, num_epochs=EPOCHS, lr=lr, device=DEVICE)

                metrics, y_true, y_pred = evaluate_model_robust(model, te_loader, device=DEVICE, mape_floor=1.0)
                classic_mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100.0)

                results_attn_ref_auto[f"{h}-hour"] = {
                    "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "MAPE": classic_mape,
                    "MAPE_robust": metrics["MAPE_robust"], "SMAPE": metrics["SMAPE"], "WMAPE": metrics["WMAPE"],
                    "Forecast": y_pred, "Actual": y_true, "Time": t_te,
                    "seq_len": seq_len, "hidden": hidden, "layers": layers, "batch": bs, "lr": lr,
                    "val_RMSE": best_attn_params_refined_auto[h].get("val_rmse", best_attn_params_refined_auto[h].get("val_RMSE")),
                }

            # summary + plots
            df = (pd.DataFrame(results_attn_ref_auto).T.loc[[f"{h}-hour" for h in HORIZONS], ["MAE","RMSE","MAPE","MAPE_robust","SMAPE","WMAPE","val_RMSE","seq_len","hidden","layers","batch","lr"]])
            try:
                from IPython.display import display
                display(df.style.format({"MAE":"{:.3f}","RMSE":"{:.3f}","MAPE":"{:.2f}","MAPE_robust":"{:.2f}","SMAPE":"{:.2f}","WMAPE":"{:.2f}","val_RMSE":"{:.3f}","lr":"{:.5f}"}).set_caption("Attention — Auto-refined Optuna: test metrics & best params"))
            except Exception:
                print(df)

            for h in HORIZONS:
                plot_forecast_from_results(results_attn_ref_auto, f"{h}-hour", model_label="Attention (refined tuned)")

    # -------------------- save back to context --------------------
    context["results_attn"] = results_attn
    context["results_attn_tuned"] = results_attn_tuned
    context["best_attn_params"] = best_attn_params
    context["refined_spaces_attn"] = refined_spaces_attn
    context["best_attn_params_refined_auto"] = best_attn_params_refined_auto
    context["results_attn_ref_auto"] = results_attn_ref_auto
    context["DEVICE"] = DEVICE
    context["SEQ_LENGTH"] = SEQ_LENGTH
    context["HORIZONS"] = HORIZONS
    context["TRAIN_RATIO"] = TRAIN_RATIO
    context["VAL_RATIO"] = VAL_RATIO
    context["BATCH_SIZE"] = BATCH_SIZE
    context["EPOCHS"] = EPOCHS
    context["LEARNING_RATE"] = LEARNING_RATE
    context["EARLY_STOPPING_PATIENCE"] = EARLY_STOPPING_PATIENCE
    return context


if __name__ == "__main__":
    run({})
