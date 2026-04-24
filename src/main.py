# main.py
import argparse
import importlib
import json
import os
from pathlib import Path
from datetime import datetime

# Non-interactive backend (prevents GUI popups)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Adjust module filenames here  ----------
PARTS = [
    ("part1", "part1_exploratory_preprocessing"),
    ("part2", "part2_ar_baseline"),
    ("part3", "part3_vanilla_rnn"),
    ("part4", "part4_attention_rnn"),
    ("part5", "part5_transformer"),
]
# ------------------------------------------------------------------

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _patch_show_to_autosave(outdir: Path, prefix: str):
    """
    Monkey-patch plt.show so every open figure is saved into outdir with the given prefix,
    then closed (to avoid duplicates). Returns a restore function.
    """
    _ensure_dir(outdir)
    orig_show = plt.show
    counter = {"i": 0}

    def auto_show(*args, **kwargs):
        # Save all current figures
        fig_nums = list(plt.get_fignums())
        for num in fig_nums:
            fig = plt.figure(num)
            counter["i"] += 1
            fname = outdir / f"{prefix}_{counter['i']:03d}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
        # Close to prevent duplicate saving next time
        plt.close("all")
        # Call original 
        return orig_show(*args, **kwargs)

    plt.show = auto_show
    return lambda: setattr(plt, "show", orig_show)

def _import_optional(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        print(f"[WARN] Could not import {module_name}: {e}")
        return None

def _bundle_pngs_to_pdf(img_dir: Path, pdf_path: Path):
    """
    Combine all PNGs in img_dir into a single PDF. Requires Pillow; if not available, skip gracefully.
    """
    try:
        from PIL import Image
    except Exception as e:
        print(f"[INFO] Pillow not installed; skipping PDF bundle for {img_dir.name}. ({e})")
        return

    pngs = sorted(img_dir.glob("*.png"))
    if not pngs:
        print(f"[INFO] No PNG plots found in {img_dir} — skipping PDF bundle.")
        return

    pages = []
    for p in pngs:
        try:
            im = Image.open(p).convert("RGB")
            pages.append(im)
        except Exception as e:
            print(f"[WARN] Could not open {p.name}: {e}")

    if pages:
        _ensure_dir(pdf_path.parent)
        first, rest = pages[0], pages[1:]
        first.save(pdf_path, save_all=True, append_images=rest)
        print(f"[INFO] Bundled {len(pages)} plots into: {pdf_path}")

def _rows_for_full_table(context, horizons):
    """
    Build rows for the 'full' table (RMSE, MAE, MAPE_robust, SMAPE, WMAPE) across models.
    """
    import pandas as pd

    def _pick_best_available(dicts, key):
        for d in dicts:
            if isinstance(d, dict) and key in d:
                return d[key]
        return None

    rows = []
    for h in horizons:
        hk = f"{h}-hour"
        ar   = _pick_best_available([context.get("results", {})], hk)
        rnn  = _pick_best_available([context.get("results_rnn_ref_auto", {}),
                                     context.get("results_rnn_tuned", {}),
                                     context.get("results_rnn", {})], hk)
        attn = _pick_best_available([context.get("results_attn_ref_auto", {}),
                                     context.get("results_attn_tuned", {}),
                                     context.get("results_attn", {})], hk)
        tfm  = _pick_best_available([context.get("results_tf_refined", {}),
                                     context.get("results_tf_tuned", {}),
                                     context.get("results_tf", {})], hk)

        def _extract(m):
            keys = ["RMSE","MAE","MAPE_robust","SMAPE","WMAPE"]
            return {k: (m.get(k) if m else None) for k in keys}

        rows.append({
            "Horizon": hk,
            **{f"AR_{k}":   v for k, v in _extract(ar).items()},
            **{f"RNN_{k}":  v for k, v in _extract(rnn).items()},
            **{f"ATTN_{k}": v for k, v in _extract(attn).items()},
            **{f"TF_{k}":   v for k, v in _extract(tfm).items()},
        })
    df = pd.DataFrame(rows).set_index("Horizon")
    order_cols = []
    for mp in ["AR","RNN","ATTN","TF"]:
        order_cols += [f"{mp}_RMSE", f"{mp}_MAE", f"{mp}_MAPE_robust", f"{mp}_SMAPE", f"{mp}_WMAPE"]
    return df[order_cols]

def _rows_for_compact_table(context, horizons):
    """
    Build rows for the 'compact' table (RMSE, MAE, MAPE) across models.
    """
    import pandas as pd
    import numpy as np

    def _pick_best_available(dicts, key):
        for d in dicts:
            if isinstance(d, dict) and key in d:
                return d[key]
        return None

    def _classic_mape(y_true, y_pred, eps=1e-6):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)

    def _extract_3(m):
        if not m:
            return {"RMSE": None, "MAE": None, "MAPE": None}
        mape = m.get("MAPE")
        if mape is None and ("Actual" in m and "Forecast" in m):
            try:
                mape = _classic_mape(m["Actual"], m["Forecast"])
            except Exception:
                mape = m.get("MAPE_robust")
        if mape is None and "MAPE_robust" in m:
            mape = m["MAPE_robust"]
        return {"RMSE": m.get("RMSE"), "MAE": m.get("MAE"), "MAPE": mape}

    rows = []
    for h in horizons:
        hk = f"{h}-hour"
        ar   = _pick_best_available([context.get("results", {})], hk)
        rnn  = _pick_best_available([context.get("results_rnn_ref_auto", {}),
                                     context.get("results_rnn_tuned", {}),
                                     context.get("results_rnn", {})], hk)
        attn = _pick_best_available([context.get("results_attn_ref_auto", {}),
                                     context.get("results_attn_tuned", {}),
                                     context.get("results_attn", {})], hk)
        tfm  = _pick_best_available([context.get("results_tf_refined", {}),
                                     context.get("results_tf_tuned", {}),
                                     context.get("results_tf", {})], hk)
        rows.append({
            "Horizon": hk,
            **{f"AR_{k}":   v for k, v in _extract_3(ar).items()},
            **{f"RNN_{k}":  v for k, v in _extract_3(rnn).items()},
            **{f"ATTN_{k}": v for k, v in _extract_3(attn).items()},
            **{f"TF_{k}":   v for k, v in _extract_3(tfm).items()},
        })
    df = pd.DataFrame(rows).set_index("Horizon")
    ordered_cols = []
    for model in ["AR", "RNN", "ATTN", "TF"]:
        ordered_cols += [f"{model}_RMSE", f"{model}_MAE", f"{model}_MAPE"]
    return df[ordered_cols]

def run_pipeline(args):
    # Base paths
    csv_path   = Path(args.csv).resolve()
    plot_dir   = _ensure_dir(Path(args.plot_dir).resolve())
    report_dir = _ensure_dir(Path(args.report_dir).resolve())

    # Context passed to all parts
    context = {
        "csv_path": str(csv_path),
        "plot_dir": str(plot_dir),
        "report_dir": str(report_dir),
        "HORIZONS": args.horizons,
        "TRAIN_RATIO": args.train_ratio,
        "VAL_RATIO": args.val_ratio,
        "RUN_STARTED": datetime.now().isoformat(timespec="seconds"),
    }

    print("[INFO] Starting pipeline with context:\n", json.dumps({
        **{k: v for k, v in context.items() if k not in ("df","df_interpolated")}, # don't print heavy objects
    }, indent=2))

    # Import modules
    imported = []
    for label, modname in PARTS:
        m = _import_optional(modname)
        imported.append((label, m))

    # Run each part; autosave every figure to its own folder and also bundle into a PDF
    for label, module in imported:
        if module is None:
            print(f"[SKIP] {label.upper()} — module not available")
            continue
        print(f"[INFO] Running {label.upper()} ...")

        # Per-part plot folder
        part_plot_dir = plot_dir / label
        _ensure_dir(part_plot_dir)

        # Patch show()
        restore = _patch_show_to_autosave(part_plot_dir, prefix=label)

        # Each part's run() should read csv, do its work, and update context
        try:
            context = module.run(context) or context
        except Exception as e:
            print(f"[ERROR] {label.upper()} failed: {e}")
        finally:
            # Restore original show()
            restore()

        # Bundle PNGs to a single PDF for convenience
        _bundle_pngs_to_pdf(part_plot_dir, report_dir / f"{label}_results.pdf")

    # ---------------- Comparison tables ----------------
    print("\n=== Overall comparison — lower is better (RMSE, MAE, MAPE_robust, SMAPE, WMAPE) ===")
    full_df = _rows_for_full_table(context, args.horizons)
    try:
        from IPython.display import display  # if running in notebook
        display(full_df.style.format("{:.3f}"))
    except Exception:
        print(full_df.round(3))
    full_csv = report_dir / "comparison_full.csv"
    full_df.to_csv(full_csv)
    print(f"[INFO] Saved: {full_csv}")

    print("\n=== Model comparison (RMSE, MAE, MAPE) — lower is better ===")
    compact_df = _rows_for_compact_table(context, args.horizons)
    try:
        from IPython.display import display
        display(compact_df.style.format("{:.3f}"))
    except Exception:
        print(compact_df.round(3))
    compact_csv = report_dir / "comparison_compact.csv"
    compact_df.to_csv(compact_csv)
    print(f"[INFO] Saved: {compact_csv}")

    print(f"\n[INFO] Finished.\n[INFO] Figures saved under: {plot_dir}\n[INFO] Reports saved under: {report_dir}")

def parse_args():
    p = argparse.ArgumentParser(description="Run full pipeline and save per-part plots & comparison tables.")
    p.add_argument("--csv", required=True, help="Path to cleaned_weather.csv")
    p.add_argument("--plot-dir", default="./plots", help="Root directory to save plots (per-part subfolders)")
    p.add_argument("--report-dir", default="./reports", help="Directory to save PDFs/CSVs")
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 6, 24], help="Forecast horizons (hours)")
    p.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio")
    p.add_argument("--val-ratio", type=float, default=0.20, help="Validation split ratio")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
