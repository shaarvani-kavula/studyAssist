import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import shap

# =========================
# Config
# =========================
IN_PATH = Path("outputs/nsch_features_6to17_corr08_vif.csv")  
OUT_DIR = Path("outputs")

# Target source binarized: 1 if ==3 else 0
TARGET_SOURCE = "flrsh6to17ct"

# SHAP
SHAP_SAMPLE_SIZE = 4000
SHAP_SEED = 123

# Monte Carlo (parameter uncertainty)
MC_N_DRAWS = 1000
ROW_SUBSAMPLE_FOR_MC = None   # e.g., 10000 for speed; None = all rows
BANDS = [(0.0, 0.35, "At-Risk"),
         (0.35, 0.65, "Moderate"),
         (0.65, 1.01, "Likely Flourishing")]

# =========================
# Helpers
# =========================
def save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def fit_logistic_sklearn(X_std: np.ndarray, y: np.ndarray, feature_names):
    logit = LogisticRegression(max_iter=2000, solver="lbfgs")
    logit.fit(X_std, y)
    coef = logit.coef_.ravel()
    coef_df = (pd.DataFrame({"feature": feature_names,
                             "coef": coef,
                             "abs_coef": np.abs(coef)})
               .sort_values("abs_coef", ascending=False))
    return logit, coef_df

def fit_logit_statsmodels(X_std_df: pd.DataFrame, y: np.ndarray):
    """Statsmodels fit for covariance matrix used in MC sampling."""
    try:
        X_design = sm.add_constant(X_std_df.values, has_constant="add")
        model = sm.Logit(y, X_design)
        res = model.fit(disp=False)
        return {"ok": True,
                "params": res.params,         # (k+1,)
                "cov": res.cov_params(),      # (k+1, k+1)
                "feature_names": ["const"] + list(X_std_df.columns)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def ensure_numpy_compat_for_shap():
    # Some SHAP versions expect np.bool/np.int/np.float aliases (removed in recent NumPy)
    for alias, real in [("bool", bool), ("int", int), ("float", float)]:
        if not hasattr(np, alias):
            setattr(np, alias, real)

def shap_summary(model, X_std: np.ndarray, feature_names, out_path: Path, seed=SHAP_SEED):
    """Compute SHAP mean |values|; uses LinearExplainer if possible, else generic Explainer."""
    ensure_numpy_compat_for_shap()
    try:
        rng = np.random.default_rng(seed)
        n = X_std.shape[0]
        k = min(n, SHAP_SAMPLE_SIZE)
        idx = rng.choice(n, size=k, replace=False)
        X_bg = X_std[idx, :]

        shap_df = None
        # Prefer LinearExplainer for linear models (faster, exact for logit link)
        try:
            explainer = shap.LinearExplainer(model, X_bg)
            sv = explainer.shap_values(X_bg)
            vals = sv if isinstance(sv, np.ndarray) else np.asarray(sv)
            mean_abs = np.abs(vals).mean(axis=0)
            shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        except Exception:
            # Fallback to model-agnostic
            explainer = shap.Explainer(model, X_bg)
            sv = explainer(X_bg)
            vals = getattr(sv, "values", sv)
            mean_abs = np.abs(vals).mean(axis=0)
            shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})

        shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
        save_df(shap_df, out_path)
        return True, shap_df
    except Exception as e:
        err = pd.DataFrame({"feature": ["<SHAP failed>"],
                            "mean_abs_shap": [np.nan],
                            "error": [str(e)]})
        save_df(err, out_path)
        return False, err

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

def mc_param_uncertainty_statsmodels(X_std_df: pd.DataFrame, stats_info: dict,
                                     n_draws: int, row_subsample: int | None):
    params = np.asarray(stats_info["params"])
    cov = np.asarray(stats_info["cov"])
    rng = np.random.default_rng(42)

    X_design = np.concatenate([np.ones((X_std_df.shape[0], 1)), X_std_df.values], axis=1)

    if row_subsample is not None and row_subsample < X_design.shape[0]:
        idx = rng.choice(X_design.shape[0], size=row_subsample, replace=False)
        X_use = X_design[idx, :]
    else:
        idx = np.arange(X_design.shape[0])
        X_use = X_design

    betas = rng.multivariate_normal(mean=params, cov=cov, size=n_draws)  # (n_draws, k+1)
    probs = sigmoid(X_use @ betas.T)  # (n_use, n_draws)

    p10 = np.percentile(probs, 10, axis=1)
    p50 = np.percentile(probs, 50, axis=1)
    p90 = np.percentile(probs, 90, axis=1)

    return pd.DataFrame({"row_index": idx, "p10": p10, "p50": p50, "p90": p90})

def mc_param_uncertainty_bootstrap(X_std: np.ndarray, y: np.ndarray, feature_names,
                                   n_draws: int, row_subsample: int | None):
    rng = np.random.default_rng(7)
    n = X_std.shape[0]
    pred_idx = rng.choice(n, size=row_subsample, replace=False) if row_subsample and row_subsample < n else np.arange(n)

    probs_draws = []
    for _ in range(n_draws):
        boot_idx = rng.choice(n, size=n, replace=True)
        Xb = X_std[boot_idx, :]
        yb = y[boot_idx]
        m, _ = fit_logistic_sklearn(Xb, yb, feature_names)
        probs_draws.append(m.predict_proba(X_std[pred_idx, :])[:, 1])

    probs = np.vstack(probs_draws).T
    p10 = np.percentile(probs, 10, axis=1)
    p50 = np.percentile(probs, 50, axis=1)
    p90 = np.percentile(probs, 90, axis=1)

    return pd.DataFrame({"row_index": pred_idx, "p10": p10, "p50": p50, "p90": p90})

def add_bands(df_pct: pd.DataFrame):
    def band_of(p):
        for lo, hi, name in BANDS:
            if (p >= lo) and (p < hi):
                return name
        return BANDS[-1][2]
    out = df_pct.copy()
    out["band_p50"] = out["p50"].apply(band_of)
    return out

# =========================
# Main
# =========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_PATH)

    if TARGET_SOURCE not in df.columns:
        raise ValueError(f"Target source column '{TARGET_SOURCE}' not in {IN_PATH}")

    # Build binary target: flourishing == 3 (meets all three items)
    df["target_bin"] = (df[TARGET_SOURCE] == 3).astype(int)
    y = df["target_bin"].values

    # X: drop target and any direct near-target source
    X_df = df.drop(columns=["target_bin", TARGET_SOURCE])

    # Standardize X
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X_df.values)
    feature_names = X_df.columns.tolist()
    X_std_df = pd.DataFrame(X_std, columns=feature_names)

    # Logistic baseline + coefficients
    model, coef_df = fit_logistic_sklearn(X_std, y, feature_names)
    save_df(coef_df, OUT_DIR / "logistic_coefficients.csv")

    # SHAP
    shap_ok, _ = shap_summary(model, X_std, feature_names, OUT_DIR / "shap_summary.csv")

    # Monte Carlo: prefer statsmodels covariance; fallback to bootstrap
    stats_info = fit_logit_statsmodels(X_std_df, y)
    if stats_info.get("ok", False):
        pct = mc_param_uncertainty_statsmodels(X_std_df, stats_info, MC_N_DRAWS, ROW_SUBSAMPLE_FOR_MC)
        mc_mode = "statsmodels_cov"
    else:
        pct = mc_param_uncertainty_bootstrap(X_std, y, feature_names, MC_N_DRAWS, ROW_SUBSAMPLE_FOR_MC)
        mc_mode = f"bootstrap_fallback: {stats_info.get('error', 'unknown error')}"

    pct = add_bands(pct)
    save_df(pct, OUT_DIR / "mc_readiness_percentiles.csv")

    # Save run config for provenance
    cfg = {
        "in_path": str(IN_PATH),
        "target_source_col": TARGET_SOURCE,
        "target_rule": "flrsh6to17ct==3 -> 1 else 0",
        "n_rows": int(df.shape[0]),
        "n_features": int(X_df.shape[1]),
        "shap_sample_size": SHAP_SAMPLE_SIZE,
        "mc_n_draws": MC_N_DRAWS,
        "row_subsample_for_mc": ROW_SUBSAMPLE_FOR_MC,
        "bands": BANDS,
        "mc_mode": mc_mode,
        "shap_ok": bool(shap_ok),
    }
    (OUT_DIR / "mc_config.json").write_text(json.dumps(cfg, indent=2))

    print("=== Logistic + SHAP + Monte Carlo complete ===")
    print(json.dumps(cfg, indent=2))
    print("Saved: logistic_coefficients.csv, shap_summary.csv, mc_readiness_percentiles.csv, mc_config.json")

if __name__ == "__main__":
    main()
