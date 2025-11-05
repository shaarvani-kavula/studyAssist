from __future__ import annotations
import json, pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def norm_id_series(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    s = s.str.replace(r"\.0$", "", regex=True).str.strip()
    s = s.replace({"nan": None, "NaN": None})
    return s

def find_col_case_insensitive(df: pd.DataFrame, wanted: str) -> str | None:
    norm_map = {c.strip().lower(): c for c in df.columns}
    return norm_map.get(wanted.strip().lower())



@dataclass
class TrainConfig:
    # Inputs
    features_with_id_csv: str = "outputs/nsch_features_6to17_corr08_vif_NO_TARGET_WITH_ID.csv"
    raw_path_for_target: str  = "NSCH_2023e_Topical_CAHMI_DRC.csv"  # to attach flrsh6to17ct by HHID
    raw_id_col: str = "HHID"    # in raw
    canon_id_col: str = "hhid"  # in engineered
    target_source_col: str = "flrsh6to17ct"

    # Outputs
    out_dir: str = "outputs"

    # Model
    max_iter: int = 2000
    solver: str = "lbfgs"

    # SHAP
    shap_sample_size: int = 4000
    shap_seed: int = 123

    # Monte Carlo (parameter uncertainty)
    mc_n_draws: int = 1000
    row_subsample_for_mc: Optional[int] = None  # None = all rows

    # Probability bands
    bands: Tuple[Tuple[float, float, str], ...] = (
        (0.0, 0.35, "At-Risk"),
        (0.35, 0.65, "Moderate"),
        (0.65, 1.01, "Likely Flourishing"),
    )


class Banding:
    @staticmethod
    def name_for(p: float, bands: Tuple[Tuple[float,float,str], ...]) -> str:
        for lo, hi, nm in bands:
            if lo <= p < hi:
                return nm
        return bands[-1][2]


class LogisticTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)

        # Loaded later
        self.df_feat: Optional[pd.DataFrame] = None  # includes hhid + features
        self.df_raw: Optional[pd.DataFrame] = None
        self.feature_names: Optional[list] = None
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[LogisticRegression] = None

    # ---------- Data loading / join ----------
    def _load_features_with_id(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.features_with_id_csv)
        if self.cfg.canon_id_col not in df.columns:
            raise ValueError(f"Missing '{self.cfg.canon_id_col}' in features CSV.")
        df[self.cfg.canon_id_col] = norm_id_series(df[self.cfg.canon_id_col])
        return df


    def _attach_target(self, feat: pd.DataFrame) -> pd.DataFrame:
        raw = pd.read_csv(self.cfg.raw_path_for_target)
        # Normalize headers
        raw.columns = raw.columns.str.strip()
        feat = feat.copy()
        feat.columns = feat.columns.str.strip()
        # Resolve ID cols and normalize IDs
        raw_id = find_col_case_insensitive(raw, self.cfg.raw_id_col) or self.cfg.raw_id_col
        feat_id = self.cfg.canon_id_col
        if raw_id not in raw.columns:
            raise ValueError(f"Missing raw ID column '{self.cfg.raw_id_col}' (resolved to '{raw_id}').")
        if feat_id not in feat.columns:
            raise ValueError(f"Missing engineered ID column '{feat_id}' in features CSV.")
        raw[raw_id] = norm_id_series(raw[raw_id])
        feat[feat_id] = norm_id_series(feat[feat_id])

        # Resolve the actual target header in RAW (case/space-insensitive)
        tgt_raw = find_col_case_insensitive(raw, self.cfg.target_source_col)
        if tgt_raw is None:
            wanted = self.cfg.target_source_col.strip().lower()
            candidates = [c for c in raw.columns if wanted in c.strip().lower()]
            raise KeyError(
                f"Target column '{self.cfg.target_source_col}' not found in RAW. "
                f"Close candidates: {candidates[:10]} | First 20 cols: {list(raw.columns)[:20]}"
            )

        # Use a TEMP column name to avoid any surprises, then rename at the end
        TEMP_TGT = "__target_source__"
        raw_small = raw[[raw_id, tgt_raw]].rename(columns={raw_id: feat_id, tgt_raw: TEMP_TGT})

        # Merge
        out = feat.merge(raw_small, on=feat_id, how="left")

        # Diagnostics to help if anything goes wrong again
        if TEMP_TGT not in out.columns:
            print("[train] DEBUG columns after merge:", list(out.columns)[:40])
            raise KeyError(f"Temporary target column '{TEMP_TGT}' missing after merge.")

        missing = out[TEMP_TGT].isna().sum()
        if missing:
            print(f"[train] Warning: {missing} rows missing '{TEMP_TGT}' after merge. Dropping them.")
            out = out.dropna(subset=[TEMP_TGT])

        # Build binary target from the TEMP column
        out["target_bin"] = (out[TEMP_TGT] == 3).astype(int)

        # Finally, expose the canonical target name expected by the rest of the code
        out = out.rename(columns={TEMP_TGT: self.cfg.target_source_col})

        # Nice final sanity check
        if self.cfg.target_source_col not in out.columns:
            print("[train] DEBUG columns just before returning:", list(out.columns)[:40])
            raise KeyError(f"Column '{self.cfg.target_source_col}' still not present after rename.")

        return out


    # ---------- Fit + artifacts ----------
    def fit(self):
        # Load and join
        df_feat = self._load_features_with_id()
        df_all = self._attach_target(df_feat)

        # Split X/y (drop id + source + bin)
        id_col = self.cfg.canon_id_col
        tgt_src = self.cfg.target_source_col

        X_df = df_all.drop(columns=[id_col, tgt_src, "target_bin"])
        y = df_all["target_bin"].values
        ids = df_all[id_col].astype(str)
        ids = norm_id_series(df_all[id_col])
        print(f"Training rows: {X_df.shape[0]}, features: {X_df.shape[1]}")

        # Standardize
        self.feature_names = X_df.columns.tolist()
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        X_std = self.scaler.fit_transform(X_df.values)

        # Logistic
        self.model = LogisticRegression(max_iter=self.cfg.max_iter, solver=self.cfg.solver)
        self.model.fit(X_std, y)

        # Save artifacts
        (self.out_dir / "feature_names.json").write_text(json.dumps(self.feature_names, indent=2))
        with open(self.out_dir / "scaler.pkl", "wb") as f: pickle.dump(self.scaler, f)
        with open(self.out_dir / "logistic_model.pkl", "wb") as f: pickle.dump(self.model, f)

        # Coefficients
        coef = self.model.coef_.ravel()
        coef_df = (pd.DataFrame({"feature": self.feature_names,
                                 "coef": coef,
                                 "abs_coef": np.abs(coef)})
                   .sort_values("abs_coef", ascending=False))
        coef_df.to_csv(self.out_dir / "logistic_coefficients.csv", index=False)

        # SHAP summary (global)
        self._shap_summary(X_std)

        # MC percentiles
        self._mc_percentiles(X_std, y, ids)

        print("=== Training complete ===")
        print("Saved: logistic_model.pkl, scaler.pkl, feature_names.json")
        print("Saved: logistic_coefficients.csv, shap_summary.csv, mc_readiness_percentiles.csv")

    # ---------- SHAP ----------
    def _shap_summary(self, X_std: np.ndarray):
        # Background sample
        rng = np.random.default_rng(self.cfg.shap_seed)
        n = X_std.shape[0]
        k = min(n, self.cfg.shap_sample_size)
        idx = rng.choice(n, size=k, replace=False)
        X_bg = X_std[idx, :]

        # Prefer LinearExplainer for linear model
        try:
            explainer = shap.LinearExplainer(self.model, X_bg)
            sv = explainer.shap_values(X_bg)
            vals = sv if isinstance(sv, np.ndarray) else np.asarray(sv)
        except Exception:
            explainer = shap.Explainer(self.model, X_bg)
            out = explainer(X_bg)
            vals = getattr(out, "values", out)

        mean_abs = np.abs(vals).mean(axis=0)
        shap_df = pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
        shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv(self.out_dir / "shap_summary.csv", index=False)

    # ---------- Monte Carlo ----------
    @staticmethod
    def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

    def _mc_from_statsmodels(self, X_std: np.ndarray, y: np.ndarray, n_draws: int) -> Optional[pd.DataFrame]:
        try:
            X_design = sm.add_constant(X_std, has_constant="add")
            model = sm.Logit(y, X_design)
            res = model.fit(disp=False)
            params = np.asarray(res.params)
            cov = np.asarray(res.cov_params())
            rng = np.random.default_rng(42)
            betas = rng.multivariate_normal(mean=params, cov=cov, size=n_draws)  # (n_draws, k+1)
            probs = self._sigmoid(X_design @ betas.T)  # (n, n_draws)
            p10 = np.percentile(probs, 10, axis=1)
            p50 = np.percentile(probs, 50, axis=1)
            p90 = np.percentile(probs, 90, axis=1)
            return pd.DataFrame({"row_index": np.arange(X_std.shape[0]), "p10": p10, "p50": p50, "p90": p90})
        except Exception as e:
            print("Statsmodels covariance MC failed:", e)
            return None

    def _mc_from_bootstrap(self, X_std: np.ndarray, y: np.ndarray, n_draws: int,
                           row_subsample: Optional[int]) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        n = X_std.shape[0]
        pred_idx = rng.choice(n, size=row_subsample, replace=False) if (row_subsample and row_subsample < n) else np.arange(n)
        probs_draws = []
        for _ in range(n_draws):
            boot_idx = rng.choice(n, size=n, replace=True)
            Xb, yb = X_std[boot_idx, :], y[boot_idx]
            m = LogisticRegression(max_iter=self.cfg.max_iter, solver=self.cfg.solver).fit(Xb, yb)
            probs_draws.append(m.predict_proba(X_std[pred_idx, :])[:, 1])
        probs = np.vstack(probs_draws).T
        p10 = np.percentile(probs, 10, axis=1)
        p50 = np.percentile(probs, 50, axis=1)
        p90 = np.percentile(probs, 90, axis=1)
        return pd.DataFrame({"row_index": pred_idx, "p10": p10, "p50": p50, "p90": p90})

    def _mc_percentiles(self, X_std: np.ndarray, y: np.ndarray, ids: pd.Series):
        df_mc = self._mc_from_statsmodels(X_std, y, self.cfg.mc_n_draws)
        mc_mode = "statsmodels_cov"
        if df_mc is None:
            df_mc = self._mc_from_bootstrap(X_std, y, self.cfg.mc_n_draws, self.cfg.row_subsample_for_mc)
            mc_mode = "bootstrap_fallback"

        # Compute bands from p50
        df_mc["prob_band"] = df_mc["p50"].apply(lambda p: Banding.name_for(p, self.cfg.bands))

        # Attach hhid using row_index alignment
        df_mc["hhid"] = ids.iloc[df_mc["row_index"]].values

        # Save
        df_mc.to_csv(self.out_dir / "mc_readiness_percentiles.csv", index=False)

        # Save minimal run-config
        cfg = {
            "n_rows": int(X_std.shape[0]),
            "n_features": int(X_std.shape[1]),
            "mc_n_draws": self.cfg.mc_n_draws,
            "row_subsample_for_mc": self.cfg.row_subsample_for_mc,
            "bands": self.cfg.bands,
            "mc_mode": mc_mode
        }
        (self.out_dir / "mc_config.json").write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    cfg = TrainConfig()
    LogisticTrainer(cfg).fit()
