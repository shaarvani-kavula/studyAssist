from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


@dataclass
class FEConfig:
    raw_path: str = "NSCH_2023e_Topical_CAHMI_DRC.csv"
    out_dir: str = "outputs"

    # Age filter
    age_col: str = "SC_AGE_YEARS"
    age_min: int = 6
    age_max: int = 17

    # ID columns
    raw_id_col: str = "HHID"   
    canon_id_col: str = "hhid" 

    # Missing codes
    special_missing: Set[int] = None

    # Type inference
    categorical_max_unique: int = 12

    # Correlation / VIF pruning
    corr_threshold: float = 0.80
    run_vif: bool = True
    vif_threshold: float = 10.0

    # Force overrides
    force_numeric: Set[str] = None
    force_categorical: Set[str] = None

    def __post_init__(self):
        if self.special_missing is None:
            self.special_missing = {95, 96, 97, 98, 99, 999, 9999, 99999}
        if self.force_numeric is None:
            self.force_numeric = {
                "TOTKIDS_R","HHCOUNT","FAMCOUNT","TOTMALE","TOTFEMALE","HHCOUNT_IF",
                "PLACESLIVED","PlacesLived_23","PlacesLived",
                "SC_AGE_YEARS","WEIGHT","BIRTHWT","BIRTHWT_VL","BIRTHWT_L",
                "HOURSLEEP","HOURSLEEP05","HrsSleep","HrsSleep_23","BedTime",
                "OutdrsWkDay_23","OutdrsWkend_23",
                "CondCnt_23","Diff2more_23","cntdiff",
            }
        if self.force_categorical is None:
            self.force_categorical = {
                "TENURE","HHLANGUAGE","SC_SEX","SC_RACE_R","SC_HISPANIC_R",
                "SC_RACE_R_IF","SC_HISPANIC_R_IF","A1_SEX","A2_SEX","BORNUSA",
                "HHLanguage_23","race4_23","race7_23","sex_23","hispanic_23",
                "CurrIns_23","InsGap_23","instype_23","InsAdeq_23",
                "CURRINS","INSGAP","INSTYPE",
                "BMICLASS","BMI3_6to17_23","BMI4_6to17_23",
                "ACE1","ACE3","ACE4","ACE5","ACE6","ACE7","ACE8","ACE9","ACE10","ACE11",
                "ACEct_23","ACEctComm_23","ACEincome_23","ACEincome2_23",
                "ACE2more_23","ACE2more6HH_23",
                "anxiety_23","depress_23","OVERWEIGHT","ToldOverweight_23",
                "ADHDind_23","AutismInd_23","CSHCN_23",
                "SchlSafe_23","NbhdSafe_23","NbhdSupp_23","NbhdAmenities_23",
                "smoking_23","vape","VAPE",
            }


class FeatureEngineer:
    COUNT_NAME_PATTERNS = [
        r"(num|count|times|moves?|moved|episodes?|visits?|days|n_)\b",
        r"\b(total|freq|frequency)\b",
        r"(Cnt|Count|COUNT)$",
    ]

    def __init__(self, cfg: FEConfig):
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Helpers ----------
    def _clean_missing_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            # do NOT touch ID here; it was removed before calling this
            if not pd.api.types.is_numeric_dtype(out[c]):
                out[c] = pd.to_numeric(out[c], errors="ignore")
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].replace({v: np.nan for v in self.cfg.special_missing})
        return out

    def _is_count_like(self, colname: str, s: pd.Series) -> bool:
        name = colname.lower()
        if any(re.search(p, name) for p in self.COUNT_NAME_PATTERNS):
            return True
        non_na = s.dropna()
        if non_na.empty:
            return False
        frac_int = (np.isclose(non_na, np.round(non_na))).mean()
        nonneg = (non_na >= 0).mean()
        return (frac_int >= 0.95) and (nonneg >= 0.99)

    def _split_categorical_numeric(self, df: pd.DataFrame, exclude: List[str]) -> Tuple[List[str], List[str]]:
        X_cols = [c for c in df.columns if c not in exclude]
        cat_like, num_like = [], []
        for c in X_cols:
            if c in self.cfg.force_numeric:
                num_like.append(c); continue
            if c in self.cfg.force_categorical:
                cat_like.append(c); continue
            if pd.api.types.is_numeric_dtype(df[c]) and self._is_count_like(c, df[c]):
                num_like.append(c); continue
            nunq = df[c].nunique(dropna=True)
            if nunq <= self.cfg.categorical_max_unique:
                cat_like.append(c)
            else:
                num_like.append(c)
        return cat_like, num_like

    def _corr_prune(self, X: pd.DataFrame, threshold: float):
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop, meta = [], []
        for col in upper.columns:
            partners = upper.index[upper[col] >= threshold].tolist()
            if partners:
                to_drop.append(col)
                meta.append({
                    "dropped_variable": col,
                    "reason": f"abs_corr>={threshold}",
                    "correlated_with": ",".join(partners)
                })
        Xp = X.drop(columns=to_drop, errors="ignore")
        return Xp, pd.DataFrame(meta)

    def _compute_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_std = scaler.fit_transform(X.values)
        names = X.columns.tolist()
        vifs = []
        for i in range(X_std.shape[1]):
            y = X_std[:, i]
            X_others = np.delete(X_std, i, axis=1)
            if X_others.shape[1] == 0:
                vifs.append((names[i], 1.0, 0.0)); continue
            model = LinearRegression()
            model.fit(X_others, y)
            r2 = model.score(X_others, y)
            vif = np.inf if (1 - r2) <= 1e-8 else 1.0 / (1.0 - r2)
            vifs.append((names[i], float(vif), float(r2)))
        return pd.DataFrame(vifs, columns=["feature","VIF","R2_on_others"]).sort_values("VIF", ascending=False)

    # ---------- Pipeline ----------
    def run(self):
        print(f"[fe] Reading raw: {self.cfg.raw_path}")
        raw = pd.read_csv(self.cfg.raw_path)

        # Basic checks
        if self.cfg.age_col not in raw.columns:
            raise ValueError(f"Age column '{self.cfg.age_col}' not found.")
        if self.cfg.raw_id_col not in raw.columns:
            raise ValueError(f"ID column '{self.cfg.raw_id_col}' not found.")

        # Filter by age and reset index once
        m = (raw[self.cfg.age_col] >= self.cfg.age_min) & (raw[self.cfg.age_col] <= self.cfg.age_max)
        df = raw.loc[m].copy().reset_index(drop=True)
        print(f"[fe] Rows after age filter {self.cfg.age_min}-{self.cfg.age_max}: {df.shape[0]}")

        # Extract ID *after* filtering; keep as string; then drop from df so cleaner never touches it
        id_series = df[self.cfg.raw_id_col].astype(str).copy()
        df = df.drop(columns=[self.cfg.raw_id_col])

        # Clean special missing codes on non-ID columns
        df = self._clean_missing_codes(df)

        # Split types (ID is already removed)
        exclude = []  # already dropped ID above
        cat_like, num_like = self._split_categorical_numeric(df, exclude=exclude)

        # Impute
        for c in num_like:
            df[c] = df[c].fillna(df[c].median())
        for c in cat_like:
            mode_val = df[c].mode(dropna=True)
            mode_val = mode_val.iloc[0] if len(mode_val) else np.nan
            df[c] = df[c].fillna(mode_val)

        # One-hot encode small-enum categoricals
        X_cat = pd.get_dummies(df[cat_like], drop_first=True) if len(cat_like) > 0 else pd.DataFrame(index=df.index)
        X_num = df[num_like].copy()
        X = pd.concat([X_cat, X_num], axis=1)

        # Drop zero-variance and reset index
        X = X.loc[:, X.nunique() > 1].reset_index(drop=True)
        assert len(X) == len(id_series), "[fe] Length mismatch after encoding; check indexing."

        print(f"[fe] Columns before prune: {X.shape[1]}")

        # Correlation prune
        X_pruned, corr_meta = self._corr_prune(X, self.cfg.corr_threshold)
        X_pruned = X_pruned.reset_index(drop=True)
        print(f"[fe] Columns after corr@{self.cfg.corr_threshold}: {X_pruned.shape[1]}")

        # VIF diagnostics and optional prune (single pass)
        vif_df = pd.DataFrame(columns=["feature","VIF","R2_on_others"])
        vif_dropped = []
        if self.cfg.run_vif and X_pruned.shape[1] >= 2:
            vif_df = self._compute_vif(X_pruned)
            high = vif_df[vif_df["VIF"] >= self.cfg.vif_threshold]["feature"].tolist()
            if len(high) > 0:
                X_pruned = X_pruned.drop(columns=high, errors="ignore").reset_index(drop=True)
                vif_dropped = high
                if X_pruned.shape[1] >= 2:
                    vif_df = self._compute_vif(X_pruned)

        # Paths
        out_no_target = self.out_dir / "nsch_features_6to17_corr08_vif_NO_TARGET.csv"
        out_with_id   = self.out_dir / "nsch_features_6to17_corr08_vif_NO_TARGET_WITH_ID.csv"
        out_vars      = self.out_dir / "nsch_features_6to17_corr08_vif_variables.txt"
        out_corr_meta = self.out_dir / "nsch_features_6to17_corr08_vif_metadata.csv"
        out_vif       = self.out_dir / "vif.csv"
        out_vif_drop  = self.out_dir / "vif_dropped_features.csv"

        # Save outputs
        X_pruned.to_csv(out_no_target, index=False)
        with_id = pd.concat(
            [id_series.reset_index(drop=True).rename(self.cfg.canon_id_col),
             X_pruned.reset_index(drop=True)],
            axis=1
        )
        with_id.to_csv(out_with_id, index=False)
        pd.DataFrame({"variable": X_pruned.columns}).to_csv(out_vars, index=False, header=False)
        corr_meta.to_csv(out_corr_meta, index=False)
        vif_df.to_csv(out_vif, index=False)
        pd.DataFrame({"dropped_feature": vif_dropped}).to_csv(out_vif_drop, index=False)

        # Console summary
        print("=== Feature Engineering Summary ===")
        print(f"Saved: {out_no_target}")
        print(f"Saved WITH ID: {out_with_id}")
        print(f"Metadata: {out_corr_meta}, {out_vif}, {out_vif_drop}")
        print(f"Dropped due to VIF: {len(vif_dropped)}")


if __name__ == "__main__":
    try:
        FeatureEngineer(FEConfig()).run()
    except Exception as e:
        # Surface any error clearly
        import traceback
        print("\n[fe] ERROR:")
        print("".join(traceback.format_exc()))
        raise
