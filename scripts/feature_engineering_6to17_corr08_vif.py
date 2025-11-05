import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# =====================================
# Config
# =====================================
RAW_PATH = "NSCH_2023e_Topical_CAHMI_DRC.csv"   
OUT_DIR = Path("outputs")

AGE_COL = "SC_AGE_YEARS"
AGE_MIN, AGE_MAX = 6, 17

# NSCH-style special missing codes (expand as needed)
SPECIAL_MISSING = {95, 96, 97, 98, 99, 999, 9999, 99999}

# Heuristic threshold for treating small-enum cols as categorical
CATEGORICAL_MAX_UNIQUE = 12

# Correlation pruning
CORR_THRESHOLD = 0.8

# VIF pruning
RUN_VIF = True
VIF_THRESHOLD = 10.0

# Columns that should always be treated as NUMERIC even if low unique count
FORCE_NUMERIC = {
    # household / person counts
    "TOTKIDS_R", "HHCOUNT", "FAMCOUNT", "TOTMALE", "TOTFEMALE",
    "HHCOUNT_IF",
    # mobility / places lived
    "PLACESLIVED", "PlacesLived_23", "PlacesLived",
    # age / weight / biometrics
    "SC_AGE_YEARS", "WEIGHT", "BIRTHWT", "BIRTHWT_VL", "BIRTHWT_L",
    # sleep & durations (often numeric)
    "HOURSLEEP", "HOURSLEEP05", "HrsSleep", "HrsSleep_23", "BedTime",
    # outdoors time
    "OutdrsWkDay_23", "OutdrsWkend_23",
    # composite counts
    "CondCnt_23", "Diff2more_23", "cntdiff",
}

# Columns that should always be treated as CATEGORICAL (small enums / binaries)
FORCE_CATEGORICAL = {
    # demographics
    "TENURE", "HHLANGUAGE", "SC_SEX", "SC_RACE_R", "SC_HISPANIC_R",
    "SC_RACE_R_IF", "SC_HISPANIC_R_IF", "A1_SEX", "A2_SEX", "BORNUSA",
    # language / race alternates
    "HHLanguage_23", "race4_23", "race7_23", "sex_23", "hispanic_23",
    # insurance & coverage types
    "CurrIns_23", "InsGap_23", "instype_23", "InsAdeq_23",
    "CURRINS", "INSGAP", "INSTYPE",
    # BMI class categorical
    "BMICLASS", "BMI3_6to17_23", "BMI4_6to17_23",
    # ACEs (binary/ordinal)
    "ACE1", "ACE3", "ACE4", "ACE5", "ACE6", "ACE7", "ACE8", "ACE9", "ACE10", "ACE11",
    "ACEct_23", "ACEctComm_23", "ACEincome_23", "ACEincome2_23", "ACE2more_23", "ACE2more6HH_23",
    # condition flags (survey yes/no)
    "anxiety_23", "depress_23", "OVERWEIGHT", "ToldOverweight_23",
    "ADHDind_23", "AutismInd_23", "CSHCN_23",
    # behavioral / engagement items (survey-coded)
    "ENGAGE_FAST", "ENGAGE_INTEREST", "ENGAGE_PICKY", "ENGAGE_BINGE", "ENGAGE_PURG",
    "ENGAGE_PILLS", "ENGAGE_EXERCISE", "ENGAGE_NOEAT", "ENGAGECONCERN",
    # school & neighborhood safety/amenities
    "SchlSafe_23", "NbhdSafe_23", "NbhdSupp_23", "NbhdAmenities_23",
    # smoking / vaping
    "smoking_23", "vape", "VAPE",
}

# Drop id columns
DROP_ID_COLS = {"HHID"}

# =====================================
# Helpers
# =====================================
def clean_missing_codes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        # coerce to numeric if possible
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="ignore")
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].replace({v: np.nan for v in SPECIAL_MISSING})
    return out

COUNT_NAME_PATTERNS = [
    r"(num|count|times|moves?|moved|episodes?|visits?|days|n_)\b",
    r"\b(total|freq|frequency)\b",
    r"(Cnt|Count|COUNT)$",
]

def is_count_like(colname: str, s: pd.Series) -> bool:
    name = colname.lower()
    if any(re.search(p, name) for p in COUNT_NAME_PATTERNS):
        return True
    non_na = s.dropna()
    if non_na.empty:
        return False
    # mostly integer and non-negative
    frac_int = (np.isclose(non_na, np.round(non_na))).mean()
    nonneg = (non_na >= 0).mean()
    return (frac_int >= 0.95) and (nonneg >= 0.99)

def split_categorical_numeric(df: pd.DataFrame, exclude: List[str]) -> Tuple[List[str], List[str]]:
    X_cols = [c for c in df.columns if c not in exclude]
    cat_like, num_like = [], []
    for c in X_cols:
        if c in DROP_ID_COLS:
            continue
        if c in FORCE_NUMERIC:
            num_like.append(c); continue
        if c in FORCE_CATEGORICAL:
            cat_like.append(c); continue
        # count heuristic
        if pd.api.types.is_numeric_dtype(df[c]) and is_count_like(c, df[c]):
            num_like.append(c); continue
        # fallback heuristic
        nunq = df[c].nunique(dropna=True)
        if nunq <= CATEGORICAL_MAX_UNIQUE:
            cat_like.append(c)
        else:
            num_like.append(c)
    return cat_like, num_like

def corr_prune(X: pd.DataFrame, threshold: float):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = []
    meta = []
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

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X.values)
    names = X.columns.tolist()
    vifs = []
    for i in range(X_std.shape[1]):
        y = X_std[:, i]
        X_others = np.delete(X_std, i, axis=1)
        if X_others.shape[1] == 0:
            vifs.append((names[i], 1.0, 0.0))
            continue
        model = LinearRegression()
        model.fit(X_others, y)
        r2 = model.score(X_others, y)
        vif = np.inf if (1 - r2) <= 1e-8 else 1.0 / (1.0 - r2)
        vifs.append((names[i], float(vif), float(r2)))
    return pd.DataFrame(vifs, columns=["feature", "VIF", "R2_on_others"]).sort_values("VIF", ascending=False)

# =====================================
# Main
# =====================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(RAW_PATH)

    # Drop identifiers and free-text desc up front
    drop_now = [c for c in df_raw.columns if (c in DROP_ID_COLS)]
    if drop_now:
        df_raw = df_raw.drop(columns=drop_now, errors="ignore")

    # Filter to ages 6–17
    if AGE_COL not in df_raw.columns:
        raise ValueError(f"Age column '{AGE_COL}' not found in CSV.")
    df = df_raw[(df_raw[AGE_COL] >= AGE_MIN) & (df_raw[AGE_COL] <= AGE_MAX)].copy()

    # Clean special missing codes
    df = clean_missing_codes(df)

    # Type split with overrides + count heuristic
    exclude_cols: List[str] = []
    cat_like, num_like = split_categorical_numeric(df, exclude=exclude_cols)

    # Impute 
    for c in num_like:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_like:
        mode_val = df[c].mode(dropna=True)
        mode_val = mode_val.iloc[0] if len(mode_val) else np.nan
        df[c] = df[c].fillna(mode_val)

    # One-hot encode categoricals
    X_cat = pd.get_dummies(df[cat_like], drop_first=True) if len(cat_like) > 0 else pd.DataFrame(index=df.index)
    X_num = df[num_like].copy()
    X_cat = X_cat.reindex(index=df.index)
    X = pd.concat([X_cat, X_num], axis=1)
    # Drop zero-variance columns
    X = X.loc[:, X.nunique() > 1]
    X = X.reset_index(drop=True)

    # Correlation pruning
    X_pruned, corr_meta = corr_prune(X, CORR_THRESHOLD)
    X_pruned = X_pruned.reset_index(drop=True)

    # VIF diagnostics and one-pass pruning
    vif_df = pd.DataFrame(columns=["feature", "VIF", "R2_on_others"])
    vif_dropped = []
    if RUN_VIF and X_pruned.shape[1] >= 2:
        vif_df = compute_vif(X_pruned)
        high = vif_df[vif_df["VIF"] >= VIF_THRESHOLD]["feature"].tolist()
        if len(high) > 0:
            X_pruned = X_pruned.drop(columns=high, errors="ignore")
            vif_dropped = high
            if X_pruned.shape[1] >= 2:
                vif_df = compute_vif(X_pruned)

    # Save artifacts
    out_data = OUT_DIR / "nsch_features_6to17_corr08_vif_NO_TARGET.csv"
    out_vars = OUT_DIR / "nsch_features_6to17_corr08_vif_variables.txt"
    out_corr_meta = OUT_DIR / "nsch_features_6to17_corr08_vif_metadata.csv"
    out_vif = OUT_DIR / "vif.csv"
    out_vif_dropped = OUT_DIR / "vif_dropped_features.csv"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_pruned.to_csv(out_data, index=False)
    pd.DataFrame({"variable": X_pruned.columns}).to_csv(out_vars, index=False, header=False)
    corr_meta.to_csv(out_corr_meta, index=False)
    vif_df.to_csv(out_vif, index=False)
    pd.DataFrame({"dropped_feature": vif_dropped}).to_csv(out_vif_dropped, index=False)

    # Console summary
    print("=== Feature Engineering Summary ===")
    print(f"Rows (ages {AGE_MIN}-{AGE_MAX}): {df.shape[0]}")
    print(f"Columns before prune: {X.shape[1]}  -> after corr@{CORR_THRESHOLD}: {X_pruned.shape[1]}")
    if RUN_VIF:
        print(f"VIF threshold: {VIF_THRESHOLD} | Dropped due to VIF: {len(vif_dropped)}")
    print(f"Saved dataset: {out_data}")
    print(f"Metadata: {out_corr_meta}, {out_vif}, {out_vif_dropped}")

if __name__ == "__main__":
    main()
