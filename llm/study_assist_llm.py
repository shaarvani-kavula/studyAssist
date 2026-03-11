from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json, pickle
import pandas as pd
import numpy as np
import shap
from sklearn.base import BaseEstimator

# ---------- Data objects ----------
@dataclass(frozen=True)
class ChildResult:
    hhid: str
    p50: Optional[float]
    band: Optional[str]
    top_strengths: List[Tuple[str, float]]  # [(feature, +shap)]
    top_growth: List[Tuple[str, float]]     # [(feature, -shap)]

# ---------- Loaders ----------
class Artifacts:
    def __init__(self, model_path: str, scaler_path: str, feat_path: str):
        with open(model_path, "rb") as f:
            self.model: BaseEstimator = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        self.features: List[str] = json.loads(open(feat_path).read())

class Dataset:
    def __init__(self, features_csv: str, id_col: str = "hhid", target_source: str = "flrsh6to17ct"):
        self.df = pd.read_csv(features_csv)
        assert id_col in self.df.columns, f"Missing {id_col}"
        self.id_col = id_col
        self.target_source = target_source

    def X_raw(self, feature_order: List[str]) -> pd.DataFrame:
        # Ensure only trained features, in training order
        return self.df[feature_order].copy()

    def ids(self) -> pd.Series:
        return self.df[self.id_col]

    def mc(self, mc_csv: Optional[str]) -> Optional[pd.DataFrame]:
        if not mc_csv:
            return None
        mc = pd.read_csv(mc_csv)
        assert self.id_col in mc.columns, f"Missing {self.id_col} in MC file"
        # locate p50 and band
        p50 = None
        for c in ["p50","prob_p50","median_prob","p_50"]:
            if c in mc.columns: p50 = c; break
        if p50 is None:
            raise ValueError("p50 column not found in MC file.")
        if "prob_band" not in mc.columns:
            def band(p):
                if p < 0.35: return "At-Risk"
                if p <= 0.65: return "Moderate"
                return "Likely Flourishing"
            mc["prob_band"] = mc[p50].apply(band)
        return mc[[self.id_col, p50, "prob_band"]].rename(columns={p50: "p50"})

# ---------- SHAP engine ----------
class ShapPerChild:
    def __init__(self, model: BaseEstimator, X_std: np.ndarray, feature_names: List[str], ids: pd.Series,
                 top_k: int = 3, min_abs: float = 0.0):
        self.model = model
        self.X_std = X_std
        self.feature_names = feature_names
        self.ids = ids
        self.top_k = top_k
        self.min_abs = min_abs
        self._shap_df: Optional[pd.DataFrame] = None

    def compute(self) -> pd.DataFrame:
        if self._shap_df is not None:
            return self._shap_df
        explainer = shap.LinearExplainer(self.model, self.X_std, feature_perturbation="interventional")
        sv = explainer.shap_values(self.X_std)
        if isinstance(sv, list):  
            sv = sv[-1]
        df = pd.DataFrame(sv, columns=self.feature_names)
        df.insert(0, "hhid", self.ids.values)
        self._shap_df = df
        return df

    def tops_for(self, hhid: str) -> Tuple[List[Tuple[str,float]], List[Tuple[str,float]]]:
        df = self.compute()
        row = df.loc[df["hhid"] == hhid]
        if row.empty:
            raise KeyError(f"hhid {hhid} not found")
        r = row[self.feature_names].iloc[0]
        pos = r[r > self.min_abs].sort_values(ascending=False).head(self.top_k)
        neg = r[r < -self.min_abs].sort_values(ascending=True).head(self.top_k)
        return list(pos.items()), list(neg.items())

# ---------- LLM renderer (child audience, simple & local) ----------
class ChildNote:
    @staticmethod
    def tips(band: str) -> List[str]:
        if band == "At-Risk":
            return ["Work for 5 minutes, then rest for 2.",
                    "Use a small checklist—cross off one thing.",
                    "Ask an adult to help you pick a start time."]
        if band == "Moderate":
            return ["Break big work into small steps.",
                    "Use a timer for short focus times.",
                    "Pack school things the night before."]
        return ["Try one extra practice problem.",
                "Teach a friend what you learned.",
                "Set one tiny stretch goal this week."]

    @staticmethod
    def list_line(pairs: List[Tuple[str,float]], title: str) -> str:
        return "" if not pairs else f"{title}: " + ", ".join([f for f,_ in pairs]) + "."

    def render(self, band: str, strengths: List[Tuple[str,float]], growth: List[Tuple[str,float]]) -> str:
        lines = [
            f"Hey! You’re in the **{band}** zone right now.",
            "You can make steady progress with small steps."
        ]
        s_line = self.list_line(strengths, "Things helping you")
        g_line = self.list_line(growth, "Things to practice")
        if s_line: lines.append(s_line)
        if g_line: lines.append(g_line)
        lines.append("Try these ideas this week:")
        lines += [f"- {t}" for t in self.tips(band)]
        lines.append("You’ve got this. Small steps add up!")
        return " ".join(lines)

# ---------- Orchestrator ----------
class StudyAssistPipeline:
    def __init__(self,
                 model_path: str,
                 scaler_path: str,
                 feat_path: str,
                 features_csv: str,
                 mc_csv: Optional[str] = None,
                 id_col: str = "hhid",
                 top_k: int = 3,
                 min_abs: float = 0.0):
        self.art = Artifacts(model_path, scaler_path, feat_path)
        self.data = Dataset(features_csv, id_col=id_col)
        X_raw = self.data.X_raw(self.art.features)
        self.ids = self.data.ids()
        # use the *saved* scaler so we match training exactly
        self.X_std = self.art.scaler.transform(X_raw.values)
        self.shap_engine = ShapPerChild(self.art.model, self.X_std, self.art.features, self.ids,
                                        top_k=top_k, min_abs=min_abs)
        self.mc = self.data.mc(mc_csv)
        self.note = ChildNote()

    def result_for(self, hhid: str) -> ChildResult:
        strengths, growth = self.shap_engine.tops_for(hhid)
        p50, band = None, None
        if self.mc is not None:
            r = self.mc.loc[self.mc["hhid"] == hhid]
            if not r.empty:
                p50 = float(r["p50"].iloc[0])
                band = str(r["prob_band"].iloc[0])
        if band is None:
            # If MC not provided, fall back to model prob
            import numpy as np
            idx = self.ids.reset_index(drop=True)
            pos = np.where(idx == hhid)[0]
            if len(pos):
                from scipy.special import expit
                beta = getattr(self.art.model, "coef_", None)
                b0 = getattr(self.art.model, "intercept_", None)
                if beta is not None and b0 is not None:
                    z = self.X_std[pos[0], :] @ beta.ravel() + b0.ravel()[0]
                    p50 = float(1/(1+np.exp(-z)))
                    band = "At-Risk" if p50 < 0.35 else ("Moderate" if p50 <= 0.65 else "Likely Flourishing")
        return ChildResult(hhid, p50, band, strengths, growth)

    def narrative_for(self, hhid: str) -> str:
        r = self.result_for(hhid)
        return self.note.render(r.band or "Moderate", r.top_strengths, r.top_growth)

    def export_shap_matrix(self, out_csv: str):
        self.shap_engine.compute().to_csv(out_csv, index=False)


if __name__ == "__main__":
    pipeline = StudyAssistPipeline(
        model_path="outputs/logistic_model.pkl",
        scaler_path="outputs/scaler.pkl",
        feat_path="outputs/feature_names.json",
        features_csv="outputs/nsch_features_6to17_corr08_vif_NO_TARGET_WITH_ID.csv",
        mc_csv="outputs/mc_readiness_percentiles.csv",   
        id_col="hhid",
        top_k=3,
        min_abs=0.1      
    )

    df_mc = pd.read_csv("outputs/mc_readiness_percentiles.csv")
    example_hhid = df_mc["hhid"].iloc[0]

    print("\n--- Example Narrative ---")
    print(pipeline.narrative_for(example_hhid))

    # Also export per-child SHAP matrix if desired
    pipeline.export_shap_matrix("outputs/shap_values_per_child.csv")
