### Modeling Determinants of Child Health using Machine Learning to Build StudyAssist — personalized to a child’s life circumstances

Child health is shaped by complex interactions among family, environment, and access to care. Traditional analyses can miss nonlinear, multi-factor patterns. Using ML and generative AI lets us explore these relationships in depth and turn them into personalized, supportive guidance—the basis for StudyAssist, a system that can adapt learning support to a child’s unique context.

#### Dataset Description
The dataset is published by the CAHMI Data Resource Center for Child and Adolescent Health (NSCH 2023). It contains 55,162 rows and ~895 columns, spanning a child’s demographics, health, family, insurance, and environment. Access and usage of data requires a request to the site.

Variables include:
- Demographics: age, race, parental education, household income
- Health: physical, emotional, and mental well-being
- Family & Environment: neighborhood safety/support, insurance coverage, etc.

#### Ethics and Data Use
All analyses are performed on de-identified, publicly available NSCH microdata.  
No personally identifiable information is used. The intent of this project is educational and research-based — not diagnostic or prescriptive.

**Citation** : Child and Adolescent Health Measurement Initiative (CAHMI) (2025). 2023 National Survey of Children's Health: SPSS dataset. Data Resource Center for Child and Adolescent Health supported by Cooperative Agreement U59MC27866 from the U.S. Department of Health and Human Services, Health Resources and Services Administration (HRSA), Maternal and Child Health Bureau (MCHB). Retrieved 10/05/2025 from childhealthdata.org

#### Project Goals
* 1. Preprocess data for ages 6–17: clean/impute, prune high correlations (Pearson), diagnose collinearity (VIF).

* 2. Model + Uncertainty: fit Logistic Regression, explain with SHAP, quantify uncertainty with Monte Carlo.

* 3. LLM Integration (StudyAssist): Design an LLM layer that turns model artifacts into gentle, personalized narratives and micro-recommendations. (In progress)

## Step 1: Data Cleaning & Feature Engineering

**Script:** `feature_engineering.py`  
What it does:
- Filters to **ages 6–17**
- Cleans NSCH special missing codes (e.g., 95/96/97/98/99)
- Heuristics + overrides to keep **count-like** fields numeric, one-hot encode small-enum categoricals
- **Correlation pruning** at |r| ≥ **0.80**  
- **VIF pass** (threshold 10) to reduce multicollinearity
- Produces an engineered dataset + metadata

### Target Construction
NSCH flourishing count `flrsh6to17ct` (0–3) is binarized:  
**flourish = 1 if `flrsh6to17ct == 3`, else 0**.

### Output
- **Rows:** ~33,638  
- **Features:** ~363–365
- Saved to: `outputs/nsch_features_6to17_corr08_vif.csv`

Run:
```bash
python feature_engineering.py
```

---

## Step 2: Monte Carlo Simulation with Logistic Model + SHAP

**Script:** `train_logistic_shap_montecarlo.py`

- Standardizes features; fits **LogisticRegression**
- Exports **coefficients**
- Computes **global SHAP** importance
- Samples coefficient uncertainty using **statsmodels covariance**
- Outputs **per-row p10 / p50 / p90** readiness probabilities

### Interpretation Bands
| Probability Range | Category |
|---|---|
| 0.00–0.35 | At-Risk |
| 0.35–0.65 | Moderate |
| 0.65–1.00 | Likely Flourishing |

Run:
```bash
python train_logistic_shap_montecarlo.py
```

---

## Step 3: StudyAssist (LLM Layer) — *In Progress*

The StudyAssist assistant will:
- Accept a child’s contextual details & strengths
- Use Monte Carlo readiness probabilities
- Explain *why* (SHAP + coefficients)
- Offer **gentle, personalized study guidance**

---

## Environment

```
Python 3.11+
pandas, numpy, scikit-learn, statsmodels, shap
```

---

## Quick Start

```bash
# 1) Run feature engineering
python feature_engineering.py

# 2) Run logistic + SHAP + Monte Carlo
python train_logistic_shap_montecarlo.py
```

---

## Notes & Scope
- Results reflect **associations**, not causation.
- Communicate probability **bands**, not deterministic conclusions.
