# AI-Based Prediction of Oil-Well Pressure Recovery Curves

> **Portfolio case study:** predicting the unobserved continuation of oil-well pressure recovery curves from partial temporal observations.

## Overview

This project investigates whether machine-learning models can reconstruct the subsequent, unobserved portion of pressure recovery curves in oil wells using the available portion of each curve together with operational, petrophysical and temporal information.

The task was formulated as a supervised regression problem with explicit temporal structure. The test observations correspond to later points in the same curves rather than a random train/test split.

### Models evaluated

- Random Forest
- XGBoost
- Multilayer Perceptron (MLP)
- TabPFN-3

### Core ML workflow

- Temporal feature engineering
- Temporal validation using the final 20% of each training curve
- Bayesian hyperparameter optimization with Optuna/TPE
- Five independent final runs using seeds 42–46
- Global and curve-level error analysis
- SHAP model interpretation
- Response-surface analysis for physically relevant feature pairs

## Main result

**XGBoost provided the best overall reconstruction performance.**

| Model | RMSE | MAE | R² | RMSE CV |
|---|---:|---:|---:|---:|
| **XGBoost** | **0.3655 ± 0.0023 MPa** | 0.2292 ± 0.0026 MPa | **0.9922 ± 0.0001** | 0.63% |
| Random Forest | 0.3723 ± 0.0058 MPa | 0.2104 ± 0.0028 MPa | 0.9919 ± 0.0003 | 1.57% |
| MLP | 0.4909 ± 0.0230 MPa | 0.2853 ± 0.0187 MPa | 0.9860 ± 0.0013 | 4.68% |
| TabPFN-3 | 0.9212 ± 0.0041 MPa | **0.0980 ± 0.0012 MPa** | 0.9507 ± 0.0004 | **0.45%** |

The low TabPFN-3 MAE should not be interpreted as the best global model: its substantially higher RMSE and lower R² indicate a different error profile, with larger errors having stronger influence on the squared-error metric.

## The engineering problem

Pressure recovery curves provide information about the dynamic behavior of a well and reservoir. In practice, a curve may be only partially available because the test has not finished, acquisition was interrupted, or some subsequent observations are unavailable.

The objective here is therefore not generic missing-value imputation. The model receives the **observed temporal segment** and estimates the **subsequent unobserved segment**.

## Data formulation

The final feature set contains original reservoir/operational variables and temporal descriptors such as:

- `Q_time`
- `t_index`
- `log_Q_time`
- `delta_Q_time`
- `elapsed_time`

The target variable is `R_pressure`, with `curve_id` used to preserve curve identity and temporal validation structure.

> Raw train/test data are intentionally not included in this public portfolio package. Add them only if you have permission to publish them.

## Optimization

Random Forest, XGBoost and MLP were optimized using **Optuna/TPE with 50 trials per model**. The validation protocol reserved the final 20% of each training curve.

Best validation RMSE:

- XGBoost: **0.3665**
- Random Forest: **0.3977**
- MLP: **0.4292**

TabPFN-3 was used in its pretrained Version 3 configuration and was not subjected to the same Optuna procedure.

## Curve reconstruction

The figure below shows six objectively selected curves at different percentiles of per-curve RMSE.

![Representative pressure-curve reconstructions](assets/representative_curves.png)

The model reproduces low-error and median cases closely, while the higher-error percentiles reveal the conditions in which reconstruction becomes more difficult.

## Error behavior across the prediction horizon

Prediction difficulty increases as the reconstruction moves farther away from the last observed point.

![Error by relative position in the unobserved segment](assets/error_vs_prediction_horizon.png)

The MAE increased from approximately **0.119 MPa** in the first 10% of the unobserved segment to **0.343 MPa** in the final 10%. Mean residuals also became more positive, indicating a growing tendency toward underestimation at longer horizons.

## Explainable AI

SHAP was used to investigate the XGBoost model.

![Global SHAP importance](assets/shap_global_importance.png)

The leading variables were:

1. Reservoir pressure — 39.99%
2. Bottomhole pressure — 12.24%
3. Temporal index — 10.91%

This is physically meaningful: the model relies strongly on the pressure state of the reservoir/well and on the temporal position within the pressure recovery process.

![SHAP beeswarm](assets/shap_beeswarm.png)

## Interaction and response surfaces

Three-dimensional XGBoost response surfaces were generated for selected high-importance variable pairs.

![XGBoost 3D response surfaces](assets/xgboost_3d_response_surfaces.png)

These surfaces complement SHAP by showing how the predicted pressure varies when important inputs change jointly while the remaining features are held at representative values.

## Practical implication

The central contribution is the ability to **predict the continuation of well pressure recovery curves from partial observations**.

Potential applications include:

- reconstruction of incomplete test records;
- early estimation of subsequent pressure behavior;
- support for pressure-test interpretation;
- consistency checking as new observations become available;
- reduced dependence on completely observed curves, subject to field validation and uncertainty controls.

The method is intended as a **decision-support tool**, not as a replacement for physical measurements or conventional reservoir-engineering analysis.

## Repository structure

```text
.
├── app.py
├── index.html
├── README.md
├── requirements-demo.txt
├── requirements-research.txt
├── assets/
├── data/
├── docs/
├── notebooks/
├── results/
└── src/
```

## Run the Streamlit portfolio locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate

pip install -r requirements-demo.txt
streamlit run app.py
```

## Reproducibility

The included notebooks document:

- Optuna optimization;
- final five-run evaluation;
- TabPFN-3 execution workflow.

The original train/test files are not bundled. To rerun the research pipeline, place the authorized files in `data/` and update file paths if necessary.

## Skills demonstrated

`Python` · `Pandas` · `Scikit-learn` · `XGBoost` · `Optuna` · `SHAP` · `TabPFN` · `Temporal validation` · `Feature engineering` · `Regression` · `Explainable AI` · `Reservoir engineering`

## Author

Replace this section with your name, LinkedIn URL and GitHub profile before publishing.
