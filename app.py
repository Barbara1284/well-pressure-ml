from pathlib import Path
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
ASSETS = BASE / "assets"
RESULTS = BASE / "results"

st.set_page_config(
    page_title="Oil-Well Pressure AI",
    page_icon="📈",
    layout="wide",
)

st.title("AI-Based Prediction of Oil-Well Pressure Recovery Curves")
st.caption("Portfolio case study · Machine Learning · Reservoir Engineering")

st.markdown(
    """
This project predicts the **subsequent unobserved segment** of oil-well pressure
recovery curves from partial temporal observations and reservoir/operational features.
"""
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best model", "XGBoost")
c2.metric("Mean RMSE", "0.3655 MPa")
c3.metric("Mean R²", "0.9922")
c4.metric("Final runs", "5 seeds")

tabs = st.tabs([
    "Problem",
    "Model comparison",
    "Curve reconstruction",
    "Explainability",
    "Try your results",
])

with tabs[0]:
    st.subheader("The engineering problem")
    st.write(
        "The task is not conventional missing-value imputation. "
        "For each pressure recovery curve, the available temporal segment is used "
        "to estimate the subsequent unobserved pressure trajectory."
    )
    st.markdown(
        """
**Workflow**
1. Preserve curve identity and temporal order.
2. Engineer temporal descriptors.
3. Optimize RF, XGBoost and MLP with Optuna/TPE.
4. Compare all models across seeds 42–46.
5. Analyze error per curve and across the prediction horizon.
6. Interpret the best model with SHAP.
"""
    )

with tabs[1]:
    st.subheader("Final five-run comparison")
    df = pd.read_csv(RESULTS / "model_comparison.csv")
    show = df.copy()
    show["RMSE"] = show.apply(lambda r: f'{r.rmse_mean:.4f} ± {r.rmse_sd:.4f}', axis=1)
    show["MAE"] = show.apply(lambda r: f'{r.mae_mean:.4f} ± {r.mae_sd:.4f}', axis=1)
    show["R²"] = show.apply(lambda r: f'{r.r2_mean:.4f} ± {r.r2_sd:.4f}', axis=1)
    show["RMSE CV (%)"] = show["rmse_cv_percent"].map(lambda x: f"{x:.2f}")
    st.dataframe(show[["model","RMSE","MAE","R²","RMSE CV (%)"]], hide_index=True, use_container_width=True)
    st.info(
        "XGBoost achieved the best overall RMSE and R². "
        "TabPFN-3 had the lowest MAE but a much higher RMSE, revealing a different error profile."
    )

with tabs[2]:
    st.subheader("Representative reconstructions")
    st.image(str(ASSETS / "representative_curves.png"), use_container_width=True)
    st.subheader("Error across the prediction horizon")
    st.image(str(ASSETS / "error_vs_prediction_horizon.png"), use_container_width=True)
    st.write(
        "The MAE generally increases as predictions move farther from the observed segment, "
        "which is an important practical limitation for long-horizon reconstruction."
    )

with tabs[3]:
    st.subheader("Global SHAP importance")
    st.image(str(ASSETS / "shap_global_importance.png"), use_container_width=True)
    st.subheader("SHAP distribution")
    st.image(str(ASSETS / "shap_beeswarm.png"), use_container_width=True)
    st.subheader("3D response surfaces")
    st.image(str(ASSETS / "xgboost_3d_response_surfaces.png"), use_container_width=True)
    st.write(
        "Reservoir pressure, bottomhole pressure and temporal descriptors dominate the model's predictions, "
        "providing an interpretable connection between the ML model and the physical problem."
    )

with tabs[4]:
    st.subheader("Visualize your own prediction file")
    st.write(
        "Upload a CSV exported from your pipeline. This portfolio demo visualizes predictions; "
        "it does not bundle the proprietary/raw training data."
    )
    st.code("curve_id,Q_time,observed_pressure,predicted_pressure,segment", language="text")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        user_df = pd.read_csv(up)
        required = {"curve_id", "Q_time", "observed_pressure", "predicted_pressure"}
        missing = required - set(user_df.columns)
        if missing:
            st.error(f"Missing columns: {sorted(missing)}")
        else:
            ids = user_df["curve_id"].dropna().unique().tolist()
            selected = st.selectbox("Curve ID", ids)
            curve = user_df[user_df["curve_id"] == selected].sort_values("Q_time")
            plot_df = curve.set_index("Q_time")[["observed_pressure","predicted_pressure"]]
            st.line_chart(plot_df)
            err = curve["observed_pressure"] - curve["predicted_pressure"]
            rmse = (err.pow(2).mean()) ** 0.5
            mae = err.abs().mean()
            a,b = st.columns(2)
            a.metric("RMSE", f"{rmse:.4f}")
            b.metric("MAE", f"{mae:.4f}")

st.divider()
st.caption(
    "Portfolio project. Raw research data are not included by default; publish them only when permitted."
)
