# =============================================================================
# app.py
# PART A1
# Imports + Config + Model Loading + PCM Loading + Shared Helpers + Tabs
# =============================================================================

import os
import io
import warnings

import joblib
import numpy as np
import pandas as pd

import streamlit as st

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)
from utils.inference import (
    BATTERY,
    FLOW_LIMITS,
    clamp_battery_temp,
    run_timeseries,
    build_feature_row,
    predict_batch,
)
from scipy.optimize import minimize


warnings.filterwarnings("ignore")

# =============================================================================
# Utils Imports
# =============================================================================

from utils.model_loader import (
    load_model_artifacts,
)

from utils.inference import (
    BATTERY,
    FLOW_LIMITS,
    clamp_battery_temp,
    run_timeseries,
)

from utils.validation import (
    load_validation_database,
    run_validation,
    validation_summary,
    add_error_columns,
    filter_case,
)

from utils.adaptive_flow import (
    DEFAULT_PROFILE,
    run_adaptive_controller,
    build_continuous_controller,
    controller_summary,
)

from utils.joint_optimization import (
    run_joint_optimization,
    optimization_summary,
    compare_against_standard,
)

from utils.screening import (
    run_material_screening,
)

# =============================================================================
# Streamlit Config
# =============================================================================

st.set_page_config(
    page_title="BTMS PINN Platform",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS
# =============================================================================

st.markdown("""
<style>

.main-title{
    font-size:2.1rem;
    font-weight:700;
    color:#1f4e79;
}

.sub-title{
    font-size:1rem;
    color:#666;
    margin-bottom:1rem;
}

.metric-card{
    background:#f0f6ff;
    border-radius:10px;
    padding:16px 20px;
    border-left:4px solid #1f4e79;
    margin-bottom:8px;
}

.metric-label{
    font-size:.78rem;
    color:#666;
    text-transform:uppercase;
    letter-spacing:.05em;
}

.metric-value{
    font-size:1.7rem;
    font-weight:700;
    color:#1f4e79;
}

.metric-unit{
    font-size:.85rem;
    color:#888;
}

.warn-box{
    background:#fff3cd;
    border-left:4px solid #f0ad4e;
    padding:10px 14px;
    border-radius:6px;
    margin:8px 0;
}

.ok-box{
    background:#d4edda;
    border-left:4px solid #28a745;
    padding:10px 14px;
    border-radius:6px;
    margin:8px 0;
}

.section-header{
    font-size:1.3rem;
    font-weight:600;
    margin-top:1rem;
    margin-bottom:0.5rem;
}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# Header
# =============================================================================

st.markdown(
    '<div class="main-title">🔋 PCM-Assisted Battery Thermal Management Platform</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">PINN Surrogate • Optimization • Validation • Materials Discovery</div>',
    unsafe_allow_html=True,
)

# =============================================================================
# Directories
# =============================================================================

MODEL_DIR = "models"
DATA_DIR = "data"

PCM_DATABASE_PATH = os.path.join(
    DATA_DIR,"materials",
    "pcm_database.csv"
)

VALIDATION_DB_PATH = os.path.join(
    DATA_DIR,
    "validation",
    "cfd_validation.csv"
)

# =============================================================================
# Model Files
# =============================================================================

PTH_PATH = os.path.join(
    MODEL_DIR,
    "best_pinn.pth"
)

INPUT_SCALER_PATH = os.path.join(
    MODEL_DIR,
    "input_scaler.pkl"
)

OUTPUT_SCALER_PATH = os.path.join(
    MODEL_DIR,
    "output_scaler.pkl"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "feature_columns.pkl"
)

TARGET_PATH = os.path.join(
    MODEL_DIR,
    "target_columns.pkl"
)

# =============================================================================
# File Validation
# =============================================================================

required_files = [
    PTH_PATH,
    INPUT_SCALER_PATH,
    OUTPUT_SCALER_PATH,
    FEATURE_PATH,
    TARGET_PATH,
]

missing_files = [
    f for f in required_files
    if not os.path.exists(f)
]

if missing_files:

    st.error(
        "Missing required model files."
    )

    for file in missing_files:
        st.write(file)

    st.stop()

# =============================================================================
# Cached Model Loading
# =============================================================================

@st.cache_resource
def load_btms_model():

    artifacts = load_model_artifacts(
        PTH_PATH,
        INPUT_SCALER_PATH,
        OUTPUT_SCALER_PATH,
        FEATURE_PATH,
        TARGET_PATH,
    )

    return artifacts

# =============================================================================
# Cached PCM Database
# =============================================================================

@st.cache_data
def load_pcm_database():

    if not os.path.exists(
        PCM_DATABASE_PATH
    ):
        raise FileNotFoundError(
            f"Missing PCM database: {PCM_DATABASE_PATH}"
        )

    df = pd.read_csv(
        PCM_DATABASE_PATH
    )

    pcm_database = {}

    for _, row in df.iterrows():

        pcm_database[
            row["pcm_name"]
        ] = {

            "density":
                row["density_kg_m3"],

            "cp":
                row["specific_heat_J_kgK"],

            "k":
                row["thermal_conductivity_W_mK"],

            "viscosity":
                row["viscosity_Pa_s"],

            "latent_heat":
                row["latent_heat_J_kg"],

            "tsolidus":
                row["tsolidus_K"],

            "tliquidus":
                row["tliquidus_K"],
        }

    return df, pcm_database

# =============================================================================
# Load Resources
# =============================================================================

with st.spinner(
    "Loading PINN model..."
):

    artifacts = load_btms_model()

model = artifacts["model"]
device = artifacts["device"]

input_scaler = artifacts[
    "input_scaler"
]

output_scaler = artifacts[
    "output_scaler"
]

FEATURES = artifacts[
    "features"
]

TARGETS = artifacts[
    "targets"
]

checkpoint = artifacts[
    "checkpoint"
]

FIDX = {
    f: i
    for i, f in enumerate(FEATURES)
}

pcm_df, PCM_DATABASE = (
    load_pcm_database()
)

# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:

    st.header("System Status")

    st.success(
        "PINN Model Loaded"
    )

    st.write(
        f"Device: {device}"
    )

    if isinstance(
        checkpoint,
        dict
    ):

        st.write(
            f"Epoch: {checkpoint.get('epoch','N/A')}"
        )

        st.write(
            f"R²: {checkpoint.get('r2','N/A')}"
        )

    st.divider()

    st.header("Database")

    st.write(
        f"PCMs Loaded: {len(PCM_DATABASE)}"
    )

    if os.path.exists(
        VALIDATION_DB_PATH
    ):
        st.success(
            "Validation DB Found"
        )
    else:
        st.warning(
            "Validation DB Missing"
        )

# =============================================================================
# Session State
# =============================================================================

if "best_pcm" not in st.session_state:
    st.session_state.best_pcm = None

if "best_flow" not in st.session_state:
    st.session_state.best_flow = None

if "screening_results" not in st.session_state:
    st.session_state.screening_results = None

if "validation_results" not in st.session_state:
    st.session_state.validation_results = None

# =============================================================================
# Shared Helpers
# =============================================================================

def figure_download_button(
    fig,
    label,
    filename,
):

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        dpi=200,
        bbox_inches="tight"
    )

    buffer.seek(0)

    st.download_button(
        label,
        data=buffer,
        file_name=filename,
        mime="image/png",
    )


def dataframe_download_button(
    df,
    label,
    filename,
):

    st.download_button(
        label,
        data=df.to_csv(
            index=False
        ).encode(),
        file_name=filename,
        mime="text/csv",
    )


def metric_card(
    label,
    value,
    unit=""
):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div>{unit}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# Tabs
# =============================================================================

(
    tab1,
    tab2,
    tab3,
    tab4,
    tab5,
    tab6,
    tab7,
    tab8,
) = st.tabs([

    "📈 Time Series",

    "🔍 Optimal PCM",

    "💧 Optimal Flow",

    "🧪 PCM Comparison",

    "📊 CFD Validation",

    "🎛 Adaptive Control",

    "🤝 Joint Optimization",

    "🚀 Materials Screening",
])

# =============================================================================
# END PART A1
# NEXT:
# PART A2 = TAB 1 (Time Series)
# =============================================================================
# =============================================================================
# TAB 1 — TIME SERIES PREDICTION
# =============================================================================

with tab1:

    st.markdown(
        '<div class="section-header">Time Series Prediction</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Predict battery temperature, PCM temperature, liquid fraction "
        "and outlet temperature over time."
    )

    # -------------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        battery_temp = st.selectbox(
            "Battery Temperature (°C)",
            [50, 70, 90],
            index=0,
        )

        pcm_name = st.selectbox(
            "PCM Material",
            list(PCM_DATABASE.keys()) + ["Custom"],
        )

    with col2:

        mass_flow_gs = st.slider(
            "Mass Flow Rate (g/s)",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
        )

        t_end = st.slider(
            "Simulation End Time (s)",
            min_value=60,
            max_value=900,
            value=900,
            step=30,
        )

    with col3:

        n_points = st.slider(
            "Time Resolution",
            min_value=50,
            max_value=500,
            value=181,
            step=10,
        )

        show_raw_data = st.checkbox(
            "Show Data Table",
            value=False,
        )

    # -------------------------------------------------------------------------
    # Custom PCM
    # -------------------------------------------------------------------------

    if pcm_name == "Custom":

        st.markdown(
            "### Custom PCM Properties"
        )

        c1, c2 = st.columns(2)

        with c1:

            custom_density = st.number_input(
                "Density (kg/m³)",
                700.0,
                1000.0,
                800.0,
                key="custom_density"
            )

            custom_cp = st.number_input(
                "Specific Heat (J/kg·K)",
                1500.0,
                3000.0,
                2000.0,
                key="custom_cp"
            )

            custom_k = st.number_input(
                "Thermal Conductivity (W/m·K)",
                0.10,
                1.00,
                0.20,
                key="custom_k"
            )

            custom_viscosity = st.number_input(
                "Viscosity (Pa·s)",
                0.001,
                0.020,
                0.004,
                format="%.4f",
                key="custom_viscosity"
            )

        with c2:

            custom_latent = st.number_input(
                "Latent Heat (J/kg)",
                100000.0,
                250000.0,
                170000.0,
            )

            custom_tsolidus = st.number_input(
                "Tsolidus (K)",
                290.0,
                360.0,
                314.15,
                key="custom_tsolidus"
            )

            custom_tliquidus = st.number_input(
                "Tliquidus (K)",
                291.0,
                365.0,
                318.15,
                key="custom_tliquidus"
            )

        pcm = {
            "density": custom_density,
            "cp": custom_cp,
            "k": custom_k,
            "viscosity": custom_viscosity,
            "latent_heat": custom_latent,
            "tsolidus": custom_tsolidus,
            "tliquidus": custom_tliquidus,
        }

    else:

        pcm = PCM_DATABASE[
            pcm_name
        ]

    # -------------------------------------------------------------------------
    # Run Prediction
    # -------------------------------------------------------------------------

    if st.button(
        "▶ Run Prediction",
        type="primary",
        use_container_width=True,
    ):

        battery_temp = clamp_battery_temp(
            float(battery_temp)
        )

        mass_flow_rate = (
            mass_flow_gs / 1000.0
        )

        time_vector = np.linspace(
            0,
            t_end,
            n_points,
        )

        with st.spinner(
            "Running PINN inference..."
        ):

            result, time_vector = (
                run_timeseries(
                    model,
                    device,
                    input_scaler,
                    output_scaler,
                    FIDX,
                    FEATURES,
                    pcm,
                    battery_temp,
                    mass_flow_rate,
                    time_vector,
                )
            )

        # ---------------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------------

        battery_temp_series = (
            result["battery_temp"]
            - 273.15
        )

        pcm_temp_series = (
            result["pcm_temp"]
            - 273.15
        )

        outlet_temp_series = (
            result["outlet_temp"]
            - 273.15
        )

        lf_series = result["lf"]

        peak_battery = np.max(
            battery_temp_series
        )

        peak_pcm = np.max(
            pcm_temp_series
        )

        peak_outlet = np.max(
            outlet_temp_series
        )

        final_lf = (
            lf_series[-1]
            * 100
        )

        st.markdown(
            "### Key Results"
        )

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            metric_card(
                "Peak Battery Temp",
                f"{peak_battery:.2f}",
                "°C"
            )

        with m2:
            metric_card(
                "Peak PCM Temp",
                f"{peak_pcm:.2f}",
                "°C"
            )

        with m3:
            metric_card(
                "Peak Outlet Temp",
                f"{peak_outlet:.2f}",
                "°C"
            )

        with m4:
            metric_card(
                "Final Liquid Fraction",
                f"{final_lf:.1f}",
                "%"
            )

        # ---------------------------------------------------------------------
        # Results DataFrame
        # ---------------------------------------------------------------------

        results_df = pd.DataFrame({

            "time_s":
                time_vector,

            "battery_temp_C":
                battery_temp_series,

            "pcm_temp_C":
                pcm_temp_series,

            "liquid_fraction":
                lf_series,

            "outlet_temp_C":
                outlet_temp_series,
        })

        # ---------------------------------------------------------------------
        # Enhanced 4-Panel Plot
        # ---------------------------------------------------------------------

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14, 8)
        )

        fig.suptitle(
            f"{pcm_name} | "
            f"T={battery_temp}°C | "
            f"ṁ={mass_flow_gs:.1f} g/s",
            fontsize=13,
            fontweight="bold"
        )

        plots = [

            (
                axes[0,0],
                battery_temp_series,
                "Battery Temperature (°C)",
                "#d62728"
            ),

            (
                axes[0,1],
                pcm_temp_series,
                "PCM Temperature (°C)",
                "#1f77b4"
            ),

            (
                axes[1,0],
                lf_series,
                "Liquid Fraction",
                "#2ca02c"
            ),

            (
                axes[1,1],
                outlet_temp_series,
                "Outlet Water Temp (°C)",
                "#ff7f0e"
            ),
        ]

        for ax, data, ylabel, color in plots:

            ax.plot(
                time_vector,
                data,
                lw=2,
                color=color
            )

            ax.fill_between(
                time_vector,
                data,
                alpha=0.10,
                color=color
            )

            ax.set_xlabel("Time (s)")
            ax.set_ylabel(ylabel)

            ax.grid(alpha=0.3)

        plt.tight_layout()

        st.pyplot(fig)

        # ---------------------------------------------------------------------
        # Downloads
        # ---------------------------------------------------------------------

        d1, d2 = st.columns(2)

        with d1:

            dataframe_download_button(
                results_df,
                "⬇ Download CSV",
                "timeseries_results.csv",
            )

        with d2:

            figure_download_button(
                fig,
                "⬇ Download Figure",
                "timeseries_plot.png",
            )

        plt.close(fig)

        # ---------------------------------------------------------------------
        # Data Table
        # ---------------------------------------------------------------------

        if show_raw_data:

            st.markdown(
                "### Raw Results"
            )

            st.dataframe(
                results_df,
                use_container_width=True,
            )

# =============================================================================
# END PART A2
# NEXT:
# PART A3
# - TAB 2 Optimal PCM
# - TAB 3 Optimal Flow
# - TAB 4 PCM Comparison
# =============================================================================
# =============================================================================
# TAB 2 — OPTIMAL PCM FINDER
# =============================================================================

with tab2:

    st.markdown(
        '<div class="section-header">Optimal PCM Finder</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Differential Evolution optimization over PCM property space."
    )

    c1, c2, c3 = st.columns(3)
    

    with c1:

        opt_battery_temp = st.selectbox(
            "Battery Temperature (°C)",
            [50, 70, 90],
            key="opt_pcm_temp"
        )

        opt_flow_gs = st.slider(
            "Mass Flow Rate (g/s)",
            1.0,
            10.0,
            5.0,
            0.5,
            key="opt_pcm_flow"
        )

    with c2:

        target_temp = st.number_input(
            "Target Max Battery Temperature (°C)",
            30.0,
            100.0,
            70.0,
            1.0,
            key="opt_pcm_target_temp"
        )

        de_iterations = st.slider(
            "DE Iterations",
            20,
            300,
            100,
        )

    with c3:

        de_population = st.slider(
            "Population Size",
            8,
            30,
            12,
        )

        optimization_seed = st.number_input(
            "Random Seed",
            value=42,
            key="opt_pcm_seed"
        )

    if st.button(
        "🔍 Find Optimal PCM",
        type="primary",
        use_container_width=True,
    ):

        from scipy.optimize import differential_evolution

        eval_times = np.array([
            150,
            300,
            450,
            600,
            750,
            900,
        ])

        battery_temp = clamp_battery_temp(
            float(opt_battery_temp)
        )

        flow_rate = (
            opt_flow_gs / 1000
        )

        bounds = [
            (700, 1000),
            (1500, 3000),
            (0.10, 0.60),
            (0.001, 0.010),
            (165000, 200000),
            (299.15, 341.15),
            (1.0, 10.0),
        ]

        progress = st.progress(0)

        iteration_counter = {
            "count": 0
        }

        def objective(x):

            density = x[0]
            cp = x[1]
            k = x[2]
            viscosity = x[3]
            latent = x[4]
            tsolidus = x[5]
            melt_range = x[6]

            tliquidus = (
                tsolidus
                + melt_range
            )

            rows = np.array([
                build_feature_row(
                    FIDX,
                    FEATURES,
                    density,
                    cp,
                    k,
                    viscosity,
                    latent,
                    tsolidus,
                    tliquidus,
                    flow_rate,
                    battery_temp,
                    t,
                )
                for t in eval_times
            ])

            result = predict_batch(
                model,
                device,
                input_scaler,
                output_scaler,
                rows,
            )

            Tmax = (
                result["battery_temp"]
                - 273.15
            ).max()

            penalty = (
                1000
                * max(
                    0,
                    Tmax - target_temp
                ) ** 2
            )

            return Tmax + penalty

        def callback(xk, convergence):

            iteration_counter["count"] += 1

            pct = int(
                (
                    iteration_counter["count"]
                    / de_iterations
                )
                * 100
            )

            progress.progress(
                min(pct, 100)
            )

        with st.spinner(
            "Running optimization..."
        ):

            result = differential_evolution(
                objective,
                bounds,
                maxiter=de_iterations,
                popsize=de_population,
                seed=optimization_seed,
                callback=callback,
                polish=True,
            )

        progress.progress(100)

        best = result.x

        density = best[0]
        cp = best[1]
        k = best[2]
        viscosity = best[3]
        latent = best[4]
        tsolidus = best[5]
        melt_range = best[6]

        tliquidus = (
            tsolidus
            + melt_range
        )

        tmelt = (
            tsolidus
            + tliquidus
        ) / 2

        st.session_state.best_pcm = {

            "density": density,
            "cp": cp,
            "k": k,
            "viscosity": viscosity,
            "latent_heat": latent,
            "tsolidus": tsolidus,
            "tliquidus": tliquidus,
        }

        # r1, r2, r3, r4 = st.columns(4)

        # with r1:
        #     metric_card(
        #         "Density",
        #         f"{density:.1f}",
        #         "kg/m³"
        #     )

        # with r2:
        #     metric_card(
        #         "Cp",
        #         f"{cp:.1f}",
        #         "J/kgK"
        #     )

        # with r3:
        #     metric_card(
        #         "Latent Heat",
        #         f"{latent/1000:.1f}",
        #         "kJ/kg"
        #     )

        # with r4:
        #     metric_card(
        #         "Tmelt",
        #         f"{tmelt-273.15:.1f}",
        #         "°C"
        #     )

        # optimal_pcm_df = pd.DataFrame([{
        #     "density": density,
        #     "cp": cp,
        #     "k": k,
        #     "viscosity": viscosity,
        #     "latent_heat": latent,
        #     "tsolidus": tsolidus,
        #     "tliquidus": tliquidus,
        #     "tmelt_C": tmelt - 273.15,
        # }])

        # dataframe_download_button(
        #     optimal_pcm_df,
        #     "⬇ Download Optimal PCM",
        #     "optimal_pcm.csv",
        # )

        optimized_pcm = st.session_state.best_pcm

        time_vector = np.linspace(
            0,
            900,
            181
        )

        optimized_result, _ = run_timeseries(

            model,
            device,

            input_scaler,
            output_scaler,

            FIDX,
            FEATURES,

            optimized_pcm,

            battery_temp,
            flow_rate,

            time_vector,
        )

        battery_series = (
            optimized_result["battery_temp"]
            - 273.15
        )

        pcm_series = (
            optimized_result["pcm_temp"]
            - 273.15
        )

        lf_series = (
            optimized_result["lf"]
        )

        outlet_series = (
            optimized_result["outlet_temp"]
            - 273.15
        )

        peak_battery = battery_series.max()

        peak_pcm = pcm_series.max()

        final_lf = lf_series[-1]

        peak_outlet = outlet_series.max()

        st.markdown("## Optimized PCM Dashboard")

        # ============================================================
        # REAL PCM COMPARISON
        # ============================================================

        Tm_opt = tmelt

        closest_name = min(
            PCM_DATABASE,
            key=lambda n: abs(
                (
                    PCM_DATABASE[n]["tsolidus"]
                    +
                    PCM_DATABASE[n]["tliquidus"]
                ) / 2
                - Tm_opt
            )
        )

        closest_pcm = PCM_DATABASE[
            closest_name
        ]

        closest_tm = (
            closest_pcm["tsolidus"]
            +
            closest_pcm["tliquidus"]
        ) / 2 - 273.15

        st.markdown(
            f"""
            ### Closest Commercial PCM

            **{closest_name}**
            (Tmelt ≈ {closest_tm:.1f} °C)
            """
        )

        # ============================================================
        # RUN COMPARISON
        # ============================================================

        time_vec = np.linspace(
            0,
            900,
            181
        )

        opt_pcm_props = {

            "density":
                density,

            "cp":
                cp,

            "k":
                k,

            "viscosity":
                viscosity,

            "latent_heat":
                latent,

            "tsolidus":
                tsolidus,

            "tliquidus":
                tliquidus,
        }

        res_opt, _ = run_timeseries(

            model,
            device,

            input_scaler,
            output_scaler,

            FIDX,
            FEATURES,

            opt_pcm_props,

            battery_temp,
            flow_rate,

            time_vec,
        )

        res_std, _ = run_timeseries(

            model,
            device,

            input_scaler,
            output_scaler,

            FIDX,
            FEATURES,

            closest_pcm,

            battery_temp,
            flow_rate,

            time_vec,
        )

        peak_opt = (
            res_opt["battery_temp"]
            - 273.15
        ).max()

        peak_std = (
            res_std["battery_temp"]
            - 273.15
        ).max()
        improvement = (
            peak_std
            -
            peak_opt
        )

        if peak_opt < target_temp:

            box_class = "ok-box"

        else:

            box_class = "warn-box"

        # st.markdown(
        #     f"""
        #     <div class="{box_class}">
        #     <b>Optimal PCM</b>:
        #     {peak_opt:.2f} °C

        #     &nbsp;&nbsp;|&nbsp;&nbsp;

        #     <b>{closest_name}</b>:
        #     {peak_std:.2f} °C

        #     &nbsp;&nbsp;|&nbsp;&nbsp;

        #     Improvement:
        #     <b>{improvement:.2f} °C</b>

        #     &nbsp;&nbsp;|&nbsp;&nbsp;

        #     Target:
        #     <b>{target_temp:.2f} °C</b>
        #     </div>
        #     """,
        #     unsafe_allow_html=True
        # )
        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "Optimal PCM Tmax",
                f"{peak_opt:.2f} °C"
            )

        with m2:

            st.metric(
                f"{closest_name} Tmax",
                f"{peak_std:.2f} °C"
            )

        with m3:

            st.metric(
                "Improvement",
                f"{improvement:.2f} °C"
            )
        fig_cmp, ax = plt.subplots(
            figsize=(12,5)
        )

        opt_temp = (
            res_opt["battery_temp"]
            - 273.15
        )

        std_temp = (
            res_std["battery_temp"]
            - 273.15
        )

        ax.plot(

            time_vec,

            opt_temp,

            color="red",

            linewidth=2.5,

            label="Optimized PCM"
        )

        ax.plot(

            time_vec,

            std_temp,

            color="blue",

            linestyle="--",

            linewidth=2,

            label=closest_name
        )

        ax.fill_between(

            time_vec,

            opt_temp,

            alpha=0.08,

            color="red"
        )

        ax.fill_between(

            time_vec,

            std_temp,

            alpha=0.05,

            color="blue"
        )

        ax.axhline(

            target_temp,

            color="black",

            linestyle=":",

            linewidth=2,

            label=f"Target ({target_temp:.0f}°C)"
        )

        ax.set_xlabel(
            "Time (s)"
        )

        ax.set_ylabel(
            "Battery Temperature (°C)"
        )

        ax.set_title(

            f"Optimized PCM vs {closest_name}"
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend()

        plt.tight_layout()

        st.pyplot(fig_cmp)

        plt.close(fig_cmp)

        # row1 = st.columns(4)

        # with row1[0]:
        #     st.metric(
        #         "Density",
        #         f"{density:.1f} kg/m³"
        #     )

        # with row1[1]:
        #     st.metric(
        #         "Cp",
        #         f"{cp:.0f} J/kgK"
        #     )

        # with row1[2]:
        #     st.metric(
        #         "Conductivity",
        #         f"{k:.3f} W/mK"
        #     )

        # with row1[3]:
        #     st.metric(
        #         "Latent Heat",
        #         f"{latent/1000:.1f} kJ/kg"
        #     )

        # row2 = st.columns(4)

        # with row2[0]:
        #     st.metric(
        #         "Tmelt",
        #         f"{tmelt-273.15:.2f} °C"
        #     )

        # with row2[1]:
        #     st.metric(
        #         "Melt Range",
        #         f"{melt_range:.2f} K"
        #     )

        # with row2[2]:
        #     st.metric(
        #         "Viscosity",
        #         f"{viscosity:.4f}"
        #     )

        # with row2[3]:
        #     st.metric(
        #         "Peak Battery Temp",
        #         f"{peak_battery:.2f} °C"
        #     )
        # comparison_rows = []
        # for pcm_name, pcm in PCM_DATABASE.items():

        #     result_pcm, _ = run_timeseries(

        #         model,
        #         device,

        #         input_scaler,
        #         output_scaler,

        #         FIDX,
        #         FEATURES,

        #         pcm,

        #         battery_temp,
        #         flow_rate,

        #         time_vector,
        #     )

        # comparison_rows.append({

        #         "PCM":
        #             pcm_name,

        #         "Peak Battery Temp":
        #             (
        #                 result_pcm["battery_temp"]
        #                 - 273.15
        #             ).max(),

        #         "Final LF":
        #             result_pcm["lf"][-1],
        #     })
        # comparison_rows.append({

        #         "PCM":
        #             "OPTIMIZED",

        #         "Peak Battery Temp":
        #             peak_battery,

        #         "Final LF":
        #             final_lf,
        #     })
        # comparison_df = pd.DataFrame(
        #         comparison_rows
        #     )
        # comparison_df = (
        #         comparison_df
        #         .sort_values(
        #             "Peak Battery Temp"
        #         )
        #     )
        # st.markdown(
        #         "## PCM Leaderboard"
        #     )

        # st.dataframe(
        #         comparison_df,
        #         use_container_width=True
        #     )
        # fig_rank = px.bar(

        #         comparison_df,

        #         x="PCM",

        #         y="Peak Battery Temp",

        #         color="Peak Battery Temp",

        #         title=
        #         "Peak Battery Temperature Ranking"
        #     )

        # st.plotly_chart(
        #         fig_rank,
        #         use_container_width=True,
        #         # key="pcm_ranking_plot"
        #     )
        # material_df = pcm_df.copy()
        # material_rows = []
        # for name, pcm in PCM_DATABASE.items():

        #         material_rows.append({

        #             "PCM": name,

        #             "Latent":
        #                 pcm["latent_heat"],

        #             "k":
        #                 pcm["k"]
        #         })
        # material_df = pd.DataFrame(
        #         material_rows
        #     )

        # fig_space = px.scatter(

        #         material_df,

        #         x="Latent",

        #         y="k",

        #         text="PCM",

        #         title=
        #         "Material Space"
        #     )
        # fig_space.add_scatter(

        #         x=[latent],

        #         y=[k],

        #         mode="markers",

        #         marker=dict(
        #             size=18,
        #             symbol="star"
        #         ),

        #         name="OPTIMIZED"
        #     )
        # st.plotly_chart(
        #         fig_space,
        #         use_container_width=True,
        #         key="optimal_pcm_material_space"

        #     )
        # st.markdown(
        #         "## Optimized PCM Thermal Response"
        #     )
        # fig, axes = plt.subplots(
        #         2,
        #         2,
        #         figsize=(14,8)
        #     )
        # plots = [

        #         (
        #             axes[0,0],
        #             battery_series,
        #             "Battery Temperature (°C)",
        #             "#d62728"
        #         ),

        #         (
        #             axes[0,1],
        #             pcm_series,
        #             "PCM Temperature (°C)",
        #             "#1f77b4"
        #         ),

        #         (
        #             axes[1,0],
        #             lf_series,
        #             "Liquid Fraction",
        #             "#2ca02c"
        #         ),

        #         (
        #             axes[1,1],
        #             outlet_series,
        #             "Outlet Temperature (°C)",
        #             "#ff7f0e"
        #         ),
        #     ]

        # for ax, data, ylabel, color in plots:

        #         ax.plot(
        #             time_vector,
        #             data,
        #             lw=2,
        #             color=color
        #         )

        #         ax.fill_between(
        #             time_vector,
        #             data,
        #             alpha=0.10,
        #             color=color
        #         )

        #         ax.set_xlabel("Time (s)")
        #         ax.set_ylabel(ylabel)

        #         ax.grid(
        #             alpha=0.3
        #         )

        # plt.tight_layout()
        # st.pyplot(fig)


# =============================================================================
# TAB 3 — OPTIMAL FLOW RATE
# =============================================================================

with tab3:

    st.markdown(
        '<div class="section-header">Optimal Flow Rate Finder</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        flow_battery_temp = st.selectbox(
            "Battery Temperature (°C)",
            [50, 70, 90],
            key="flow_temp"
        )

        flow_pcm = st.selectbox(
            "PCM",
            list(PCM_DATABASE.keys()),
            key="flow_pcm"
        )

    with c2:

        target_temp = st.number_input(
            "Target Max Temperature (°C)",
            30.0,
            100.0,
            45.0,
            key="flow_target_temp"
        )

    with c3:

        sweep_points = st.slider(
            "Sweep Resolution",
            10,
            50,
            30,
        )

    if st.button(
        "💧 Optimize Flow Rate",
        type="primary",
        use_container_width=True,
    ):

        pcm = PCM_DATABASE[
            flow_pcm
        ]

        battery_temp = clamp_battery_temp(
            flow_battery_temp
        )

        eval_times = np.array([
            150,
            300,
            450,
            600,
            750,
            900,
        ])

        def flow_objective(mdot_arr):

            mdot = mdot_arr[0]

            rows = np.array([
                build_feature_row(
                    FIDX,
                    FEATURES,
                    pcm["density"],
                    pcm["cp"],
                    pcm["k"],
                    pcm["viscosity"],
                    pcm["latent_heat"],
                    pcm["tsolidus"],
                    pcm["tliquidus"],
                    mdot,
                    battery_temp,
                    t,
                )
                for t in eval_times
            ])

            result = predict_batch(
                model,
                device,
                input_scaler,
                output_scaler,
                rows,
            )

            Tmax = (
                result["battery_temp"]
                - 273.15
            ).max()

            penalty = (
                5000
                * max(
                    0,
                    Tmax - target_temp
                ) ** 2
            )

            return mdot + penalty

        result = minimize(
            flow_objective,
            x0=[5e-3],
            bounds=[
                (
                    FLOW_LIMITS["min"],
                    FLOW_LIMITS["max"]
                )
            ],
            method="L-BFGS-B",
        )

        optimal_flow = result.x[0]

        st.session_state.best_flow = (
            optimal_flow
        )

        metric_card(
            "Optimal Flow Rate",
            f"{optimal_flow*1000:.2f}",
            "g/s"
        )

        st.markdown(
            "## Flow Rate Sensitivity Analysis"
        )
        flow_sweep = np.linspace(
            FLOW_LIMITS["min"],
            FLOW_LIMITS["max"],
            25
        )

        sweep_rows = []

        progress = st.progress(
            0,
            text="Evaluating flow sweep..."
        )
        for i, mdot_sw in enumerate(flow_sweep):

            rows_sw = np.array([

                build_feature_row(

                    FIDX,
                    FEATURES,

                    pcm["density"],
                    pcm["cp"],
                    pcm["k"],
                    pcm["viscosity"],
                    pcm["latent_heat"],
                    pcm["tsolidus"],
                    pcm["tliquidus"],

                    mdot_sw,

                    battery_temp,

                    t

                )

                for t in eval_times
            ])

            res_sw = predict_batch(

                model,
                device,

                input_scaler,
                output_scaler,

                rows_sw,
            )

            sweep_rows.append({

                "Flow_g_s":
                    mdot_sw * 1000,

                "Peak_Battery_Temp":
                    (
                        res_sw["battery_temp"]
                        - 273.15
                    ).max(),
            })

            progress.progress(
                int(
                    (i+1)
                    /
                    len(flow_sweep)
                    * 100
                )
            )
        sweep_df = pd.DataFrame(
                sweep_rows
            )
        time_vector = np.linspace(
            0,
            900,
            181
        )

        result_opt, _ = run_timeseries(
            model,
            device,
            input_scaler,
            output_scaler,
            FIDX,
            FEATURES,
            pcm,
            battery_temp,
            optimal_flow,
            time_vector,
        )
        peak_battery = (
            result_opt["battery_temp"]
            - 273.15
        ).max()
        margin = (
            target_temp
            -
            peak_battery
        )
        fig, axes = plt.subplots(
                1,
                2,
                figsize=(14,5)
            )
        ax = axes[0]

        ax.plot(

                sweep_df["Flow_g_s"],

                sweep_df["Peak_Battery_Temp"],

                "b-o",

                lw=2,
                ms=5,
            )

        ax.fill_between(

                sweep_df["Flow_g_s"],

                sweep_df["Peak_Battery_Temp"],

                alpha=0.08,
            )

        ax.axhline(

                target_temp,

                color="red",

                linestyle="--",

                linewidth=2,

                label=f"Limit ({target_temp}°C)"
            )

        ax.axvline(

                optimal_flow*1000,

                color="green",

                linestyle="-.",

                linewidth=2,

                label=f"Optimal ({optimal_flow*1000:.2f} g/s)"
            )

        ax.set_title(
                "Flow Rate Sweep"
            )

        ax.set_xlabel(
                "Mass Flow Rate (g/s)"
            )

        ax.set_ylabel(
                "Peak Battery Temperature (°C)"
            )

        ax.grid(alpha=0.3)

        ax.legend()
        ax = axes[1]

        battery_curve = (
                result_opt["battery_temp"]
                - 273.15
            )

        ax.plot(

                time_vector,

                battery_curve,

                color="green",

                lw=2.5,
            )

        ax.fill_between(

                time_vector,

                battery_curve,

                alpha=0.1,

                color="green"
            )

        ax.axhline(

                target_temp,

                color="red",

                linestyle="--",

                linewidth=2,

                label=f"Limit ({target_temp}°C)"
            )

        ax.set_title(
                f"Thermal Response @ {optimal_flow*1000:.2f} g/s"
            )

        ax.set_xlabel(
             "Time (s)"
            )

        ax.set_ylabel(
                "Battery Temperature (°C)"
            )

        ax.grid(alpha=0.3)

        ax.legend()
        if peak_battery <= target_temp:

                # st.markdown(
                #     f"""
                #     <div class="ok-box">
                #     Target satisfied.

                #     Optimal Flow:
                #     <b>{optimal_flow*1000:.2f} g/s</b>

                #     Safety Margin:
                #     <b>{margin:.2f} °C</b>
                #     </div>
                #     """,
                #     unsafe_allow_html=True
                # )
                st.markdown(
    f"""
    <div style="
        background-color:#d4edda;
        color:#155724;
        padding:15px;
        border-radius:8px;
        border-left:5px solid #28a745;
        font-size:16px;
        font-weight:600;
    ">
        Target satisfied.<br><br>

        Optimal Flow:
        <b>{optimal_flow*1000:.2f} g/s</b>

        <br><br>

        Safety Margin:
        {margin:.2f} °C
    </div>
    """,
    unsafe_allow_html=True
)

        else:

                # st.markdown(
                #     f"""
                #     <div class="warn-box">
                #     Thermal target not achieved.

                #     Peak Battery Temperature:
                #     <b>{peak_battery:.2f} °C</b>
                #     </div>
                #     """,
                #     unsafe_allow_html=True
                # )
                st.markdown(
    f"""
    <div style="
        background-color:#fff3cd;
        color:#856404;
        padding:15px;
        border-radius:8px;
        border-left:5px solid #ffc107;
        font-size:16px;
        font-weight:600;
    ">
        Thermal target not achieved.<br><br>

        Peak Battery Temperature:
        {peak_battery:.2f} °C
    </div>
    """,
    unsafe_allow_html=True
)
                st.metric(
                    "Flow Reduction vs 10 g/s",
                    f"{(1 - optimal_flow/FLOW_LIMITS['max'])*100:.1f}%"
                )
        st.pyplot(fig)
# =============================================================================
# TAB 4 — PCM COMPARISON
# =============================================================================

with tab4:

    st.markdown(
        '<div class="section-header">PCM Comparison</div>',
        unsafe_allow_html=True
    )

    cmp_col1, cmp_col2 = st.columns(2)

    with cmp_col1:

        cmp_temp = st.selectbox(
            "Battery Temperature (°C)",
            [50, 70, 90],
            key="cmp_temp"
        )

    with cmp_col2:

        cmp_flow = st.slider(
            "Mass Flow Rate (g/s)",
            1.0,
            10.0,
            5.0,
            0.5,
            key="cmp_flow"
        )

    if st.button(
        "📊 Compare PCMs",
        type="primary",
        use_container_width=True,
    ):

        battery_temp = clamp_battery_temp(
            cmp_temp
        )

        flow_rate = (
            cmp_flow / 1000
        )

        time_vector = np.linspace(
            0,
            900,
            181,
        )

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14, 8)
        )

        summary_rows = []

        for pcm_name, pcm in PCM_DATABASE.items():

            result, _ = run_timeseries(
                model,
                device,
                input_scaler,
                output_scaler,
                FIDX,
                FEATURES,
                pcm,
                battery_temp,
                flow_rate,
                time_vector,
            )

            battery = (
                result["battery_temp"]
                - 273.15
            )

            pcm_temp = (
                result["pcm_temp"]
                - 273.15
            )

            outlet = (
                result["outlet_temp"]
                - 273.15
            )

            lf = result["lf"]

            axes[0,0].plot(
                time_vector,
                battery,
                lw=2,
                label=pcm_name
            )

            axes[0,0].axhline(
                60,
                color="black",
                linestyle="--",
                linewidth=2,
                label="Target Limit"
            )

            axes[0,1].plot(
                time_vector,
                pcm_temp,
                label=pcm_name
            )

            axes[1,0].plot(
                time_vector,
                lf,
                label=pcm_name
            )

            axes[1,1].plot(
                time_vector,
                outlet,
                label=pcm_name
            )

            summary_rows.append({

                "PCM":
                    pcm_name,

                "Peak Battery Temp":
                    battery.max(),

                "Peak PCM Temp":
                    pcm_temp.max(),

                "Final LF":
                    lf[-1],

                "Peak Outlet Temp":
                    outlet.max(),
            })

        axes[0,0].set_title(
            "Battery Temperature"
        )

        axes[0,1].set_title(
            "PCM Temperature"
        )

        axes[1,0].set_title(
            "Liquid Fraction"
        )

        axes[1,1].set_title(
            "Outlet Temperature"
        )

        for ax in axes.flatten():

            ax.grid(True)

            ax.legend(
                fontsize=8
            )

        plt.tight_layout()

        st.pyplot(fig)

        summary_df = pd.DataFrame(
            summary_rows
        )
        summary_df["Score"] = (

            0.5 * (
                summary_df["Peak Battery Temp"].max()
                -
                summary_df["Peak Battery Temp"]
            )

            +

            0.3 * (
                summary_df["Final LF"]
            )

            +

            0.2 * (
                summary_df["Peak Outlet Temp"].max()
                -
                summary_df["Peak Outlet Temp"]
            )
        )

        summary_df = (
            summary_df
            .sort_values(
                "Score",
                ascending=False
            )
            .reset_index(drop=True)
        )

        summary_df.index += 1

        # st.markdown(
        #     "## PCM Ranking"
        # )

        # st.dataframe(
        #     summary_df,
        #     use_container_width=True
        # )

        st.dataframe(
            summary_df,
            use_container_width=True
        )

        dataframe_download_button(
            summary_df,
            "⬇ Download Comparison",
            "pcm_comparison.csv"
        )

        figure_download_button(
            fig,
            "⬇ Download Figure",
            "pcm_comparison.png"
        )

        fig_rank = px.bar(

            summary_df,

            x="PCM",

            y="Score",

            color="Score",

            title="PCM Performance Ranking"
        )

        st.plotly_chart(
            fig_rank,
            use_container_width=True,
            key="idkwhattosay"
        )
        radar_df = summary_df.head(4)

        fig_radar = go.Figure()

        for _, row in radar_df.iterrows():

            fig_radar.add_trace(

                go.Scatterpolar(

                    r=[

                        row["Score"],

                        row["Final LF"],

                        100-row["Peak Battery Temp"],

                        100-row["Peak Outlet Temp"],
                    ],

                    theta=[

                        "Score",

                        "LF",

                        "Battery",

                        "Outlet",
                    ],

                    fill="toself",

                    name=row["PCM"],
                )
            )

        st.plotly_chart(
            fig_radar,
            use_container_width=True,
            key="radar_plot"
        )

        plt.close(fig)

# =============================================================================
# END PART A3
# NEXT:
# PART B1
# TAB 5 CFD VALIDATION
# =============================================================================
# =============================================================================
# TAB 5 — CFD VALIDATION
# =============================================================================

with tab5:

    st.markdown(
        '<div class="section-header">CFD Validation</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Validate PINN predictions against CFD ground-truth database."
    )

    # -------------------------------------------------------------------------
    # Validation DB Check
    # -------------------------------------------------------------------------

    if not os.path.exists(
        VALIDATION_DB_PATH
    ):

        st.error(
            f"Validation database not found:\n\n{VALIDATION_DB_PATH}"
        )

    else:

        # ---------------------------------------------------------------------
        # Load Database
        # ---------------------------------------------------------------------

        @st.cache_data
        def cached_validation_db():

            return load_validation_database(
                VALIDATION_DB_PATH
            )

        validation_db = (
            cached_validation_db()
        )

        st.success(
            f"Loaded {len(validation_db):,} CFD records"
        )

        # ---------------------------------------------------------------------
        # Sidebar Filters
        # ---------------------------------------------------------------------

        st.markdown("### Validation Filters")

        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:

            if "PCM_Name" in validation_db.columns:

                pcm_options = (
                    ["All"]
                    +
                    sorted(
                        validation_db[
                            "PCM_Name"
                        ]
                        .dropna()
                        .unique()
                        .tolist()
                    )
                )

                selected_pcm = st.selectbox(
                    "PCM",
                    pcm_options,
                )

            else:

                selected_pcm = "All"

        with filter_col2:

            if (
                "Initial_PCM_Temperature_C"
                in validation_db.columns
            ):

                temp_options = (
                    ["All"]
                    +
                    sorted(
                        validation_db[
                            "Initial_PCM_Temperature_C"
                        ]
                        .unique()
                        .tolist()
                    )
                )

                selected_temp = st.selectbox(
                    "Temperature",
                    temp_options,
                )

            else:

                selected_temp = "All"

        with filter_col3:

            if (
                "Mass_Flow_Rate_kg_s"
                in validation_db.columns
            ):

                flow_options = (
                    ["All"]
                    +
                    sorted(
                        validation_db[
                            "Mass_Flow_Rate_kg_s"
                        ]
                        .round(6)
                        .unique()
                        .tolist()
                    )
                )

                selected_flow = st.selectbox(
                    "Flow Rate (kg/s)",
                    flow_options,
                )

            else:

                selected_flow = "All"

        # ---------------------------------------------------------------------
        # Run Validation
        # ---------------------------------------------------------------------

        if st.button(
            "📊 Run Validation",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Running PINN on CFD database..."
            ):

                validation_results = (
                    run_validation(
                        validation_db,
                        model,
                        device,
                        input_scaler,
                        output_scaler,
                        FIDX,
                        FEATURES,
                    )
                )

                validation_results = (
                    add_error_columns(
                        validation_results
                    )
                )

                st.session_state[
                    "validation_results"
                ] = validation_results

            st.success(
                "Validation complete."
            )

        # ---------------------------------------------------------------------
        # Results Section
        # ---------------------------------------------------------------------

        if (
            st.session_state[
                "validation_results"
            ]
            is not None
        ):

            validation_results = (
                st.session_state[
                    "validation_results"
                ]
            )



            metrics = validation_summary(
                validation_results
            )

            # -------------------------------------------------------------
            # Metrics Dashboard
            # -------------------------------------------------------------

            st.markdown(
                "## Overall Validation Metrics"
            )

            metric_tabs = st.tabs([
                "Battery",
                "PCM",
                "LF",
                "Outlet"
            ])

            names = [
                "battery",
                "pcm",
                "lf",
                "outlet"
            ]

            for tab_obj, key in zip(
                metric_tabs,
                names
            ):

                with tab_obj:

                    m = metrics[key]

                    c1, c2, c3, c4 = (
                        st.columns(4)
                    )

                    with c1:
                        metric_card(
                            "MAE",
                            f"{m['MAE']:.4f}"
                        )

                    with c2:
                        metric_card(
                            "RMSE",
                            f"{m['RMSE']:.4f}"
                        )

                    with c3:
                        metric_card(
                            "R²",
                            f"{m['R2']:.4f}"
                        )

                    with c4:
                        metric_card(
                            "MAPE",
                            f"{m['MAPE']:.2f}",
                            "%"
                        )

            # -------------------------------------------------------------
            # Parity Plots
            # -------------------------------------------------------------

            # st.markdown(
            #     "## Parity Plots"
            # )

            # parity_fig, axes = plt.subplots(
            #     2,
            #     2,
            #     figsize=(12, 10)
            # )

            # parity_map = [

            #     (
            #         "battery-temp",
            #         "pinn_battery_temp_K",
            #         "Battery Temperature"
            #     ),

            #     (
            #         "pcmtemp",
            #         "pinn_pcm_temp_K",
            #         "PCM Temperature"
            #     ),

            #     (
            #         "lf",
            #         "pinn_lf",
            #         "Liquid Fraction"
            #     ),

            #     (
            #         "outlet-temp",
            #         "pinn_outlet_temp_K",
            #         "Outlet Temperature"
            #     ),
            # ]

            # for ax, (
            #     true_col,
            #     pred_col,
            #     title
            # ) in zip(
            #     axes.flatten(),
            #     parity_map
            # ):

            #     if (
            #         true_col
            #         not in validation_results.columns
            #     ):
            #         continue

            #     x = validation_results[
            #         true_col
            #     ]

            #     y = validation_results[
            #         pred_col
            #     ]

            #     ax.scatter(
            #         x,
            #         y,
            #         alpha=0.5,
            #         s=10
            #     )

            #     mn = min(
            #         x.min(),
            #         y.min()
            #     )

            #     mx = max(
            #         x.max(),
            #         y.max()
            #     )

            #     ax.plot(
            #         [mn, mx],
            #         [mn, mx],
            #         "r--"
            #     )

            #     ax.set_title(
            #         title
            #     )

            #     ax.set_xlabel(
            #         "CFD"
            #     )

            #     ax.set_ylabel(
            #         "PINN"
            #     )

            #     ax.grid(True)

            # plt.tight_layout()

            # st.pyplot(
            #     parity_fig
            # )
                        
            # -------------------------------------------------------------
            # PCM-Wise Validation Analysis
            # -------------------------------------------------------------

            pcm_column = None

            possible_pcm_columns = [
                "PCM_Name",
                "pcm_name",
                "PCM",
                "Material",
            ]

            for col in possible_pcm_columns:

                if col in validation_results.columns:
                    pcm_column = col
                    break

            if pcm_column is not None:

                st.markdown(
                    "## PCM-wise Validation Analysis"
                )

                pcm_metrics = []

                for pcm in sorted(
                    validation_results[pcm_column]
                    .dropna()
                    .unique()
                ):

                    df_pcm = validation_results[
                        validation_results[
                            pcm_column
                        ] == pcm
                    ]

                    row = {
                        "PCM": pcm
                    }

                    try:
                        row["Battery MAE"] = mean_absolute_error(
                            df_pcm["battery-temp"],
                            df_pcm["pinn_battery_temp_K"]
                        )

                        row["Battery R²"] = r2_score(
                            df_pcm["battery-temp"],
                            df_pcm["pinn_battery_temp_K"]
                        )
                    except:
                        row["Battery MAE"] = np.nan
                        row["Battery R²"] = np.nan

                    try:
                        row["LF MAE"] = mean_absolute_error(
                            df_pcm["lf"],
                            df_pcm["pinn_lf"]
                        )
                    except:
                        row["LF MAE"] = np.nan

                    pcm_metrics.append(row)

                pcm_metrics_df = pd.DataFrame(
                    pcm_metrics
                )

                st.dataframe(
                    pcm_metrics_df,
                    use_container_width=True
                )

                # ---------------------------------------------------------
                # Enhanced Parity Plots
                # ---------------------------------------------------------

                st.markdown(
                    "## PCM-Coloured Parity Plots"
                )

                fig, axes = plt.subplots(
                    2,
                    2,
                    figsize=(14,10)
                )

                parity_map = [

                    (
                        "battery-temp",
                        "pinn_battery_temp_K",
                        "Battery Temperature"
                    ),

                    (
                        "pcmtemp",
                        "pinn_pcm_temp_K",
                        "PCM Temperature"
                    ),

                    (
                        "lf",
                        "pinn_lf",
                        "Liquid Fraction"
                    ),

                    (
                        "outlet-temp",
                        "pinn_outlet_temp_K",
                        "Outlet Temperature"
                    ),
                ]

                for ax, (
                    true_col,
                    pred_col,
                    title
                ) in zip(
                    axes.flatten(),
                    parity_map
                ):

                    if (
                        true_col not in validation_results.columns
                        or
                        pred_col not in validation_results.columns
                    ):
                        continue

                    for pcm in sorted(
                        validation_results[
                            pcm_column
                        ]
                        .dropna()
                        .unique()
                    ):

                        subset = validation_results[
                            validation_results[
                                pcm_column
                            ] == pcm
                        ]

                        ax.scatter(
                            subset[true_col],
                            subset[pred_col],
                            label=str(pcm),
                            alpha=0.6,
                            s=25
                        )

                    mn = min(
                        validation_results[
                            true_col
                        ].min(),
                        validation_results[
                            pred_col
                        ].min()
                    )

                    mx = max(
                        validation_results[
                            true_col
                        ].max(),
                        validation_results[
                            pred_col
                        ].max()
                    )

                    ax.plot(
                        [mn, mx],
                        [mn, mx],
                        "k--",
                        linewidth=2,
                    )

                    ax.set_title(title)

                    ax.set_xlabel(
                        "CFD"
                    )

                    ax.set_ylabel(
                        "PINN"
                    )

                    ax.grid(True)

                    ax.legend(
                        fontsize=8
                    )

                plt.tight_layout()

                st.pyplot(fig)

                plt.close(fig)

                # ---------------------------------------------------------
                # Error Distribution
                # ---------------------------------------------------------

                if "err_battery_K" in validation_results.columns:

                    st.markdown(
                        "## Error Distribution by PCM"
                    )

                    fig_err, ax = plt.subplots(
                        figsize=(10,5)
                    )

                    validation_results.boxplot(
                        column="err_battery_K",
                        by=pcm_column,
                        ax=ax
                    )

                    ax.set_title(
                        "Battery Temperature Error"
                    )

                    ax.set_ylabel(
                        "Absolute Error (K)"
                    )

                    plt.suptitle("")

                    st.pyplot(fig_err)

                    plt.close(fig_err)

            else:

                st.info(
                    "No PCM column found in validation dataset."
                )
            # -------------------------------------------------------------
            # Case Viewer
            # -------------------------------------------------------------

            st.markdown(
                "## Case Viewer"
            )

            df_case = validation_results

            if selected_pcm != "All":

                df_case = df_case[
                    df_case[
                        "PCM_Name"
                    ] == selected_pcm
                ]

            if selected_temp != "All":

                df_case = df_case[
                    df_case[
                        "Initial_PCM_Temperature_C"
                    ] == selected_temp
                ]

            if selected_flow != "All":

                df_case = df_case[
                    np.isclose(
                        df_case[
                            "Mass_Flow_Rate_kg_s"
                        ],
                        selected_flow
                    )
                ]

            if len(df_case):

                fig_case, ax = plt.subplots(
                    figsize=(10, 5)
                )

                ax.plot(
                    df_case["Time_s"],
                    df_case[
                        "battery-temp"
                    ],
                    label="CFD",
                    linewidth=2,
                )

                ax.plot(
                    df_case["Time_s"],
                    df_case[
                        "pinn_battery_temp_K"
                    ],
                    "--",
                    label="PINN",
                    linewidth=2,
                )

                ax.set_title(
                    "Battery Temperature Validation"
                )

                ax.set_xlabel(
                    "Time (s)"
                )

                ax.set_ylabel(
                    "Temperature"
                )

                ax.grid(True)

                ax.legend()

                st.pyplot(
                    fig_case
                )

            # -------------------------------------------------------------
            # Error Statistics
            # -------------------------------------------------------------

            st.markdown(
                "## Error Statistics"
            )

            error_cols = [

                "err_battery_K",
                "err_pcm_K",
                "err_lf",
                "err_outlet_K",
            ]

            available_error_cols = [
                c
                for c in error_cols
                if c in validation_results.columns
            ]

            if available_error_cols:

                st.dataframe(
                    validation_results[
                        available_error_cols
                    ]
                    .describe()
                    .T,
                    use_container_width=True,
                )

            # -------------------------------------------------------------
            # Download
            # -------------------------------------------------------------

            dataframe_download_button(
                validation_results,
                "⬇ Download Validation Results",
                "validation_results.csv",
            )

# =============================================================================
# END PART B1
# NEXT:
# PART B2
# TAB 6 — ADAPTIVE FLOW CONTROL
# =============================================================================
# =============================================================================
# TAB 6 — ADAPTIVE FLOW CONTROL
# =============================================================================

with tab6:

    st.markdown(
        '<div class="section-header">Adaptive Flow Rate Control</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Generate an adaptive coolant flow profile based on a changing battery load."
    )

    # -------------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------------

    left_col, right_col = st.columns([1, 1])

    with left_col:

        pcm_name = st.selectbox(
            "PCM Material",
            list(PCM_DATABASE.keys()),
            key="adaptive_pcm"
        )

        target_temp = st.number_input(
            "Target Maximum Battery Temperature (°C)",
            min_value=30.0,
            max_value=120.0,
            value=60.0,
            step=1.0,
            key="adaptive_target_temp"
        )

        look_ahead_seconds = st.slider(
            "Look-Ahead Horizon (s)",
            min_value=60,
            max_value=900,
            value=300,
            step=30,
        )

        interpolation_type = st.selectbox(
            "Interpolation",
            [
                "linear",
                "quadratic",
                "cubic"
            ],
            index=2,
        )

    with right_col:

        n_profile_points = st.slider(
            "Number of Profile Points",
            min_value=4,
            max_value=25,
            value=len(DEFAULT_PROFILE),
        )

        use_default_profile = st.checkbox(
            "Use Default Profile",
            value=True,
        )

    # -------------------------------------------------------------------------
    # Profile Editor
    # -------------------------------------------------------------------------

    st.markdown("### Battery Load Profile")

    if use_default_profile:

        profile_df = pd.DataFrame(
            DEFAULT_PROFILE,
            columns=[
                "Time_s",
                "Battery_Temp_C"
            ]
        )

    else:

        profile_df = pd.DataFrame({

            "Time_s":
                np.linspace(
                    0,
                    900,
                    n_profile_points
                ),

            "Battery_Temp_C":
                np.full(
                    n_profile_points,
                    50.0
                ),
        })

    profile_df = st.data_editor(
        profile_df,
        use_container_width=True,
        num_rows="dynamic",
        key="adaptive_profile_editor",
    )

    # -------------------------------------------------------------------------
    # Run Controller
    # -------------------------------------------------------------------------

    if st.button(
        "🎛 Run Adaptive Controller",
        type="primary",
        use_container_width=True,
    ):

        pcm = PCM_DATABASE[
            pcm_name
        ]

        profile_df = profile_df.sort_values(
            "Time_s"
        )

        profile = list(
            zip(
                profile_df["Time_s"],
                profile_df["Battery_Temp_C"]
            )
        )

        with st.spinner(
            "Computing optimal flow schedule..."
        ):

            controller_result = (
                run_adaptive_controller(
                    model=model,
                    device=device,
                    input_scaler=input_scaler,
                    output_scaler=output_scaler,
                    FIDX=FIDX,
                    FEATURES=FEATURES,
                    pcm=pcm,
                    temperature_profile=profile,
                    target_temp=target_temp,
                    look_ahead_seconds=look_ahead_seconds,
                )
            )

            summary = controller_summary(
                controller_result
            )

            continuous_signal = (
                build_continuous_controller(
                    controller_result[
                        "time_s"
                    ],
                    controller_result[
                        "required_flow_kg_s"
                    ],
                    interpolation=
                    interpolation_type,
                )
            )

        # ---------------------------------------------------------------------
        # Dashboard Metrics
        # ---------------------------------------------------------------------

        st.markdown(
            "### Controller Summary"
        )

        m1, m2, m3, m4 = st.columns(4)

        with m1:

            metric_card(
                "Average Flow",
                f"{summary['avg_flow_g_s']:.2f}",
                "g/s"
            )

        with m2:

            metric_card(
                "Peak Flow",
                f"{summary['max_flow_g_s']:.2f}",
                "g/s"
            )

        with m3:

            metric_card(
                "Peak Battery Temp",
                f"{summary['max_battery_temp_C']:.2f}",
                "°C"
            )

        with m4:

            metric_card(
                "Average LF",
                f"{summary['avg_lf']:.3f}"
            )

        # ---------------------------------------------------------------------
        # Results DataFrame
        # ---------------------------------------------------------------------

        controller_df = pd.DataFrame({

            "time_s":
                controller_result[
                    "time_s"
                ],

            "battery_temp_C":
                controller_result[
                    "battery_temp_C"
                ],

            "required_flow_g_s":
                controller_result[
                    "required_flow_g_s"
                ],

            "predicted_peak_temp_C":
                controller_result[
                    "predicted_peak_temp_C"
                ],

            "predicted_lf":
                controller_result[
                    "predicted_lf"
                ],
        })

        # ---------------------------------------------------------------------
        # Plotly Dashboard
        # ---------------------------------------------------------------------

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=controller_result[
                    "time_s"
                ],
                y=controller_result[
                    "battery_temp_C"
                ],
                name="Battery Temp",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=controller_result[
                    "time_s"
                ],
                y=controller_result[
                    "required_flow_g_s"
                ],
                name="Required Flow (g/s)",
                yaxis="y2",
            )
        )

        fig.update_layout(

            title=
            "Adaptive Control Response",

            xaxis_title=
            "Time (s)",

            yaxis=dict(
                title="Temperature (°C)"
            ),

            yaxis2=dict(
                title="Flow Rate (g/s)",
                overlaying="y",
                side="right"
            ),

            height=500,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="adaptive_controller_plot"
        )

        # ---------------------------------------------------------------------
        # Continuous Control Signal
        # ---------------------------------------------------------------------

        st.markdown(
            "### Continuous Flow Signal"
        )

        fig2 = px.line(

            x=continuous_signal[
                "time_s"
            ],

            y=continuous_signal[
                "flow_g_s"
            ],

            labels={
                "x": "Time (s)",
                "y": "Flow (g/s)"
            }
        )

        fig2.update_layout(
            title=
            f"{interpolation_type.title()} Interpolated Flow Signal",
            height=400,
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
            key="continuous_flow_signal"
        )

        # ---------------------------------------------------------------------
        # Multi-Panel Diagnostic Plot
        # ---------------------------------------------------------------------

        fig3, axes = plt.subplots(
            2,
            2,
            figsize=(14, 8)
        )

        axes[0, 0].plot(
            controller_result[
                "time_s"
            ],
            controller_result[
                "battery_temp_C"
            ]
        )

        axes[0, 0].set_title(
            "Battery Load Profile"
        )

        axes[0, 0].grid(True)

        axes[0, 1].plot(
            controller_result[
                "time_s"
            ],
            controller_result[
                "required_flow_g_s"
            ]
        )

        axes[0, 1].set_title(
            "Required Flow Rate"
        )

        axes[0, 1].grid(True)

        axes[1, 0].plot(
            controller_result[
                "time_s"
            ],
            controller_result[
                "predicted_peak_temp_C"
            ]
        )

        axes[1, 0].set_title(
            "Predicted Peak Temperature"
        )

        axes[1, 0].grid(True)

        axes[1, 1].plot(
            controller_result[
                "time_s"
            ],
            controller_result[
                "predicted_lf"
            ]
        )

        axes[1, 1].set_title(
            "Predicted Liquid Fraction"
        )

        axes[1, 1].grid(True)

        plt.tight_layout()

        st.pyplot(fig3)

        # ---------------------------------------------------------------------
        # Data
        # ---------------------------------------------------------------------

        st.markdown(
            "### Controller Table"
        )

        st.dataframe(
            controller_df,
            use_container_width=True,
        )

        # ---------------------------------------------------------------------
        # Downloads
        # ---------------------------------------------------------------------

        d1, d2 = st.columns(2)

        with d1:

            dataframe_download_button(
                controller_df,
                "⬇ Download Controller CSV",
                "adaptive_controller_results.csv",
            )

        with d2:

            figure_download_button(
                fig3,
                "⬇ Download Figure",
                "adaptive_controller.png",
            )

        plt.close(fig3)

# =============================================================================
# END PART B2
# NEXT:
# PART C1
# TAB 7 — JOINT PCM + FLOW OPTIMIZATION
# =============================================================================
# =============================================================================
# TAB 7 — JOINT PCM + FLOW OPTIMIZATION
# =============================================================================

with tab7:

    st.markdown(
        '<div class="section-header">Joint PCM + Flow Optimization</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Simultaneously optimize PCM properties and coolant flow rate."
    )

    # -------------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        joint_battery_temp = st.selectbox(
            "Battery Temperature (°C)",
            [50, 70, 90],
            key="joint_battery_temp"
        )

        joint_target_temp = st.number_input(
            "Target Maximum Temperature (°C)",
            min_value=30.0,
            max_value=120.0,
            value=60.0,
            step=1.0,
            key="joint_target_temp"
        )

    with col2:

        flow_weight = st.slider(
            "Flow Penalty Weight",
            min_value=0.00,
            max_value=1.00,
            value=0.20,
            step=0.05,
        )

        temperature_weight = st.slider(
            "Temperature Weight",
            min_value=0.50,
            max_value=5.00,
            value=1.00,
            step=0.10,
        )

    with col3:

        maxiter = st.slider(
            "Max Iterations",
            min_value=20,
            max_value=300,
            value=150,
        )

        popsize = st.slider(
            "Population Size",
            min_value=5,
            max_value=30,
            value=12,
        )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Run Optimization
    # -------------------------------------------------------------------------

    if st.button(
        "🤝 Run Joint Optimization",
        type="primary",
        use_container_width=True,
    ):

        battery_temp = clamp_battery_temp(
            float(joint_battery_temp)
        )

        progress_placeholder = st.empty()

        with st.spinner(
            "Running Differential Evolution..."
        ):

            result = run_joint_optimization(

                model=model,
                device=device,

                input_scaler=input_scaler,
                output_scaler=output_scaler,

                FIDX=FIDX,
                FEATURES=FEATURES,

                battery_temp_C=battery_temp,

                target_temp=
                joint_target_temp,

                flow_weight=
                flow_weight,

                temperature_weight=
                temperature_weight,

                maxiter=maxiter,

                popsize=popsize,
            )

        st.session_state[
            "joint_optimization"
        ] = result

        st.success(
            "Optimization completed."
        )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    if (
        "joint_optimization"
        in st.session_state
    ):

        result = st.session_state[
            "joint_optimization"
        ]

        summary = (
            optimization_summary(
                result
            )
        )

        pcm = result["pcm"]

        metrics = result["metrics"]

        # ---------------------------------------------------------------------
        # Dashboard Metrics
        # ---------------------------------------------------------------------

        st.markdown(
            "## Optimal Design"
        )

        m1, m2, m3, m4 = st.columns(4)

        with m1:

            metric_card(
                "Flow Rate",
                f"{summary['flow_g_s']:.2f}",
                "g/s"
            )

        with m2:

            metric_card(
                "Peak Battery Temp",
                f"{summary['peak_battery_temp_C']:.2f}",
                "°C"
            )

        with m3:

            metric_card(
                "Average LF",
                f"{summary['avg_lf']:.3f}"
            )

        with m4:

            metric_card(
                "Final LF",
                f"{summary['final_lf']:.3f}"
            )

        # ---------------------------------------------------------------------
        # PCM Properties
        # ---------------------------------------------------------------------

        st.markdown(
            "## Optimized PCM Properties"
        )

        pcm_df = pd.DataFrame([{

            "Density (kg/m³)":
                pcm["density"],

            "Specific Heat (J/kgK)":
                pcm["cp"],

            "Thermal Conductivity (W/mK)":
                pcm["k"],

            "Viscosity (Pa.s)":
                pcm["viscosity"],

            "Latent Heat (J/kg)":
                pcm["latent_heat"],

            "Tsolidus (K)":
                pcm["tsolidus"],

            "Tliquidus (K)":
                pcm["tliquidus"],

            "Tmelt (°C)":
                (
                    (
                        pcm["tsolidus"]
                        +
                        pcm["tliquidus"]
                    ) / 2
                ) - 273.15,
        }])

        st.dataframe(
            pcm_df,
            use_container_width=True,
        )

        # ---------------------------------------------------------------------
        # Comparison Against Standard PCMs
        # ---------------------------------------------------------------------

        st.markdown(
            "## Comparison Against Standard PCMs"
        )

        comparison_rows = (
            compare_against_standard(
                result,
                PCM_DATABASE,
            )
        )

        comparison_df = pd.DataFrame(
            comparison_rows
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
        )

        # ---------------------------------------------------------------------
        # Radar Chart
        # ---------------------------------------------------------------------

        st.markdown(
            "## Radar Comparison"
        )

        radar_columns = [
            "density",
            "cp",
            "k",
            "latent_heat",
        ]

        radar_df = comparison_df.copy()

        for col in radar_columns:

            mn = radar_df[col].min()
            mx = radar_df[col].max()

            radar_df[col] = (
                radar_df[col] - mn
            ) / (
                mx - mn + 1e-9
            )

        fig_radar = go.Figure()

        for _, row in radar_df.iterrows():

            fig_radar.add_trace(
                go.Scatterpolar(

                    r=[
                        row["density"],
                        row["cp"],
                        row["k"],
                        row["latent_heat"],
                    ],

                    theta=[
                        "Density",
                        "Cp",
                        "k",
                        "Latent Heat",
                    ],

                    fill="toself",

                    name=row["name"],
                )
            )

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            height=600,
        )

        st.plotly_chart(
            fig_radar,
            use_container_width=True,
            key="radar_comparison_plot"
        )

        # ---------------------------------------------------------------------
        # Performance Summary
        # ---------------------------------------------------------------------

        st.markdown(
            "## Thermal Performance"
        )

        perf_df = pd.DataFrame([{

            "Peak Battery Temp (°C)":
                metrics[
                    "peak_battery_temp"
                ],

            "Average Battery Temp (°C)":
                metrics[
                    "avg_battery_temp"
                ],

            "Peak PCM Temp (°C)":
                metrics[
                    "peak_pcm_temp"
                ],

            "Average LF":
                metrics[
                    "avg_lf"
                ],

            "Final LF":
                metrics[
                    "final_lf"
                ],

            "Peak Outlet Temp (°C)":
                metrics[
                    "peak_outlet_temp"
                ],
        }])

        st.dataframe(
            perf_df,
            use_container_width=True,
        )

        # ---------------------------------------------------------------------
        # Optimization Statistics
        # ---------------------------------------------------------------------

        st.markdown(
            "## Optimization Statistics"
        )

        stat1, stat2, stat3, stat4 = st.columns(4)

        with stat1:

            metric_card(
                "Objective",
                f"{result['objective']:.3f}"
            )

        with stat2:

            metric_card(
                "Iterations",
                result["nit"]
            )

        with stat3:

            metric_card(
                "Evaluations",
                result["nfev"]
            )

        with stat4:

            metric_card(
                "Success",
                str(
                    result["success"]
                )
            )

        # ---------------------------------------------------------------------
        # Downloads
        # ---------------------------------------------------------------------

        d1, d2 = st.columns(2)

        with d1:

            dataframe_download_button(
                pcm_df,
                "⬇ Download Optimized PCM",
                "joint_optimized_pcm.csv"
            )

        with d2:

            dataframe_download_button(
                comparison_df,
                "⬇ Download Comparison",
                "joint_pcm_comparison.csv"
            )

# =============================================================================
# END PART C1
# NEXT:
# PART C2
# TAB 8 — HIGH THROUGHPUT MATERIALS SCREENING
# =============================================================================
# =============================================================================
# TAB 8 — HIGH THROUGHPUT MATERIALS SCREENING
# =============================================================================

with tab8:

    st.markdown(
        '<div class="section-header">High Throughput Materials Screening</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Generate and evaluate thousands of synthetic PCM candidates using the PINN surrogate."
    )

    # -------------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        screening_temp = st.selectbox(
            "Battery Temperature (°C)",
            [50, 70, 90],
            key="screening_temp"
        )

        screening_flow = st.slider(
            "Mass Flow Rate (g/s)",
            1.0,
            10.0,
            5.0,
            0.5,
            key="screening_flow"
        )

    with c2:

        n_candidates = st.selectbox(
            "Candidates",
            [
                1000,
                5000,
                10000,
                25000,
                50000,
                100000,
            ],
            index=2,
        )

        top_n = st.slider(
            "Top Results",
            10,
            500,
            100,
            10,
        )

    with c3:

        evaluation_time = st.slider(
            "Evaluation Time (s)",
            60,
            900,
            600,
            30,
        )

        batch_size = st.selectbox(
            "Batch Size",
            [
                1024,
                2048,
                4096,
                8192,
            ],
            index=2,
        )

    with c4:

        random_seed = st.number_input(
            "Random Seed",
            value=42,
            key="screening_seed"
        )

        runtime_text = {
            1000: "Very Fast",
            5000: "Fast",
            10000: "Fast",
            25000: "Moderate",
            50000: "Heavy",
            100000: "Very Heavy",
        }

        st.info(
            f"Expected Runtime: {runtime_text[n_candidates]}"
        )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Run Screening
    # -------------------------------------------------------------------------

    if st.button(
        "🚀 Run Screening",
        type="primary",
        use_container_width=True,
    ):

        battery_temp = clamp_battery_temp(
            float(screening_temp)
        )

        mass_flow_rate = (
            screening_flow / 1000.0
        )

        progress_bar = st.progress(0)

        with st.spinner(
            f"Evaluating {n_candidates:,} candidates..."
        ):

            progress_bar.progress(20)

            screening_result = run_material_screening(

                model=model,
                device=device,

                input_scaler=input_scaler,
                output_scaler=output_scaler,

                FIDX=FIDX,
                FEATURES=FEATURES,

                n_candidates=n_candidates,

                battery_temp_C=
                battery_temp,

                mass_flow_rate=
                mass_flow_rate,

                eval_time_s=
                evaluation_time,

                top_n=top_n,

                batch_size=
                batch_size,

                seed=random_seed,
            )

            progress_bar.progress(100)

        st.session_state[
            "screening_results"
        ] = screening_result

        st.success(
            f"Successfully screened {n_candidates:,} PCM candidates."
        )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    if (
        st.session_state[
            "screening_results"
        ]
        is not None
    ):

        screening_result = (
            st.session_state[
                "screening_results"
            ]
        )

        top_df = screening_result[
            "top_df"
        ]

        # ---------------------------------------------------------------------
        # Summary Metrics
        # ---------------------------------------------------------------------

        st.markdown(
            "## Screening Summary"
        )

        best = top_df.iloc[0]

        m1, m2, m3, m4 = st.columns(4)

        with m1:

            metric_card(
                "Best Score",
                f"{best['score']:.4f}"
            )

        with m2:

            metric_card(
                "Best Tmax",
                f"{best['battery_temp_C']:.2f}",
                "°C"
            )

        with m3:

            metric_card(
                "Best LF",
                f"{best['liquid_fraction']:.3f}"
            )

        with m4:

            metric_card(
                "Best Tm",
                f"{best['Tm_C']:.2f}",
                "°C"
            )

        # ---------------------------------------------------------------------
        # Best Candidate
        # ---------------------------------------------------------------------

        st.markdown(
            "## Rank #1 Candidate"
        )

        best_candidate_df = pd.DataFrame(
            [best]
        )

        st.dataframe(
            best_candidate_df,
            use_container_width=True,
        )

        # ---------------------------------------------------------------------
        # Top Candidates Table
        # ---------------------------------------------------------------------

        st.markdown(
            f"## Top {len(top_df)} Candidates"
        )

        st.dataframe(
            top_df,
            use_container_width=True,
            height=500,
        )

        # ---------------------------------------------------------------------
        # Temperature vs Latent Heat
        # ---------------------------------------------------------------------

        st.markdown(
            "## Materials Landscape"
        )

        fig1 = px.scatter(

            top_df,

            x="latent_heat_kJ_kg",

            y="battery_temp_C",

            color="score",

            hover_data=[
                "density_kg_m3",
                "cp_J_kgK",
                "k_W_mK",
                "Tm_C",
            ],

            title=
            "Battery Temperature vs Latent Heat",
        )

        st.plotly_chart(
            fig1,
            use_container_width=True,
            key="temp_vs_latent_heat_plot"
        )

        # ---------------------------------------------------------------------
        # Density vs Cp
        # ---------------------------------------------------------------------

        fig2 = px.scatter(

            top_df,

            x="density_kg_m3",

            y="cp_J_kgK",

            color="score",

            title=
            "Density vs Specific Heat",
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
            key="density_vs_cp_plot"
        )

        # ---------------------------------------------------------------------
        # Conductivity vs LF
        # ---------------------------------------------------------------------

        fig3 = px.scatter(

            top_df,

            x="k_W_mK",

            y="liquid_fraction",

            color="score",

            title=
            "Thermal Conductivity vs LF",
        )

        st.plotly_chart(
            fig3,
            use_container_width=True,
            key="conductivity_vs_lf_plot"
        )

        # ---------------------------------------------------------------------
        # Histograms
        # ---------------------------------------------------------------------

        st.markdown(
            "## Property Distributions"
        )

        h1, h2 = st.columns(2)

        with h1:

            fig4 = px.histogram(
                top_df,
                x="Tm_C",
                nbins=20,
                title=
                "Melting Temperature Distribution",
            )

            st.plotly_chart(
                fig4,
                use_container_width=True,
                key="melting_temp_histogram"
            )

        with h2:

            fig5 = px.histogram(
                top_df,
                x="latent_heat_kJ_kg",
                nbins=20,
                title=
                "Latent Heat Distribution",
            )

            st.plotly_chart(
                fig5,
                use_container_width=True,
                key="latent_heat_histogram"
            )

        # ---------------------------------------------------------------------
        # Parallel Coordinates
        # ---------------------------------------------------------------------

        st.markdown(
            "## Parallel Coordinates (Top 20)"
        )

        parallel_df = (
            top_df
            .head(20)
            .copy()
        )

        fig_parallel = (
            px.parallel_coordinates(

                parallel_df,

                dimensions=[

                    "density_kg_m3",

                    "cp_J_kgK",

                    "k_W_mK",

                    "latent_heat_kJ_kg",

                    "Tm_C",

                    "battery_temp_C",

                    "liquid_fraction",

                    "score",
                ],

                color="score",
            )
        )

        st.plotly_chart(
            fig_parallel,
            use_container_width=True,
            key="parallel_coordinates_plot"
        )

        # ---------------------------------------------------------------------
        # Top 10 Leaderboard
        # ---------------------------------------------------------------------

        st.markdown(
            "## Top 10 Leaderboard"
        )

        leaderboard = (
            top_df[
                [
                    "rank",
                    "score",
                    "battery_temp_C",
                    "liquid_fraction",
                    "latent_heat_kJ_kg",
                    "Tm_C",
                ]
            ]
            .head(10)
        )

        st.dataframe(
            leaderboard,
            use_container_width=True,
        )

        # ---------------------------------------------------------------------
        # Downloads
        # ---------------------------------------------------------------------

        st.markdown(
            "## Export"
        )

        d1, d2 = st.columns(2)

        with d1:

            dataframe_download_button(
                top_df,
                "⬇ Download Top Candidates",
                "top_pcm_candidates.csv",
            )

        with d2:

            summary_export = pd.DataFrame([{

                "best_score":
                    best["score"],

                "best_temperature":
                    best["battery_temp_C"],

                "best_liquid_fraction":
                    best["liquid_fraction"],

                "best_tm":
                    best["Tm_C"],

                "best_density":
                    best["density_kg_m3"],

                "best_cp":
                    best["cp_J_kgK"],

                "best_k":
                    best["k_W_mK"],

                "best_latent_heat":
                    best["latent_heat_kJ_kg"],
            }])

            dataframe_download_button(
                summary_export,
                "⬇ Download Summary",
                "screening_summary.csv",
            )

# =============================================================================
# END PART C2
# END OF APP.PY
# =============================================================================