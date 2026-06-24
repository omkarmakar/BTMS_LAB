

"""
PINN-Based PCM Battery Thermal Management — Streamlit App
Replicates the inference notebook with live interactive controls.

Run:
    streamlit run btms_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import joblib
from scipy.optimize import minimize, differential_evolution
import io, os, warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTMS PINN Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {font-size:2rem;font-weight:700;color:#1f4e79;}
    .sub-title  {font-size:1rem;color:#555;margin-bottom:1.2rem;}
    .metric-card{
        background:#f0f6ff;border-radius:10px;padding:16px 20px;
        border-left:4px solid #1f4e79;margin-bottom:8px;
    }
    .metric-label{font-size:.78rem;color:#666;text-transform:uppercase;letter-spacing:.05em;}
    .metric-value{font-size:1.7rem;font-weight:700;color:#1f4e79;}
    .metric-unit {font-size:.85rem;color:#888;}
    .warn-box{background:#fff3cd;border-left:4px solid #f0ad4e;padding:10px 14px;border-radius:6px;margin:8px 0;}
    .ok-box  {background:#d4edda;border-left:4px solid #28a745;padding:10px 14px;border-radius:6px;margin:8px 0;}
    .section-header{font-size:1.25rem;font-weight:600;color:#1f4e79;margin-top:1.4rem;margin-bottom:.4rem;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PINN Architecture (identical to training V7)
# ─────────────────────────────────────────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self, in_features=20, out_features=4, hidden=256, depth=6):
        super().__init__()
        layers = [nn.Linear(in_features, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BATTERY = dict(density=3025, cp=763.25, k=2.0, heat_generation=652173.9)
BATTERY_VOLUME = 2.3e-5
BATTERY["mass"]             = BATTERY["density"] * BATTERY_VOLUME
BATTERY["thermal_capacity"] = BATTERY["mass"] * BATTERY["cp"]
BATTERY["power"]            = BATTERY["heat_generation"] * BATTERY_VOLUME

T_INLET_K   = 293.15
T_AMBIENT_K = 293.15
TRAINING_BATTERY_TEMPS_C = [50.0, 70.0, 90.0]
FLOW_LIMITS = {"min": 1e-3, "max": 1e-2}

PCM_COUNT = 48
PCM_OD_m, PCM_ID_m, PCM_H_m = 4e-3, 3e-3, 5e-3
PCM_VOLUME = PCM_COUNT * (np.pi / 4) * (PCM_ID_m**2) * PCM_H_m

PCM_DATABASE = {
    "RT27": dict(density=880, cp=2000, k=0.20, viscosity=0.0040, latent_heat=179000, tsolidus=299.15, tliquidus=301.15),
    "RT45": dict(density=800, cp=2000, k=0.20, viscosity=0.0040, latent_heat=170000, tsolidus=314.15, tliquidus=318.15),
    "RT55": dict(density=770, cp=2000, k=0.20, viscosity=0.0045, latent_heat=180000, tsolidus=324.15, tliquidus=328.15),
    "RT70": dict(density=780, cp=2200, k=0.20, viscosity=0.0050, latent_heat=190000, tsolidus=341.15, tliquidus=345.15),
}

PCM_BOUNDS = {
    "density":     (770,    880),
    "cp":          (2000,   2200),
    "k":           (0.18,   0.22),
    "viscosity":   (0.003,  0.006),
    "latent_heat": (165000, 200000),
    "tsolidus_K":  (299.15, 341.15),
    "melt_range":  (2.0,    6.0),
}

# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading PINN model…")
def load_model(pth_path, iscaler_path, oscaler_path, features_path, targets_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # checkpoint = torch.load(pth_path, map_location=device)
    # model = PINN().to(device)
    # model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint = torch.load(
        pth_path,
        map_location=device
    )

    model = PINN().to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    input_scaler  = joblib.load(iscaler_path)
    output_scaler = joblib.load(oscaler_path)
    FEATURES      = joblib.load(features_path)
    TARGETS       = joblib.load(targets_path)
    return model, device, input_scaler, output_scaler, FEATURES, TARGETS, checkpoint

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def clamp_battery_temp(T_C):
    return min(TRAINING_BATTERY_TEMPS_C, key=lambda v: abs(v - T_C))


def build_feature_row(FIDX, FEATURES,
                      pcm_density, pcm_cp, pcm_k, pcm_viscosity, pcm_latent_heat,
                      tsolidus_K, tliquidus_K, mass_flow_rate,
                      battery_patch_temp_C, time_s):
    Tm_K       = (tsolidus_K + tliquidus_K) / 2
    melt_range = tliquidus_K - tsolidus_K
    T_bat_K    = battery_patch_temp_C + 273.15
    stefan               = pcm_cp * (T_bat_K - Tm_K) / pcm_latent_heat
    pcm_thermal_storage  = pcm_density * pcm_latent_heat
    flow_pcm_interaction = mass_flow_rate * pcm_k
    row = np.zeros(len(FEATURES))
    row[FIDX["Density"]]                     = pcm_density
    row[FIDX["Specific_Heat"]]               = pcm_cp
    row[FIDX["Thermal_Conductivity"]]        = pcm_k
    row[FIDX["Viscosity"]]                   = pcm_viscosity
    row[FIDX["Latent_Heat"]]                 = pcm_latent_heat
    row[FIDX["Tm_K"]]                        = Tm_K
    row[FIDX["Melt_Range_K"]]               = melt_range
    row[FIDX["Battery_Density"]]             = BATTERY["density"]
    row[FIDX["Battery_Cp"]]                  = BATTERY["cp"]
    row[FIDX["Battery_k"]]                   = BATTERY["k"]
    row[FIDX["Battery_HeatGeneration_W_m3"]] = BATTERY["heat_generation"]
    row[FIDX["Mass_Flow_Rate_kg_s"]]         = mass_flow_rate
    row[FIDX["Initial_PCM_Temperature_C"]]   = battery_patch_temp_C
    row[FIDX["Time_s"]]                      = time_s
    row[FIDX["Stefan_Number"]]               = stefan
    row[FIDX["PCM_ThermalStorage"]]          = pcm_thermal_storage
    row[FIDX["Battery_ThermalCapacity_J_K"]] = BATTERY["thermal_capacity"]
    row[FIDX["Tsolidus_K"]]                  = tsolidus_K
    row[FIDX["Tliquidus_K"]]                = tliquidus_K
    row[FIDX["Flow_PCM_Interaction"]]        = flow_pcm_interaction
    return row


def predict_batch(model, device, input_scaler, output_scaler, rows_2d):
    X_scaled = input_scaler.transform(rows_2d).astype(np.float32)
    with torch.no_grad():
        pred_scaled = model(torch.tensor(X_scaled, device=device)).cpu().numpy()
    pred_phys = output_scaler.inverse_transform(pred_scaled)
    return {
        "battery_temp": pred_phys[:, 0],
        "pcm_temp":     pred_phys[:, 1],
        "lf":           np.clip(pred_phys[:, 2], 0, 1),
        "outlet_temp":  np.maximum(pred_phys[:, 3], T_INLET_K),
    }


def run_timeseries(model, device, input_scaler, output_scaler, FIDX, FEATURES,
                   pcm, battery_temp_C, mass_flow_rate, time_vec=None):
    if time_vec is None:
        time_vec = np.linspace(0, 900, 181)
    rows = np.array([
        build_feature_row(FIDX, FEATURES,
                          pcm["density"], pcm["cp"], pcm["k"],
                          pcm["viscosity"], pcm["latent_heat"],
                          pcm["tsolidus"], pcm["tliquidus"],
                          mass_flow_rate, battery_temp_C, t)
        for t in time_vec
    ])
    return predict_batch(model, device, input_scaler, output_scaler, rows), time_vec


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔋 BTMS PINN Dashboard")
    st.markdown("Local Model Deployment")
    model_status = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="main-title">🔋 PCM-Assisted Battery Thermal Management</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">Physics-Informed Neural Network · Real-time inference dashboard</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Local model paths
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR = "models"

pth_path = os.path.join(MODEL_DIR, "best_pinn.pth")
iscaler_path = os.path.join(MODEL_DIR, "input_scaler.pkl")
oscaler_path = os.path.join(MODEL_DIR, "output_scaler.pkl")
features_path = os.path.join(MODEL_DIR, "feature_columns.pkl")
targets_path = os.path.join(MODEL_DIR, "target_columns.pkl")

required_files = [
    pth_path,
    iscaler_path,
    oscaler_path,
    features_path,
    targets_path,
]

missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    st.error("Missing model files:")
    for f in missing:
        st.write(f"❌ {f}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

try:
    (
        model,
        device,
        input_scaler,
        output_scaler,
        FEATURES,
        TARGETS,
        checkpoint,
    ) = load_model(
        pth_path,
        iscaler_path,
        oscaler_path,
        features_path,
        targets_path,
    )

    FIDX = {f: i for i, f in enumerate(FEATURES)}

    epoch = "N/A"
    r2 = "N/A"

    if isinstance(checkpoint, dict):
        epoch = checkpoint.get("epoch", "N/A")
        r2 = checkpoint.get("r2", "N/A")

    with model_status.container():
        st.success("✅ Model Loaded")
        st.markdown(f"**Epoch:** {epoch}")
        st.markdown(f"**R²:** {r2}")
        st.markdown(f"**Device:** `{device}`")
        st.markdown(f"**Features:** {len(FEATURES)}")
        st.markdown(f"**Targets:** {len(TARGETS)}")

except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Time-Series Prediction
# ═══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Time-Series Prediction",
    "🔍 Optimal PCM Finder",
    "💧 Optimal Flow Rate",
    "🧪 PCM Comparison",
])
with tab1:
    st.markdown('<div class="section-header">Scenario Setup</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        battery_temp = st.selectbox("Battery patch temperature (°C)",
                                    options=[50, 70, 90], index=0,
                                    help="Heat source temperature — 3 CFD-validated cases")
        pcm_choice   = st.selectbox("PCM material", list(PCM_DATABASE.keys()) + ["Custom"])

    with c2:
        mass_flow_g  = st.slider("Mass flow rate (g/s)", 1.0, 10.0, 5.0, 0.5,
                                 format="%.1f g/s")
        mass_flow    = mass_flow_g * 1e-3

    with c3:
        t_end = st.slider("Simulation end time (s)", 60, 900, 900, 30)
        n_pts = st.slider("Time points", 50, 300, 181, 10)

    # Custom PCM expander
    if pcm_choice == "Custom":
        with st.expander("🔧 Custom PCM Properties", expanded=True):
            cc1, cc2 = st.columns(2)
            with cc1:
                c_density     = st.number_input("Density (kg/m³)", 700, 1000, 800)
                c_cp          = st.number_input("Specific Heat (J/kg·K)", 1000, 3000, 2000)
                c_k           = st.number_input("Conductivity (W/m·K)", 0.05, 1.0, 0.20, format="%.3f")
                c_visc        = st.number_input("Viscosity (Pa·s)", 0.001, 0.02, 0.004, format="%.4f")
            with cc2:
                c_latent      = st.number_input("Latent Heat (J/kg)", 100000, 250000, 170000, 1000)
                c_tsolidus    = st.number_input("T_solidus (K)", 290.0, 360.0, 314.15, 0.5)
                c_tliquidus   = st.number_input("T_liquidus (K)", 291.0, 365.0, 318.15, 0.5)
        pcm_props = dict(density=c_density, cp=c_cp, k=c_k, viscosity=c_visc,
                         latent_heat=c_latent, tsolidus=c_tsolidus, tliquidus=c_tliquidus)
    else:
        pcm_props = PCM_DATABASE[pcm_choice]

    if st.button("▶ Run Prediction", type="primary", key="run_ts"):
        battery_temp_c = clamp_battery_temp(float(battery_temp))
        time_vec = np.linspace(0, t_end, n_pts)

        with st.spinner("Running PINN inference…"):
            res, tv = run_timeseries(model, device, input_scaler, output_scaler,
                                     FIDX, FEATURES,
                                     pcm_props, battery_temp_c, mass_flow, time_vec)

        # Metrics row
        st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        peak_bat = (res["battery_temp"] - 273.15).max()
        final_lf = res["lf"][-1]
        peak_out = (res["outlet_temp"] - 273.15).max()
        peak_pcm = (res["pcm_temp"] - 273.15).max()

        for col, label, val, unit, thresh, lo in [
            (m1, "Peak Battery Temp", peak_bat, "°C", 60, True),
            (m2, "Final Liquid Fraction", final_lf*100, "%", 80, True),
            (m3, "Peak Outlet Temp", peak_out, "°C", 35, True),
            (m4, "Peak PCM Temp", peak_pcm, "°C", 70, True),
        ]:
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{val:.1f}</div>
              <div class="metric-unit">{unit}</div>
            </div>
            """, unsafe_allow_html=True)

        # 4-panel time-series plot
        fig, axes = plt.subplots(2, 2, figsize=(13, 7))
        fig.suptitle(
            f"PINN Time-Series  |  {pcm_choice}  |  T_bat={battery_temp_c}°C  |  ṁ={mass_flow_g:.1f} g/s",
            fontsize=12, fontweight="bold"
        )

        plots = [
            (axes[0,0], res["battery_temp"]-273.15, "Battery Temperature (°C)", "#d62728"),
            (axes[0,1], res["pcm_temp"]-273.15,     "PCM Temperature (°C)",     "#1f77b4"),
            (axes[1,0], res["lf"],                   "Liquid Fraction",           "#2ca02c"),
            (axes[1,1], res["outlet_temp"]-273.15,  "Outlet Water Temp (°C)",   "#ff7f0e"),
        ]
        for ax, data, ylabel, color in plots:
            ax.plot(tv, data, color=color, lw=2)
            ax.set(xlabel="Time (s)", ylabel=ylabel)
            ax.grid(alpha=0.3)
            ax.fill_between(tv, data, alpha=0.08, color=color)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # CSV download
        df_out = pd.DataFrame({
            "time_s":         tv,
            "battery_temp_C": res["battery_temp"] - 273.15,
            "pcm_temp_C":     res["pcm_temp"] - 273.15,
            "liquid_fraction": res["lf"],
            "outlet_temp_C":  res["outlet_temp"] - 273.15,
        })
        st.download_button(
            "⬇ Download CSV",
            data=df_out.to_csv(index=False).encode(),
            file_name="timeseries_results.csv",
            mime="text/csv"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Optimal PCM Finder
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Optimal PCM Finder</div>', unsafe_allow_html=True)
    st.caption("Differential evolution over PCM property space to minimise peak battery temperature.")

    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        opt_bat_temp = st.selectbox("Battery patch temperature (°C)",
                                    [50, 70, 90], index=0, key="opt_bat")
        opt_mdot_g   = st.slider("Mass flow rate (g/s)", 1.0, 10.0, 5.0, 0.5, key="opt_mdot")
    with oc2:
        opt_target_max = st.number_input("Target max battery temp (°C)", 30.0, 90.0, 70.0, 1.0)
    with oc3:
        de_maxiter = st.slider("DE max iterations", 50, 300, 150, 25,
                               help="More iterations = better result but slower")
        de_popsize = st.slider("DE population size", 8, 20, 12)

    EVAL_TIMES = np.array([150, 300, 450, 600, 750, 900])

    if st.button("🔍 Find Optimal PCM", type="primary", key="run_opt_pcm"):
        bat_c  = clamp_battery_temp(float(opt_bat_temp))
        mdot   = opt_mdot_g * 1e-3

        def pcm_objective(x):
            density, cp, k, visc, latent, Ts, mr = x
            Tl   = Ts + mr
            rows = np.array([
                build_feature_row(FIDX, FEATURES, density, cp, k, visc, latent,
                                  Ts, Tl, mdot, bat_c, t)
                for t in EVAL_TIMES
            ])
            T_bat = predict_batch(model, device, input_scaler, output_scaler, rows)["battery_temp"] - 273.15
            peak  = T_bat.max()
            penalty = 1000 * max(0, peak - opt_target_max)**2
            return peak + penalty

        progress = st.progress(0, text="Running differential evolution…")
        iters_done = [0]

        def de_callback(xk, convergence):
            iters_done[0] += 1
            pct = min(int(iters_done[0] / de_maxiter * 100), 99)
            progress.progress(pct, text=f"DE iteration {iters_done[0]}/{de_maxiter} — convergence={convergence:.4f}")

        with st.spinner("Optimising PCM properties…"):
            bounds_list = list(PCM_BOUNDS.values())
            de_result = differential_evolution(
                pcm_objective, bounds=bounds_list,
                maxiter=de_maxiter, popsize=de_popsize,
                tol=1e-4, seed=42, disp=False,
                mutation=(0.5, 1.0), recombination=0.9,
                callback=de_callback,
            )

        progress.progress(100, text="Done!")
        opt = de_result.x
        density_opt, cp_opt, k_opt, visc_opt, latent_opt, Ts_opt, mr_opt = opt
        Tl_opt = Ts_opt + mr_opt
        Tm_opt = (Ts_opt + Tl_opt) / 2

        st.markdown('<div class="section-header">Optimal PCM Properties</div>', unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        for col, label, val, unit in [
            (p1, "Density",       density_opt,           "kg/m³"),
            (p2, "Specific Heat", cp_opt,                "J/kg·K"),
            (p3, "Latent Heat",   latent_opt/1000,       "kJ/kg"),
            (p4, "T_melt",        Tm_opt - 273.15,       "°C"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{val:.1f}</div>
              <div class="metric-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)

        q1, q2, q3, q4 = st.columns(4)
        for col, label, val, unit in [
            (q1, "Conductivity",  k_opt,                 "W/m·K"),
            (q2, "Viscosity",     visc_opt*1000,         "mPa·s"),
            (q3, "T_solidus",     Ts_opt - 273.15,       "°C"),
            (q4, "Melt Range",    mr_opt,                "K"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{val:.2f}</div>
              <div class="metric-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)

        # Find closest standard PCM
        closest_name = min(PCM_DATABASE,
                           key=lambda n: abs((PCM_DATABASE[n]["tsolidus"]+PCM_DATABASE[n]["tliquidus"])/2 - Tm_opt))
        closest_pcm  = PCM_DATABASE[closest_name]

        st.markdown(f"**Closest standard PCM:** `{closest_name}` (T_melt ≈ {(closest_pcm['tsolidus']+closest_pcm['tliquidus'])/2-273.15:.1f}°C)")

        time_vec = np.linspace(0, 900, 181)
        opt_pcm_props = dict(density=density_opt, cp=cp_opt, k=k_opt,
                             viscosity=visc_opt, latent_heat=latent_opt,
                             tsolidus=Ts_opt, tliquidus=Tl_opt)
        res_opt, _ = run_timeseries(model, device, input_scaler, output_scaler,
                                    FIDX, FEATURES, opt_pcm_props, bat_c, mdot, time_vec)
        res_cls, _ = run_timeseries(model, device, input_scaler, output_scaler,
                                    FIDX, FEATURES, closest_pcm, bat_c, mdot, time_vec)

        peak_opt = (res_opt["battery_temp"]-273.15).max()
        peak_cls = (res_cls["battery_temp"]-273.15).max()

        box_cls = "ok-box" if peak_opt < opt_target_max else "warn-box"
        st.markdown(f"""
        <div class="{box_cls}">
          ⚡ Optimal PCM peak T_bat: <b>{peak_opt:.2f}°C</b> &nbsp;|&nbsp;
          {closest_name} peak T_bat: <b>{peak_cls:.2f}°C</b> &nbsp;|&nbsp;
          Target: <b>&lt;{opt_target_max}°C</b>
        </div>""", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(time_vec, res_opt["battery_temp"]-273.15, "r-",  lw=2.5, label="Optimal PCM")
        ax.plot(time_vec, res_cls["battery_temp"]-273.15, "b--", lw=2,   label=f"Closest std ({closest_name})")
        ax.axhline(opt_target_max, color="black", ls=":", lw=1.5, label=f"Limit ({opt_target_max}°C)")
        ax.fill_between(time_vec, res_opt["battery_temp"]-273.15, alpha=0.08, color="red")
        ax.set(xlabel="Time (s)", ylabel="Battery Temperature (°C)",
               title=f"Optimal PCM vs {closest_name}  |  T_bat={bat_c}°C  ṁ={opt_mdot_g:.1f} g/s")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Optimal Flow Rate
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Optimal Flow Rate Finder</div>', unsafe_allow_html=True)
    st.caption("Find the minimum mass flow rate that keeps battery below target temperature.")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        fl_bat_temp  = st.selectbox("Battery temperature (°C)", [50, 70, 90], key="fl_bat")
        fl_pcm_name  = st.selectbox("PCM material", list(PCM_DATABASE.keys()), key="fl_pcm")
    with fc2:
        fl_target    = st.number_input("Target max temp (°C)", 30.0, 90.0, 45.0, 1.0)
    with fc3:
        fl_sweep_n   = st.slider("Flow rate sweep points", 5, 40, 20,
                                 help="Sweep visualization resolution")

    EVAL_TIMES_FL = np.array([150, 300, 450, 600, 750, 900])

    if st.button("💧 Find Optimal Flow Rate", type="primary", key="run_opt_fl"):
        fl_bat_c  = clamp_battery_temp(float(fl_bat_temp))
        pcm_fl    = PCM_DATABASE[fl_pcm_name]

        def flow_objective(mdot_arr):
            mdot = mdot_arr[0]
            rows = np.array([
                build_feature_row(FIDX, FEATURES,
                                  pcm_fl["density"], pcm_fl["cp"], pcm_fl["k"],
                                  pcm_fl["viscosity"], pcm_fl["latent_heat"],
                                  pcm_fl["tsolidus"], pcm_fl["tliquidus"],
                                  mdot, fl_bat_c, t)
                for t in EVAL_TIMES_FL
            ])
            res = predict_batch(model, device, input_scaler, output_scaler, rows)
            peak = (res["battery_temp"] - 273.15).max()
            penalty = 5000 * max(0, peak - fl_target)**2
            return mdot + penalty

        with st.spinner("Optimising flow rate…"):
            fl_result = minimize(
                flow_objective, x0=[5e-3],
                bounds=[(FLOW_LIMITS["min"], FLOW_LIMITS["max"])],
                method="L-BFGS-B",
                options={"ftol": 1e-9, "maxiter": 200},
            )
        opt_mdot = fl_result.x[0]

        # Verify at optimal flow
        time_vec  = np.linspace(0, 900, 181)
        res_opt, _ = run_timeseries(model, device, input_scaler, output_scaler,
                                    FIDX, FEATURES, pcm_fl, fl_bat_c, opt_mdot, time_vec)
        peak_at_opt = (res_opt["battery_temp"]-273.15).max()

        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Optimal Mass Flow Rate</div>
          <div class="metric-value">{opt_mdot*1000:.2f}</div>
          <div class="metric-unit">g/s &nbsp;|&nbsp; Peak T_bat = {peak_at_opt:.2f}°C</div>
        </div>""", unsafe_allow_html=True)

        # Sweep visualisation
        sweep_rates = np.linspace(FLOW_LIMITS["min"], FLOW_LIMITS["max"], fl_sweep_n)
        sweep_peaks = []
        prog2 = st.progress(0, text="Computing sweep…")
        for i, mdot_sw in enumerate(sweep_rates):
            rows_sw = np.array([
                build_feature_row(FIDX, FEATURES,
                                  pcm_fl["density"], pcm_fl["cp"], pcm_fl["k"],
                                  pcm_fl["viscosity"], pcm_fl["latent_heat"],
                                  pcm_fl["tsolidus"], pcm_fl["tliquidus"],
                                  mdot_sw, fl_bat_c, t)
                for t in EVAL_TIMES_FL
            ])
            res_sw = predict_batch(model, device, input_scaler, output_scaler, rows_sw)
            sweep_peaks.append((res_sw["battery_temp"]-273.15).max())
            prog2.progress(int((i+1)/fl_sweep_n*100))

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        # Sweep
        ax = axes[0]
        ax.plot(sweep_rates*1000, sweep_peaks, "b-o", ms=5)
        ax.axhline(fl_target, color="red", ls="--", lw=1.5, label=f"Limit ({fl_target}°C)")
        ax.axvline(opt_mdot*1000, color="green", ls="-.", lw=2,
                   label=f"Optimal = {opt_mdot*1000:.2f} g/s")
        ax.set(xlabel="Mass Flow Rate (g/s)", ylabel="Peak Battery Temp (°C)",
               title=f"Flow Rate Sweep — {fl_pcm_name}, T_bat={fl_bat_c}°C")
        ax.legend(); ax.grid(alpha=0.3)

        # Time-series at optimal
        ax2 = axes[1]
        ax2.plot(time_vec, res_opt["battery_temp"]-273.15, "g-", lw=2)
        ax2.axhline(fl_target, color="red", ls="--", lw=1.5, label=f"Limit ({fl_target}°C)")
        ax2.fill_between(time_vec, res_opt["battery_temp"]-273.15, alpha=0.1, color="green")
        ax2.set(xlabel="Time (s)", ylabel="Battery Temp (°C)",
                title=f"Time-Series @ {opt_mdot*1000:.2f} g/s")
        ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PCM Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Compare All Standard PCMs</div>', unsafe_allow_html=True)

    cp1, cp2 = st.columns(2)
    with cp1:
        cmp_bat  = st.selectbox("Battery temperature (°C)", [50, 70, 90], key="cmp_bat")
        cmp_mdot_g = st.slider("Mass flow rate (g/s)", 1.0, 10.0, 5.0, 0.5, key="cmp_mdot")
    with cp2:
        cmp_target = st.number_input("Temperature limit for annotation (°C)", 30.0, 90.0, 70.0, 1.0)

    if st.button("📊 Compare PCMs", type="primary", key="run_cmp"):
        cmp_bat_c = clamp_battery_temp(float(cmp_bat))
        cmp_mdot  = cmp_mdot_g * 1e-3
        time_vec  = np.linspace(0, 900, 181)

        results = {}
        prog3 = st.progress(0)
        for i, (name, pcm) in enumerate(PCM_DATABASE.items()):
            res, _ = run_timeseries(model, device, input_scaler, output_scaler,
                                    FIDX, FEATURES, pcm, cmp_bat_c, cmp_mdot, time_vec)
            results[name] = res
            prog3.progress(int((i+1)/len(PCM_DATABASE)*100))

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(f"PCM Comparison  |  T_bat={cmp_bat_c}°C  |  ṁ={cmp_mdot_g:.1f} g/s",
                     fontsize=12, fontweight="bold")

        ylabels = ["Battery Temp (°C)", "PCM Temp (°C)", "Liquid Fraction", "Outlet Temp (°C)"]
        keys    = ["battery_temp", "pcm_temp", "lf", "outlet_temp"]
        offsets = [-273.15, -273.15, 0, -273.15]

        for ax, key, ylabel, off in zip(axes.flatten(), keys, ylabels, offsets):
            for (name, res), color in zip(results.items(), colors):
                ax.plot(time_vec, res[key]+off, lw=2, label=name, color=color)
            if key == "battery_temp":
                ax.axhline(cmp_target, color="black", ls=":", lw=1.5,
                           label=f"Limit ({cmp_target}°C)")
            ax.set(xlabel="Time (s)", ylabel=ylabel)
            ax.legend(fontsize=8); ax.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Summary table
        st.markdown("**Peak Values Summary**")
        rows_sum = []
        for name, res in results.items():
            rows_sum.append({
                "PCM": name,
                "Peak Battery Temp (°C)": f"{(res['battery_temp']-273.15).max():.2f}",
                "Peak PCM Temp (°C)":     f"{(res['pcm_temp']-273.15).max():.2f}",
                "Final Liquid Fraction":  f"{res['lf'][-1]:.3f}",
                "Peak Outlet Temp (°C)":  f"{(res['outlet_temp']-273.15).max():.2f}",
                "Within Limit":           "✅" if (res['battery_temp']-273.15).max() < cmp_target else "❌",
            })
        st.dataframe(pd.DataFrame(rows_sum), use_container_width=True, hide_index=True)
