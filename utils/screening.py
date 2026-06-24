"""
utils/screening.py

Notebook Section 8
High Throughput Materials Screening

Generate synthetic PCM candidates
Run PINN inference
Rank candidates
Return Top-N materials
"""

import numpy as np
import pandas as pd
import torch

from .inference import (
    BATTERY,
    PCM_VOLUME,
)


# ============================================================================
# SYNTHETIC PCM GENERATION
# ============================================================================

def generate_synthetic_pcm_library(
    n_candidates=100000,
    seed=42,
):
    """
    Generate synthetic PCM library
    using uniform sampling.
    """

    np.random.seed(seed)

    rho = np.random.uniform(
        700,
        1000,
        n_candidates,
    )

    cp = np.random.uniform(
        1500,
        3000,
        n_candidates,
    )

    k = np.random.uniform(
        0.10,
        0.60,
        n_candidates,
    )

    viscosity = np.random.uniform(
        0.001,
        0.010,
        n_candidates,
    )

    latent_heat = np.random.uniform(
        165000,
        200000,
        n_candidates,
    )

    tsolidus = np.random.uniform(
        299.15,
        341.15,
        n_candidates,
    )

    melt_range = np.random.uniform(
        1.0,
        10.0,
        n_candidates,
    )

    tliquidus = (
        tsolidus
        + melt_range
    )

    tmelt = (
        tsolidus
        + tliquidus
    ) / 2

    return {
        "density": rho,
        "cp": cp,
        "k": k,
        "viscosity": viscosity,
        "latent_heat": latent_heat,
        "tsolidus": tsolidus,
        "tliquidus": tliquidus,
        "tmelt": tmelt,
        "melt_range": melt_range,
    }


# ============================================================================
# FEATURE MATRIX
# ============================================================================

def build_screening_matrix(
    library,
    FIDX,
    FEATURES,
    battery_temp_C,
    mass_flow_rate,
    eval_time_s,
):
    """
    Build PINN feature matrix.
    """

    N = len(
        library["density"]
    )

    X = np.zeros(
        (
            N,
            len(FEATURES)
        )
    )

    rho = library["density"]
    cp = library["cp"]
    k = library["k"]
    viscosity = library["viscosity"]
    latent_heat = library["latent_heat"]

    Ts = library["tsolidus"]
    Tl = library["tliquidus"]

    Tm = library["tmelt"]

    melt_range = (
        Tl - Ts
    )

    T0_K = (
        battery_temp_C
        + 273.15
    )

    stefan = (
        cp
        * (T0_K - Tm)
        / (latent_heat + 1e-9)
    )

    pcm_storage = (
        rho
        * latent_heat
        * PCM_VOLUME
    )

    flow_pcm = (
        mass_flow_rate
        * latent_heat
        / (cp + 1e-9)
    )

    X[:, FIDX["Density"]] = rho
    X[:, FIDX["Specific_Heat"]] = cp
    X[:, FIDX["Thermal_Conductivity"]] = k
    X[:, FIDX["Viscosity"]] = viscosity
    X[:, FIDX["Latent_Heat"]] = latent_heat

    X[:, FIDX["Tm_K"]] = Tm
    X[:, FIDX["Melt_Range_K"]] = melt_range

    X[:, FIDX["Battery_Density"]] = BATTERY["density"]
    X[:, FIDX["Battery_Cp"]] = BATTERY["cp"]
    X[:, FIDX["Battery_k"]] = BATTERY["k"]

    X[:, FIDX["Battery_HeatGeneration_W_m3"]] = \
        BATTERY["heat_generation"]

    X[:, FIDX["Mass_Flow_Rate_kg_s"]] = \
        mass_flow_rate

    X[:, FIDX["Initial_PCM_Temperature_C"]] = \
        battery_temp_C

    X[:, FIDX["Time_s"]] = eval_time_s

    X[:, FIDX["Stefan_Number"]] = stefan

    X[:, FIDX["PCM_ThermalStorage"]] = \
        pcm_storage

    X[:, FIDX["Battery_ThermalCapacity_J_K"]] = \
        BATTERY["thermal_capacity"]

    X[:, FIDX["Tsolidus_K"]] = Ts
    X[:, FIDX["Tliquidus_K"]] = Tl

    X[:, FIDX["Flow_PCM_Interaction"]] = \
        flow_pcm

    return X


# ============================================================================
# BATCH INFERENCE
# ============================================================================

def run_screening_inference(
    model,
    device,
    input_scaler,
    output_scaler,
    X,
    batch_size=4096,
):
    """
    Run PINN inference
    over large screening matrix.
    """

    X_scaled = (
        input_scaler
        .transform(X)
        .astype(np.float32)
    )

    all_preds = []

    n_batches = (
        len(X_scaled)
        + batch_size
        - 1
    ) // batch_size

    model.eval()

    with torch.no_grad():

        for i in range(
            n_batches
        ):

            start = (
                i * batch_size
            )

            end = min(
                start + batch_size,
                len(X_scaled),
            )

            X_batch = torch.tensor(
                X_scaled[start:end],
                device=device,
            )

            pred = (
                model(X_batch)
                .cpu()
                .numpy()
            )

            all_preds.append(
                pred
            )

    pred_scaled = np.vstack(
        all_preds
    )

    pred_phys = (
        output_scaler
        .inverse_transform(
            pred_scaled
        )
    )

    return {
        "battery_temp":
            pred_phys[:, 0] - 273.15,

        "pcm_temp":
            pred_phys[:, 1] - 273.15,

        "lf":
            np.clip(
                pred_phys[:, 2],
                0,
                1,
            ),

        "outlet_temp":
            pred_phys[:, 3] - 273.15,
    }


# ============================================================================
# SCORING
# ============================================================================

def compute_scores(
    library,
    predictions,
):
    """
    Notebook scoring system.
    """

    T_bat = predictions[
        "battery_temp"
    ]

    LF = predictions[
        "lf"
    ]

    rho = library["density"]

    latent_heat = library[
        "latent_heat"
    ]

    k = library["k"]

    T_norm = (
        T_bat.max()
        - T_bat
    ) / (
        T_bat.max()
        - T_bat.min()
        + 1e-9
    )

    LF_norm = (
        1
        - np.abs(
            LF - 0.5
        ) / 0.5
    )

    rho_L = (
        rho
        * latent_heat
    )

    storage_norm = (
        rho_L
        - rho_L.min()
    ) / (
        rho_L.max()
        - rho_L.min()
        + 1e-9
    )

    k_norm = (
        k
        - k.min()
    ) / (
        k.max()
        - k.min()
        + 1e-9
    )

    score = (
        0.50 * T_norm
        +
        0.25 * LF_norm
        +
        0.15 * storage_norm
        +
        0.10 * k_norm
    )

    return score


# ============================================================================
# TOP CANDIDATES
# ============================================================================

def extract_top_candidates(
    library,
    predictions,
    scores,
    top_n=100,
):
    """
    Return ranked dataframe.
    """

    idx = np.argsort(
        scores
    )[::-1][:top_n]

    rho_L = (
        library["density"]
        * library["latent_heat"]
    )

    df = pd.DataFrame({

        "rank":
            range(
                1,
                top_n + 1
            ),

        "score":
            scores[idx],

        "battery_temp_C":
            predictions[
                "battery_temp"
            ][idx],

        "pcm_temp_C":
            predictions[
                "pcm_temp"
            ][idx],

        "liquid_fraction":
            predictions[
                "lf"
            ][idx],

        "outlet_temp_C":
            predictions[
                "outlet_temp"
            ][idx],

        "density_kg_m3":
            library["density"][idx],

        "cp_J_kgK":
            library["cp"][idx],

        "k_W_mK":
            library["k"][idx],

        "viscosity_Pa_s":
            library["viscosity"][idx],

        "latent_heat_kJ_kg":
            library[
                "latent_heat"
            ][idx] / 1000,

        "Tm_C":
            library[
                "tmelt"
            ][idx] - 273.15,

        "Tsolidus_C":
            library[
                "tsolidus"
            ][idx] - 273.15,

        "Tliquidus_C":
            library[
                "tliquidus"
            ][idx] - 273.15,

        "melt_range_K":
            library[
                "melt_range"
            ][idx],

        "vol_storage_MJ_m3":
            rho_L[idx] / 1e6,
    })

    return df


# ============================================================================
# FULL SCREENING PIPELINE
# ============================================================================

def run_material_screening(
    model,
    device,
    input_scaler,
    output_scaler,
    FIDX,
    FEATURES,

    n_candidates=100000,

    battery_temp_C=50,

    mass_flow_rate=5e-3,

    eval_time_s=600,

    top_n=100,

    batch_size=4096,

    seed=42,
):
    """
    Complete notebook workflow.
    """

    library = (
        generate_synthetic_pcm_library(
            n_candidates,
            seed,
        )
    )

    X = build_screening_matrix(
        library,
        FIDX,
        FEATURES,
        battery_temp_C,
        mass_flow_rate,
        eval_time_s,
    )

    predictions = (
        run_screening_inference(
            model,
            device,
            input_scaler,
            output_scaler,
            X,
            batch_size,
        )
    )

    scores = compute_scores(
        library,
        predictions,
    )

    top_df = (
        extract_top_candidates(
            library,
            predictions,
            scores,
            top_n,
        )
    )

    return {
        "library": library,
        "predictions": predictions,
        "scores": scores,
        "top_df": top_df,
    }