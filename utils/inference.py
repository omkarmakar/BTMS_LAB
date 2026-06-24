"""
utils/inference.py

BTMS PINN inference utilities
"""

import numpy as np
import torch


# ============================================================================
# Global Constants
# ============================================================================

BATTERY = {
    "density": 3025,
    "cp": 763.25,
    "k": 2.0,
    "heat_generation": 652173.9,
}

BATTERY_VOLUME = 2.3e-5

BATTERY["mass"] = (
    BATTERY["density"]
    * BATTERY_VOLUME
)

BATTERY["thermal_capacity"] = (
    BATTERY["mass"]
    * BATTERY["cp"]
)

BATTERY["power"] = (
    BATTERY["heat_generation"]
    * BATTERY_VOLUME
)

T_INLET_K = 293.15

TRAINING_BATTERY_TEMPS_C = [
    50.0,
    70.0,
    90.0,
]

FLOW_LIMITS = {
    "min": 1e-3,
    "max": 1e-2,
}

PCM_COUNT = 48

PCM_OD_M = 4e-3
PCM_ID_M = 3e-3
PCM_H_M = 5e-3

PCM_VOLUME = (
    PCM_COUNT
    * (np.pi / 4)
    * (PCM_ID_M ** 2)
    * PCM_H_M
)


# ============================================================================
# Helpers
# ============================================================================

def clamp_battery_temp(
    battery_temp_c,
):
    """
    Restrict temperature
    to training domain.

    50
    70
    90
    """

    return min(
        TRAINING_BATTERY_TEMPS_C,
        key=lambda x: abs(
            x - battery_temp_c
        )
    )


# ============================================================================
# Feature Builder
# ============================================================================

def build_feature_row(
    FIDX,
    FEATURES,
    pcm_density,
    pcm_cp,
    pcm_k,
    pcm_viscosity,
    pcm_latent_heat,
    tsolidus_K,
    tliquidus_K,
    mass_flow_rate,
    battery_patch_temp_C,
    time_s,
):
    """
    Build single 20-feature vector.
    """

    Tm_K = (
        tsolidus_K
        + tliquidus_K
    ) / 2

    melt_range = (
        tliquidus_K
        - tsolidus_K
    )

    T_bat_K = (
        battery_patch_temp_C
        + 273.15
    )

    stefan = (
        pcm_cp
        * (T_bat_K - Tm_K)
        / pcm_latent_heat
    )

    pcm_storage = (
        pcm_density
        * pcm_latent_heat
    )

    flow_pcm_interaction = (
        mass_flow_rate
        * pcm_k
    )

    row = np.zeros(
        len(FEATURES)
    )

    row[FIDX["Density"]] = pcm_density
    row[FIDX["Specific_Heat"]] = pcm_cp
    row[FIDX["Thermal_Conductivity"]] = pcm_k
    row[FIDX["Viscosity"]] = pcm_viscosity
    row[FIDX["Latent_Heat"]] = pcm_latent_heat
    row[FIDX["Tm_K"]] = Tm_K
    row[FIDX["Melt_Range_K"]] = melt_range

    row[FIDX["Battery_Density"]] = BATTERY["density"]
    row[FIDX["Battery_Cp"]] = BATTERY["cp"]
    row[FIDX["Battery_k"]] = BATTERY["k"]

    row[
        FIDX[
            "Battery_HeatGeneration_W_m3"
        ]
    ] = BATTERY["heat_generation"]

    row[
        FIDX[
            "Mass_Flow_Rate_kg_s"
        ]
    ] = mass_flow_rate

    row[
        FIDX[
            "Initial_PCM_Temperature_C"
        ]
    ] = battery_patch_temp_C

    row[FIDX["Time_s"]] = time_s

    row[
        FIDX[
            "Stefan_Number"
        ]
    ] = stefan

    row[
        FIDX[
            "PCM_ThermalStorage"
        ]
    ] = pcm_storage

    row[
        FIDX[
            "Battery_ThermalCapacity_J_K"
        ]
    ] = BATTERY["thermal_capacity"]

    row[
        FIDX[
            "Tsolidus_K"
        ]
    ] = tsolidus_K

    row[
        FIDX[
            "Tliquidus_K"
        ]
    ] = tliquidus_K

    row[
        FIDX[
            "Flow_PCM_Interaction"
        ]
    ] = flow_pcm_interaction

    return row


# ============================================================================
# Prediction
# ============================================================================

def predict_batch(
    model,
    device,
    input_scaler,
    output_scaler,
    rows,
):
    """
    Batch PINN inference.
    """

    X_scaled = (
        input_scaler
        .transform(rows)
        .astype(np.float32)
    )

    with torch.no_grad():

        pred_scaled = (
            model(
                torch.tensor(
                    X_scaled,
                    device=device
                )
            )
            .cpu()
            .numpy()
        )

    pred_phys = (
        output_scaler
        .inverse_transform(
            pred_scaled
        )
    )

    return {
        "battery_temp":
            pred_phys[:, 0],

        "pcm_temp":
            pred_phys[:, 1],

        "lf":
            np.clip(
                pred_phys[:, 2],
                0,
                1
            ),

        "outlet_temp":
            np.maximum(
                pred_phys[:, 3],
                T_INLET_K
            ),
    }


# ============================================================================
# Time-Series Runner
# ============================================================================

def run_timeseries(
    model,
    device,
    input_scaler,
    output_scaler,
    FIDX,
    FEATURES,
    pcm,
    battery_temp_C,
    mass_flow_rate,
    time_vec=None,
):
    """
    Complete PINN simulation.

    Returns:
        results
        time_vector
    """

    if time_vec is None:
        time_vec = np.linspace(
            0,
            900,
            181
        )

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

            mass_flow_rate,
            battery_temp_C,
            t,
        )
        for t in time_vec
    ])

    results = predict_batch(
        model,
        device,
        input_scaler,
        output_scaler,
        rows,
    )

    return results, time_vec