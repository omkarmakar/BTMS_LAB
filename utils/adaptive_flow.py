"""
utils/adaptive_flow.py

Notebook Section 5
Adaptive Flow Rate Control

Given:
    PCM
    Battery temperature profile

Determine:
    Required mass flow rate profile
    to satisfy battery temperature limit
"""

import numpy as np

from scipy.optimize import brentq
from scipy.interpolate import interp1d

from .inference import (
    FLOW_LIMITS,
    build_feature_row,
    predict_batch,
)


# ============================================================================
# Default Profile
# ============================================================================

DEFAULT_PROFILE = [
    (0,   50.0),
    (100, 50.0),
    (200, 70.0),
    (300, 70.0),
    (400, 70.0),
    (500, 90.0),
    (600, 90.0),
    (700, 70.0),
    (800, 70.0),
    (900, 50.0),
]


# ============================================================================
# Profile Utilities
# ============================================================================

def profile_to_arrays(
    profile,
):
    """
    Convert profile list
    into numpy arrays.
    """

    times = np.array(
        [p[0] for p in profile]
    )

    temps = np.array(
        [p[1] for p in profile]
    )

    return times, temps


# ============================================================================
# Peak Battery Temperature
# ============================================================================

def peak_battery_temperature(
    model,
    device,
    input_scaler,
    output_scaler,
    FIDX,
    FEATURES,
    pcm,
    mdot,
    battery_temp,
    evaluation_times,
):
    """
    Peak battery temperature
    over look-ahead horizon.
    """

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
        for t in evaluation_times
    ])

    result = predict_batch(
        model,
        device,
        input_scaler,
        output_scaler,
        rows,
    )

    return (
        result["battery_temp"]
        - 273.15
    ).max()


# ============================================================================
# Liquid Fraction Estimation
# ============================================================================

def estimate_liquid_fraction(
    model,
    device,
    input_scaler,
    output_scaler,
    FIDX,
    FEATURES,
    pcm,
    mdot,
    battery_temp,
    eval_time=150.0,
):
    """
    Estimate LF
    at midpoint.
    """

    row = np.array([
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
            eval_time,
        )
    ])

    result = predict_batch(
        model,
        device,
        input_scaler,
        output_scaler,
        row,
    )

    return float(
        result["lf"][0]
    )


# ============================================================================
# Required Flow Rate
# ============================================================================

def find_required_flow_rate(
    model,
    device,
    input_scaler,
    output_scaler,
    FIDX,
    FEATURES,
    pcm,
    battery_temp,
    target_temp,
    look_ahead_times,
):
    """
    Notebook bisection solver.
    """

    Tmax_min = peak_battery_temperature(
        model,
        device,
        input_scaler,
        output_scaler,
        FIDX,
        FEATURES,
        pcm,
        FLOW_LIMITS["min"],
        battery_temp,
        look_ahead_times,
    )

    Tmax_max = peak_battery_temperature(
        model,
        device,
        input_scaler,
        output_scaler,
        FIDX,
        FEATURES,
        pcm,
        FLOW_LIMITS["max"],
        battery_temp,
        look_ahead_times,
    )

    if Tmax_min <= target_temp:
        return FLOW_LIMITS["min"]

    if Tmax_max > target_temp:
        return FLOW_LIMITS["max"]

    try:

        mdot = brentq(
            lambda m:
            peak_battery_temperature(
                model,
                device,
                input_scaler,
                output_scaler,
                FIDX,
                FEATURES,
                pcm,
                m,
                battery_temp,
                look_ahead_times,
            )
            - target_temp,

            FLOW_LIMITS["min"],
            FLOW_LIMITS["max"],

            xtol=1e-5,
            maxiter=50,
        )

        return mdot

    except ValueError:

        return FLOW_LIMITS["max"]


# ============================================================================
# Main Adaptive Solver
# ============================================================================

def run_adaptive_controller(
    model,
    device,
    input_scaler,
    output_scaler,
    FIDX,
    FEATURES,
    pcm,
    temperature_profile,
    target_temperature,
    look_ahead_seconds=300,
    n_lookahead_points=30,
):
    """
    Main adaptive flow solver.

    Returns dataframe-ready dictionary.
    """

    look_ahead_times = np.linspace(
        0,
        look_ahead_seconds,
        n_lookahead_points,
    )

    profile_times = []
    profile_temps = []

    required_flow = []
    predicted_peak_temp = []
    predicted_lf = []

    for (
        time_point,
        battery_temp
    ) in temperature_profile:

        mdot_req = find_required_flow_rate(
            model,
            device,
            input_scaler,
            output_scaler,
            FIDX,
            FEATURES,
            pcm,
            battery_temp,
            target_temperature,
            look_ahead_times,
        )

        Tmax = peak_battery_temperature(
            model,
            device,
            input_scaler,
            output_scaler,
            FIDX,
            FEATURES,
            pcm,
            mdot_req,
            battery_temp,
            look_ahead_times,
        )

        lf = estimate_liquid_fraction(
            model,
            device,
            input_scaler,
            output_scaler,
            FIDX,
            FEATURES,
            pcm,
            mdot_req,
            battery_temp,
        )

        profile_times.append(
            time_point
        )

        profile_temps.append(
            battery_temp
        )

        required_flow.append(
            mdot_req
        )

        predicted_peak_temp.append(
            Tmax
        )

        predicted_lf.append(
            lf
        )

    return {
        "time_s":
            np.array(profile_times),

        "battery_temp_C":
            np.array(profile_temps),

        "required_flow_kg_s":
            np.array(required_flow),

        "required_flow_g_s":
            np.array(required_flow) * 1000,

        "predicted_peak_temp_C":
            np.array(predicted_peak_temp),

        "predicted_lf":
            np.array(predicted_lf),
    }


# ============================================================================
# Continuous Controller
# ============================================================================

def build_continuous_controller(
    time_points,
    flow_points,
    interpolation="cubic",
    n_samples=500,
):
    """
    Generate continuous
    flow-control signal.
    """

    t_cont = np.linspace(
        time_points.min(),
        time_points.max(),
        n_samples,
    )

    interpolator = interp1d(
        time_points,
        flow_points,
        kind=interpolation,
        fill_value="extrapolate",
    )

    flow_cont = interpolator(
        t_cont
    )

    flow_cont = np.clip(
        flow_cont,
        FLOW_LIMITS["min"],
        FLOW_LIMITS["max"],
    )

    return {
        "time_s": t_cont,
        "flow_kg_s": flow_cont,
        "flow_g_s": flow_cont * 1000,
    }


# ============================================================================
# Pump Energy Proxy
# ============================================================================

def estimate_pump_energy_proxy(
    flow_profile,
):
    """
    Relative pumping effort.

    Useful for optimisation.
    """

    flow = np.asarray(
        flow_profile
    )

    return np.trapezoid(
    flow ** 2
    )


# ============================================================================
# Summary Statistics
# ============================================================================

def controller_summary(
    controller_results,
):
    """
    Dashboard metrics.
    """

    flow = controller_results[
        "required_flow_g_s"
    ]

    Tmax = controller_results[
        "predicted_peak_temp_C"
    ]

    LF = controller_results[
        "predicted_lf"
    ]

    return {
        "max_flow_g_s":
            float(flow.max()),

        "min_flow_g_s":
            float(flow.min()),

        "avg_flow_g_s":
            float(flow.mean()),

        "max_battery_temp_C":
            float(Tmax.max()),

        "avg_battery_temp_C":
            float(Tmax.mean()),

        "avg_lf":
            float(LF.mean()),

        "pump_energy_proxy":
            float(
                estimate_pump_energy_proxy(
                    flow / 1000
                )
            ),
    }