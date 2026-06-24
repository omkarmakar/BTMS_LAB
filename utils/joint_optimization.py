"""
utils/joint_optimization.py

Notebook Section 7
Joint PCM + Flow Optimization

Simultaneously optimize:

1. PCM properties
2. Coolant mass flow rate

Objective:
    Minimize battery temperature
    while minimizing pumping effort.

Used by:
    Joint Optimization Tab
"""

import numpy as np

from scipy.optimize import differential_evolution

from .inference import (
    FLOW_LIMITS,
    build_feature_row,
    predict_batch,
)


# ============================================================================
# PCM SEARCH SPACE
# ============================================================================

PCM_BOUNDS = {
    "density":     (700.0, 1000.0),
    "cp":          (1500.0, 3000.0),
    "k":           (0.10, 0.60),
    "viscosity":   (0.001, 0.010),
    "latent_heat": (165000.0, 200000.0),
    "tsolidus":    (299.15, 341.15),
    "melt_range":  (1.0, 10.0),
}


# ============================================================================
# EVALUATION TIMES
# ============================================================================

DEFAULT_EVAL_TIMES = np.array([
    150,
    300,
    450,
    600,
    750,
    900,
])


# ============================================================================
# BUILD PCM
# ============================================================================

def vector_to_pcm(
    x,
):
    """
    Convert optimizer vector
    into PCM dictionary.
    """

    density = x[0]
    cp = x[1]
    k = x[2]
    viscosity = x[3]
    latent_heat = x[4]

    tsolidus = x[5]

    melt_range = x[6]

    tliquidus = (
        tsolidus
        + melt_range
    )

    return {
        "density": density,
        "cp": cp,
        "k": k,
        "viscosity": viscosity,
        "latent_heat": latent_heat,
        "tsolidus": tsolidus,
        "tliquidus": tliquidus,
    }


# ============================================================================
# PREDICT PEAK BATTERY TEMPERATURE
# ============================================================================

def evaluate_candidate(
    model,
    device,
    input_scaler,
    output_scaler,
    FIDX,
    FEATURES,
    pcm,
    mass_flow_rate,
    battery_temp_C,
    eval_times,
):
    """
    Evaluate candidate PCM + flow.
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

            mass_flow_rate,
            battery_temp_C,
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

    T_bat = (
        result["battery_temp"]
        - 273.15
    )

    T_pcm = (
        result["pcm_temp"]
        - 273.15
    )

    LF = result["lf"]

    T_out = (
        result["outlet_temp"]
        - 273.15
    )

    return {
        "peak_battery_temp":
            float(T_bat.max()),

        "avg_battery_temp":
            float(T_bat.mean()),

        "peak_pcm_temp":
            float(T_pcm.max()),

        "avg_lf":
            float(LF.mean()),

        "final_lf":
            float(LF[-1]),

        "peak_outlet_temp":
            float(T_out.max()),
    }


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def joint_objective(
    x,
    model,
    device,
    input_scaler,
    output_scaler,
    FIDX,
    FEATURES,
    battery_temp_C,
    target_temperature,
    flow_weight,
    temperature_weight,
    eval_times,
):
    """
    Optimization objective.

    x contains:

    density
    cp
    k
    viscosity
    latent_heat
    tsolidus
    melt_range
    flow_rate
    """

    pcm = vector_to_pcm(
        x[:7]
    )

    flow_rate = x[7]

    result = evaluate_candidate(
        model,
        device,
        input_scaler,
        output_scaler,
        FIDX,
        FEATURES,
        pcm,
        flow_rate,
        battery_temp_C,
        eval_times,
    )

    Tmax = result[
        "peak_battery_temp"
    ]

    penalty = (
        1000.0
        * max(
            0.0,
            Tmax - target_temperature
        ) ** 2
    )

    flow_term = (
        flow_rate
        / FLOW_LIMITS["max"]
    )

    objective = (
        temperature_weight
        * Tmax
        +
        flow_weight
        * flow_term
        +
        penalty
    )

    return objective


# ============================================================================
# OPTIMIZATION DRIVER
# ============================================================================

def run_joint_optimization(
    model,
    device,
    input_scaler,
    output_scaler,
    FIDX,
    FEATURES,

    battery_temp_C,

    target_temperature=70.0,

    flow_weight=0.20,
    temperature_weight=1.0,

    maxiter=150,
    popsize=12,

    seed=42,

    eval_times=None,
):
    """
    Main optimization routine.
    """

    if eval_times is None:
        eval_times = DEFAULT_EVAL_TIMES

    bounds = [

        PCM_BOUNDS["density"],
        PCM_BOUNDS["cp"],
        PCM_BOUNDS["k"],
        PCM_BOUNDS["viscosity"],
        PCM_BOUNDS["latent_heat"],
        PCM_BOUNDS["tsolidus"],
        PCM_BOUNDS["melt_range"],

        (
            FLOW_LIMITS["min"],
            FLOW_LIMITS["max"],
        ),
    ]

    result = differential_evolution(

        func=joint_objective,

        bounds=bounds,

        args=(
            model,
            device,
            input_scaler,
            output_scaler,
            FIDX,
            FEATURES,
            battery_temp_C,
            target_temperature,
            flow_weight,
            temperature_weight,
            eval_times,
        ),

        maxiter=maxiter,
        popsize=popsize,

        mutation=(0.5, 1.0),
        recombination=0.9,

        tol=1e-4,

        seed=seed,

        polish=True,
        disp=False,
    )

    x = result.x

    pcm = vector_to_pcm(
        x[:7]
    )

    flow_rate = x[7]

    metrics = evaluate_candidate(
        model,
        device,
        input_scaler,
        output_scaler,
        FIDX,
        FEATURES,
        pcm,
        flow_rate,
        battery_temp_C,
        eval_times,
    )

    return {

        "pcm": pcm,

        "flow_rate_kg_s":
            flow_rate,

        "flow_rate_g_s":
            flow_rate * 1000.0,

        "metrics":
            metrics,

        "objective":
            result.fun,

        "success":
            result.success,

        "message":
            result.message,

        "nfev":
            result.nfev,

        "nit":
            result.nit,

        "raw_result":
            result,
    }


# ============================================================================
# COMPARISON AGAINST STANDARD PCMS
# ============================================================================

def compare_against_standard(
    optimized_result,
    pcm_database,
):
    """
    Compare optimized PCM
    against RT27 / RT45 / RT55 / RT70.
    """

    rows = []

    opt_pcm = optimized_result["pcm"]

    rows.append({
        "name": "Optimized PCM",
        "density": opt_pcm["density"],
        "cp": opt_pcm["cp"],
        "k": opt_pcm["k"],
        "latent_heat": opt_pcm["latent_heat"],
        "Tm_C":
            (
                (
                    opt_pcm["tsolidus"]
                    +
                    opt_pcm["tliquidus"]
                ) / 2
            ) - 273.15,
    })

    for name, pcm in pcm_database.items():

        rows.append({
            "name": name,
            "density": pcm["density"],
            "cp": pcm["cp"],
            "k": pcm["k"],
            "latent_heat": pcm["latent_heat"],
            "Tm_C":
                (
                    (
                        pcm["tsolidus"]
                        +
                        pcm["tliquidus"]
                    ) / 2
                ) - 273.15,
        })

    return rows


# ============================================================================
# SUMMARY
# ============================================================================

def optimization_summary(
    result,
):
    """
    Dashboard-ready metrics.
    """

    pcm = result["pcm"]

    metrics = result["metrics"]

    return {

        "flow_g_s":
            result["flow_rate_g_s"],

        "peak_battery_temp_C":
            metrics["peak_battery_temp"],

        "avg_battery_temp_C":
            metrics["avg_battery_temp"],

        "avg_lf":
            metrics["avg_lf"],

        "final_lf":
            metrics["final_lf"],

        "density":
            pcm["density"],

        "cp":
            pcm["cp"],

        "k":
            pcm["k"],

        "viscosity":
            pcm["viscosity"],

        "latent_heat":
            pcm["latent_heat"],

        "Tm_C":
            (
                (
                    pcm["tsolidus"]
                    +
                    pcm["tliquidus"]
                ) / 2
            ) - 273.15,
    }