"""
utils/validation.py

CFD Validation Utilities

Compares PINN predictions against CFD ground truth
from Extended_Master_DB_V2.xlsx
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .inference import (
    build_feature_row,
    predict_batch,
)


# ============================================================================
# Load Validation Dataset
# ============================================================================

def load_validation_database(
    excel_path,
):
    """
    Load CFD database.

    Parameters
    ----------
    excel_path : str

    Returns
    -------
    pd.DataFrame
    """

    return pd.read_csv(excel_path)


# ============================================================================
# Build Validation Features
# ============================================================================

def build_validation_matrix(
    df,
    FIDX,
    FEATURES,
):
    """
    Convert CFD database rows
    into PINN feature matrix.
    """

    rows = np.array([
        build_feature_row(
            FIDX,
            FEATURES,

            row["Density"],
            row["Specific_Heat"],
            row["Thermal_Conductivity"],
            row["Viscosity"],
            row["Latent_Heat"],

            row["Tsolidus_K"],
            row["Tliquidus_K"],

            row["Mass_Flow_Rate_kg_s"],
            row["Battery Patch Temperature in C"],

            row["Time_s"]

        )
        for _, row in df.iterrows()
    ])

    return rows


# ============================================================================
# Run PINN Validation
# ============================================================================

def run_validation(
    df,
    model,
    device,
    input_scaler,
    output_scaler,
    FIDX,
    FEATURES,
):
    """
    Run PINN on full CFD database.

    Returns
    -------
    validation dataframe
    """

    X = build_validation_matrix(
        df,
        FIDX,
        FEATURES,
    )

    pred = predict_batch(
        model,
        device,
        input_scaler,
        output_scaler,
        X,
    )

    df_val = df.copy()

    df_val["pinn_battery_temp_K"] = pred[
        "battery_temp"
    ]

    df_val["pinn_pcm_temp_K"] = pred[
        "pcm_temp"
    ]

    df_val["pinn_lf"] = pred[
        "lf"
    ]

    df_val["pinn_outlet_temp_K"] = pred[
        "outlet_temp"
    ]

    return df_val


# ============================================================================
# Error Metrics
# ============================================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    """
    Standard validation metrics.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    mape = np.mean(
        np.abs(
            (y_true - y_pred)
            / (np.abs(y_true) + 1e-9)
        )
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
    }


# ============================================================================
# Full Validation Summary
# ============================================================================

def validation_summary(
    df_val,
):
    """
    Returns validation metrics
    for all PINN outputs.
    """

    mask_t0 = (
        df_val["Time_s"] > 0
    )

    battery_metrics = calculate_metrics(
        df_val["battery-temp"],
        df_val["pinn_battery_temp_K"],
    )

    pcm_metrics = calculate_metrics(
        df_val["pcmtemp"],
        df_val["pinn_pcm_temp_K"],
    )

    lf_metrics = calculate_metrics(
        df_val["lf"],
        df_val["pinn_lf"],
    )

    outlet_metrics = calculate_metrics(
        df_val.loc[
            mask_t0,
            "outlet-temp"
        ],
        df_val.loc[
            mask_t0,
            "pinn_outlet_temp_K"
        ],
    )

    return {
        "battery": battery_metrics,
        "pcm": pcm_metrics,
        "lf": lf_metrics,
        "outlet": outlet_metrics,
    }


# ============================================================================
# Absolute Error Columns
# ============================================================================

def add_error_columns(
    df_val,
):
    """
    Adds error columns
    to validation dataframe.
    """

    df_val = df_val.copy()

    df_val["err_battery_K"] = (
        df_val["pinn_battery_temp_K"]
        - df_val["battery-temp"]
    ).abs()

    df_val["err_pcm_K"] = (
        df_val["pinn_pcm_temp_K"]
        - df_val["pcmtemp"]
    ).abs()

    df_val["err_lf"] = (
        df_val["pinn_lf"]
        - df_val["lf"]
    ).abs()

    mask = (
        df_val["Time_s"] > 0
    )

    df_val["err_outlet_K"] = 0.0

    df_val.loc[
        mask,
        "err_outlet_K"
    ] = (
        df_val.loc[
            mask,
            "pinn_outlet_temp_K"
        ]
        -
        df_val.loc[
            mask,
            "outlet-temp"
        ]
    ).abs()

    return df_val


# ============================================================================
# Scenario Filter
# ============================================================================

def filter_case(
    df_val,
    pcm_name=None,
    battery_temp=None,
    flow_rate=None,
):
    """
    Extract single CFD case.
    """

    df_case = df_val.copy()

    if pcm_name is not None:
        df_case = df_case[
            df_case["PCM_Name"]
            == pcm_name
        ]

    if battery_temp is not None:
        df_case = df_case[
            df_case[
                "Initial_Battery_Patch_Temp_C"
            ] == battery_temp
        ]

    if flow_rate is not None:
        df_case = df_case[
            np.isclose(
                df_case[
                    "Mass_Flow_Rate_kg_s"
                ],
                flow_rate,
            )
        ]

    return df_case.sort_values(
        "Time_s"
    )


# ============================================================================
# Parity Data
# ============================================================================

def get_parity_data(
    df_val,
):
    """
    Convenient dictionary for parity plots.
    """

    return {
        "battery": (
            df_val["battery-temp"],
            df_val["pinn_battery_temp_K"],
        ),

        "pcm": (
            df_val["pcmtemp"],
            df_val["pinn_pcm_temp_K"],
        ),

        "lf": (
            df_val["lf"],
            df_val["pinn_lf"],
        ),

        "outlet": (
            df_val[
                df_val["Time_s"] > 0
            ]["outlet-temp"],

            df_val[
                df_val["Time_s"] > 0
            ]["pinn_outlet_temp_K"],
        ),
    }


# ============================================================================
# Export Validation Results
# ============================================================================

def export_validation_csv(
    df_val,
    output_path,
):
    """
    Save validation dataframe.
    """

    df_val.to_csv(
        output_path,
        index=False,
    )

    return output_path