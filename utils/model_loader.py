"""
utils/model_loader.py

PINN model loading utilities.
"""

import torch
import torch.nn as nn
import joblib


# ============================================================================
# PINN Architecture
# ============================================================================

class PINN(nn.Module):
    """
    Identical architecture used during training.

    Inputs  : 20
    Hidden  : 256
    Depth   : 6
    Outputs : 4

    Outputs:
        Battery Temperature
        PCM Temperature
        Liquid Fraction
        Outlet Temperature
    """

    def __init__(
        self,
        in_features=20,
        out_features=4,
        hidden=256,
        depth=6,
    ):
        super().__init__()

        layers = [
            nn.Linear(in_features, hidden),
            nn.SiLU()
        ]

        for _ in range(depth - 1):
            layers.extend([
                nn.Linear(hidden, hidden),
                nn.SiLU()
            ])

        layers.append(
            nn.Linear(hidden, out_features)
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Load Model
# ============================================================================

def load_model_artifacts(
    pth_path,
    input_scaler_path,
    output_scaler_path,
    features_path,
    targets_path,
):
    """
    Loads:

    - PINN model
    - input scaler
    - output scaler
    - feature columns
    - target columns

    Returns
    -------
    dict
    """

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        pth_path,
        map_location=device
    )

    model = PINN().to(device)

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    input_scaler = joblib.load(
        input_scaler_path
    )

    output_scaler = joblib.load(
        output_scaler_path
    )

    features = joblib.load(
        features_path
    )

    targets = joblib.load(
        targets_path
    )

    return {
        "model": model,
        "device": device,
        "input_scaler": input_scaler,
        "output_scaler": output_scaler,
        "features": features,
        "targets": targets,
        "checkpoint": checkpoint,
    }