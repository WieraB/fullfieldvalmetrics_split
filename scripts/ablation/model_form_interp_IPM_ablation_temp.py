import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ablation_funcs import mape_func, interval_score, regression_accuracy_metrics, load_data_temp

# -------------------------
# Set random seed
# -------------------------

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------------------------------------------------------
# Folders and files
# -----------------------------------------------------------------------------

# INPUT_FILE = (
#     Path.cwd()
#     / "images_pointsensors_pulse25X_v4"
#     / "pointsensors_dextremes.csv"
# )

INPUT_FILE = (
    Path(__file__).resolve().parents[1] / "mavm_nuisance_split" / "nen17_net250_na400" / "ablation" / "inputs"
    / "pointsensors_total_uncertainty_temperature.csv"
)
INPUT_FILE_TEMP = (
    Path(__file__).resolve().parents[1] / "mavm_nuisance_split" / "nen17_net250_na400" / "ablation" / "inputs"
    / "mean_sim_temperature.csv"
)
EXP_DIR = Path(__file__).resolve().parents[1] / "mavm_nuisance_split" / "nen17_net250_na400" / "ablation" / "outputs" / "ipm_interp_temp"

TOLERANCE=0.1
EPOCHS = 12000

# -----------------------------------------------------------------------------
# Define IPM model
# -----------------------------------------------------------------------------

class IPM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.center_head = nn.Linear(hidden_dim, 1)
        self.width_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)

        center = self.center_head(h)
        width = F.softplus(self.width_head(h))

        lower = center - width
        upper = center + width

        return lower, upper


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def ipm_loss(y, lower, upper, lam_width=1.0, lam_violation=100.0):

    # Interval width penalty to minimise width
    width = torch.mean(upper - lower)

    # Constraint violations (soft-constrained)
    violation_lower = torch.relu(lower - y)
    violation_upper = torch.relu(y - upper)

    violation = torch.mean(violation_lower + violation_upper)

    loss = lam_width * width + lam_violation * violation

    return loss, width, violation


# def ipm_loss(y, lower, upper, alpha=0.05):
#     """
#     Interval Score loss for a central (1-alpha) prediction interval.

#     Smaller is better.

#     alpha=0.05 -> 95% prediction interval.
#     """

#     # Interval width
#     width = upper - lower

#     # Penalty if y is below the lower bound
#     below = torch.relu(lower - y)

#     # Penalty if y is above the upper bound
#     above = torch.relu(y - upper)

#     # Interval score
#     score = (
#         width
#         + (2.0 / alpha) * below
#         + (2.0 / alpha) * above
#     )

#     # Mean score across the batch
#     loss = torch.mean(score)

#     return loss, torch.mean(width), torch.mean(below + above)

# def ipm_loss(y, lower, upper, alpha=0.05):
#     width = torch.mean(upper - lower)

#     violation_lower = torch.relu(lower - y)
#     violation_upper = torch.relu(y - upper)

#     violation = torch.mean(
#         violation_lower + violation_upper
#     )

#     loss = width + (2.0 / alpha) * violation

#     return loss, width, violation


def train_ipm(
    model,
    X,
    y,
    epochs=2000,
    lr=1e-3,
    patience=50,
    min_delta=1e-4,
    device="cpu",
    gradient_clip=None
):
    """
    Train an Interval Predictor Model (IPM) with early stopping.

    Parameters
    ----------
    model : torch.nn.Module
        IPM model.

    X : array-like
        Training input features.

    y : array-like
        Training target values.

    epochs : int, optional
        Maximum number of training epochs.

    lr : float, optional
        Adam learning rate.

    patience : int, optional
        Number of consecutive epochs without significant improvement
        before early stopping.

    min_delta : float, optional
        Minimum decrease in loss required to count as an improvement.

    device : str, optional
        Device to use, e.g. "cpu", "cuda", or "cuda:0".

    gradient_clip : float or None, optional
        Maximum gradient norm. If None, no gradient clipping is used.

    Returns
    -------
    model : torch.nn.Module
        Trained IPM with the best model state restored.

    epochs_completed : int
        Number of epochs actually completed.
    """

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but CUDA is not available."
        )

    print(f"Training IPM on: {device}")
    print(
        f"epochs: {epochs} - "
        f"lr: {lr} - "
        f"patience: {patience} - "
        f"min_delta: {min_delta}"
    )

    # -------------------------------------------------------------------------
    # Training data
    # -------------------------------------------------------------------------

    X = torch.tensor(
        X,
        dtype=torch.float32,
        device=device
    )

    y = torch.tensor(
        y,
        dtype=torch.float32,
        device=device
    ).view(-1, 1)

    # Move model to selected device
    model = model.to(device)

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    # -------------------------------------------------------------------------
    # Early stopping variables
    # -------------------------------------------------------------------------

    best_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    model.train()

    for epoch in range(epochs):

        optimizer.zero_grad()

        # Forward pass
        lower, upper = model(X)

        # IPM loss
        loss, width, violation = ipm_loss(
            y,
            lower,
            upper
        )

        # Backpropagation
        loss.backward()

        # Optional gradient clipping
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip
            )

        optimizer.step()

        current_loss = loss.item()

        # ---------------------------------------------------------------------
        # Check whether loss has improved
        # ---------------------------------------------------------------------

        if current_loss < best_loss - min_delta:

            best_loss = current_loss
            epochs_without_improvement = 0

            # Save best model state on CPU
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        else:

            epochs_without_improvement += 1

        # ---------------------------------------------------------------------
        # Progress information
        # ---------------------------------------------------------------------

        if (epoch + 1) % 200 == 0 or epoch == 0:

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {current_loss:.6f} | "
                f"Best: {best_loss:.6f} | "
                f"Width: {width.item():.6f} | "
                f"Violation: {violation.item():.6f}"
            )

        # ---------------------------------------------------------------------
        # Early stopping
        # ---------------------------------------------------------------------

        if epochs_without_improvement >= patience:

            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Loss has not improved by more than {min_delta} "
                f"for {patience} epochs."
            )

            break

    # -------------------------------------------------------------------------
    # Restore best model
    # -------------------------------------------------------------------------

    if best_model_state is not None:

        model.load_state_dict(best_model_state)

        model.to("cpu")

        print(
            f"Restored best model with loss = "
            f"{best_loss:.6f}"
        )

    epochs_completed = epoch + 1

    return model, epochs_completed


def fit_ipm_model(df,
                  d_type,
                  epochs=2000,
                  lr=1e-3,
                  hidden_dim=64,
                  device="cpu"):
    
    # -------------------------------------------------------------------------
    # Training data
    # -------------------------------------------------------------------------

    X_train = df['T'].values
    y_train = df[d_type].values.reshape(-1,1)

    # -------------------------------------------------------------------------
    # Normalise inputs and outputs
    # -------------------------------------------------------------------------

    X_scaler = StandardScaler()

    X_train = np.asarray(X_train).reshape(-1, 1)
    X_scaled = X_scaler.fit_transform(X_train)

    y_mean = y_train.mean()
    y_std = y_train.std()

    if y_std < 1e-12:
        y_std = 1.0

    y_scaled = (y_train - y_mean) / y_std

    # -------------------------------------------------------------------------
    # Train model
    # -------------------------------------------------------------------------

    model = IPM(
        input_dim=1,
        hidden_dim=hidden_dim
    )

    model, iter = train_ipm(
        model,
        X_scaled,
        y_scaled.flatten(),
        epochs=epochs,
        lr=lr,
        device=device
    )

    norm_params = {
        "X_scaler": X_scaler,
        "y_mean": y_mean,
        "y_std": y_std
    }

    return model, norm_params, iter


def predict_ipm(model,
                norm_params,
                X_query):

    X_query = np.asarray(X_query).reshape(-1, 1)
    X_scaled = norm_params["X_scaler"].transform(X_query)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )

    model.eval()

    with torch.no_grad():
        lower_norm, upper_norm = model(X_tensor)

    lower_norm = lower_norm.numpy().flatten()
    upper_norm = upper_norm.numpy().flatten()

    pred_norm = 0.5 * (lower_norm + upper_norm)

    # Denormalise predictions

    y_mean = norm_params["y_mean"]
    y_std = norm_params["y_std"]

    pred_mean = pred_norm * y_std + y_mean
    lower = lower_norm * y_std + y_mean
    upper = upper_norm * y_std + y_mean

    return pred_mean, lower, upper


def evaluate_training_points(model,
                             norm_params,
                             df,
                             d_type):

    X_query = df['T'].values

    pred_mean, lower, upper = predict_ipm(model, norm_params, X_query)

    out_df = pd.DataFrame({
        "TC": df.index,
        "measured": df[d_type].values,
        "predicted": pred_mean,
        "lower_95": lower,
        "upper_95": upper
    })

    return out_df


def leave_one_out_ablation(df,
                           d_type,
                           epochs=2000,
                           lr=1e-3,
                           hidden_dim=64,
                           device="cpu"):

    results = []

    tc_names = list(df.index)

    for excluded_tc in tc_names:

        print(f"Excluding {excluded_tc}")

        train_df = df.drop(
            index=excluded_tc
        )

        test_df = df.loc[
            [excluded_tc]
        ]

        model, norm_params, iter = fit_ipm_model(
            train_df,
            d_type,
            epochs,
            lr,
            hidden_dim,
            device=device
        )

        pred_mean, lower, upper = predict_ipm(
            model,
            norm_params,
            test_df['T'].values
        )

        measured = test_df[d_type].values[0]

        predicted = pred_mean[0]

        lower_95 = lower[0]
        upper_95 = upper[0]

        error = predicted - measured

        abs_error = abs(error)

        rel_error = (
            abs_error / abs(measured)
            if abs(measured) > 1e-12
            else np.nan
        )

        within_pi = (
            lower_95 <= measured <= upper_95
        )

        if measured < lower_95:
            pi_error = lower_95 - measured

        elif measured > upper_95:
            pi_error = measured - upper_95

        else:
            pi_error = 0.0

        pi_width = upper_95 - lower_95

        iscore = interval_score(
            measured,
            lower_95,
            upper_95
        )

        results.append({
            "excluded_tc": excluded_tc,
            "measured": measured,
            "predicted": predicted,

            # mean-based errors
            "error": error,
            "abs_error": abs_error,
            "rel_error": rel_error,

            # confidence interval
            "lower_95": lower_95,
            "upper_95": upper_95,
            "pi_width": pi_width,

            # interval-based errors
            "within_pi": within_pi,
            "pi_error": pi_error,
            "interval_score": iscore,

            "iteration": iter
        })

    return pd.DataFrame(results)


def main():

    EXP_DIR.mkdir(parents=True, exist_ok=True)

    train_dir = EXP_DIR / "training_predictions"
    ablation_dir = EXP_DIR / "ablation_results"

    train_dir.mkdir(exist_ok=True)
    ablation_dir.mkdir(exist_ok=True)

    merged_df, d_types = load_data_temp(INPUT_FILE, INPUT_FILE_TEMP)

    summary_errors = []

    for d_type in d_types:

        print("="*80)
        print(f"Processing {d_type}")
                
        # ---------------------------------------------------------------------
        # Fit model to all TC data
        # ---------------------------------------------------------------------

        model, norm_params = fit_ipm_model(merged_df, d_type, epochs=EPOCHS, lr=1e-3)

        # -------------------------------------------------------------
        # Training predictions using the model fitted to all TC data
        # -------------------------------------------------------------

        train_pred_df = evaluate_training_points(
            model,
            norm_params,
            merged_df,
            d_type
        )

        train_pred_df.to_csv(train_dir / f"{d_type}_training_predictions.csv", index=False)

        # -------------------------------------------------------------
        # Ablation study
        # -------------------------------------------------------------

        ablation_df = leave_one_out_ablation(merged_df, d_type, epochs=EPOCHS, lr=1e-3)

        ablation_df.to_csv(ablation_dir / f"{d_type}_ablation.csv", index=False)

        mae = mean_absolute_error(ablation_df["measured"], ablation_df["predicted"])
        rmse = np.sqrt(mean_squared_error(ablation_df["measured"], ablation_df["predicted"]))


        mape = mape_func(ablation_df["measured"], ablation_df["predicted"])

        within_tol = (
            np.abs(
                ablation_df["predicted"] -
                ablation_df["measured"]
            )
            <=
            TOLERANCE*np.abs(ablation_df["measured"])
        )

        accuracy = within_tol.mean()

        metrics = regression_accuracy_metrics(
            ablation_df["measured"],
            ablation_df["predicted"],
            tolerance=TOLERANCE
        )
        
        summary_errors.append(
            {
            "d_type": d_type,

            # mean-based errors
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,

            "mean_abs_error":ablation_df["abs_error"].mean(),

            "mean_rel_error": ablation_df["rel_error"].mean(skipna=True),

            # prediction interval
            "mean_pi_width": ablation_df["pi_width"].mean(),

            # interval-based errors
            "mean_pi_error": ablation_df["pi_error"].mean(),
            "pi_coverage": ablation_df["within_pi"].mean(),
            "mean_interval_score": ablation_df["interval_score"].mean(),

            # prediction accuracy
            "accuracy_2pct": accuracy,
            "accuracy_2pct_matrix": metrics["accuracy"],
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "TN": metrics["TN"],
            "FN": metrics["FN"],
            }
        )

    summary_df = pd.DataFrame(summary_errors)

    summary_df.to_csv(EXP_DIR / "ablation_summary.csv", index=False)

    print(summary_df)


def test_model(model_type, device):

    lr = model_type["lr"]
    hidden_dim = model_type["hidden_dim"]
    model_type_print = f"{lr}_{hidden_dim}"

    EXP_DIR.mkdir(parents=True, exist_ok=True)

    coeff_dir = EXP_DIR / "model_coefficients"
    train_dir = EXP_DIR / "training_predictions"
    ablation_dir = EXP_DIR / "ablation_results"

    for d in [coeff_dir, train_dir, ablation_dir]:
        d.mkdir(parents=True, exist_ok=True)

    merged_df, d_types = load_data_temp(INPUT_FILE, INPUT_FILE_TEMP)

    summary_errors = []

    for d_type in d_types:

        print("="*80)
        print(f"Processing {d_type}")
                
        # ---------------------------------------------------------------------
        # Fit model to all TC data
        # ---------------------------------------------------------------------

        model_fitted, norm_params, iter = fit_ipm_model(merged_df, d_type, epochs=EPOCHS, 
                                           lr=lr, hidden_dim=hidden_dim, device=device)

        checkpoint = {
            "model_state_dict": model_fitted.state_dict(),
            "norm_params": norm_params,
            "d_type": d_type,
        }
        
        torch.save(checkpoint, coeff_dir / f"{d_type}_gpr_model_{model_type_print}.pth")

        # -------------------------------------------------------------
        # Training predictions using the model fitted to all TC data
        # -------------------------------------------------------------

        train_pred_df = evaluate_training_points(
            model_fitted,
            norm_params,
            merged_df,
            d_type
        )

        train_pred_df.to_csv(train_dir / f"{d_type}_training_predictions_{model_type_print}.csv", index=False)

        # -------------------------------------------------------------
        # Ablation study
        # -------------------------------------------------------------

        ablation_df = leave_one_out_ablation(merged_df, d_type, epochs=EPOCHS, 
                                             lr=lr, hidden_dim=hidden_dim, device=device)

        ablation_df.to_csv(ablation_dir / f"{d_type}_ablation_{model_type_print}.csv", index=False)

        mae = mean_absolute_error(ablation_df["measured"], ablation_df["predicted"])
        rmse = np.sqrt(mean_squared_error(ablation_df["measured"], ablation_df["predicted"]))


        mape = mape_func(ablation_df["measured"], ablation_df["predicted"])

        within_tol = (
            np.abs(
                ablation_df["predicted"] -
                ablation_df["measured"]
            )
            <=
            TOLERANCE*np.abs(ablation_df["measured"])
        )

        accuracy = within_tol.mean()

        metrics = regression_accuracy_metrics(
            ablation_df["measured"],
            ablation_df["predicted"],
            tolerance=TOLERANCE
        )
        
        summary_errors.append(
            {
            "d_type": d_type,

            # mean-based errors
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,

            "mean_abs_error":ablation_df["abs_error"].mean(),

            "mean_rel_error": ablation_df["rel_error"].mean(skipna=True),

            # prediction interval
            "mean_pi_width": ablation_df["pi_width"].mean(),

            # interval-based errors
            "mean_pi_error": ablation_df["pi_error"].mean(),
            "pi_coverage": ablation_df["within_pi"].mean(),
            "mean_interval_score": ablation_df["interval_score"].mean(),

            # prediction accuracy
            "accuracy_2pct": accuracy,
            "accuracy_2pct_matrix": metrics["accuracy"],
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "TN": metrics["TN"],
            "FN": metrics["FN"],
            }
        )

    summary_df = pd.DataFrame(summary_errors)

    summary_df.to_csv(EXP_DIR / f"ablation_summary_{model_type_print}.csv", index=False)

    print(summary_df)

# if __name__ == "__main__":
#     main()



lrs = np.array([0.003, 0.01, 0.03, 0.1])
hidden_dims = np.array([1, 4, 16, 32, 64])

# model_type = {
#   "lr": 1e-3,
#   "hidden_dim": 64
# }

# test_model(model_type, device="cpu")


for lr in lrs:
    for hidden_dim in hidden_dims:
        model_type = {
          "lr": lr,
          "hidden_dim": hidden_dim
        }
        test_model(model_type, device="cuda")