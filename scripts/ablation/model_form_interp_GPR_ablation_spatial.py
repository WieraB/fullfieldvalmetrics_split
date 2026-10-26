import numpy as np
import pandas as pd
from pathlib import Path
import torch
import gpytorch
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ablation_funcs import mape_func, interval_score, regression_accuracy_metrics, load_data

# -------------------------
# Set random seed
# -------------------------

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Folders and files
# -----------------------------------------------------------------------------

# INPUT_FILE = (Path.cwd()
#     / "images_pointsensors_pulse25X_v4"
#     / "pointsensors_mavm.csv"
# )

# INPUT_FILE = (
#     Path.cwd()
#     / "images_pointsensors_pulse25X_v4"
#     / "pointsensors_dextremes.csv"
# )
INPUT_FILE = (
    Path(__file__).resolve().parents[1] / "mavm_nuisance_split" / "nen17_net250_na400" / "ablation" / "inputs"
    / "pointsensors_total_uncertainty_temperature.csv"
)

EXP_DIR = Path(__file__).resolve().parents[1] / "mavm_nuisance_split" / "nen17_net250_na400" / "ablation" / "outputs" / "interp_gpr_spatial"

TOLERANCE=0.1
# EPOCHS = 6000
EPOCHS = 12000

# -----------------------------------------------------------------------------
# Define GPR model
# -----------------------------------------------------------------------------

class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, 
                 kernel_type="RBF", ard_num_dims=1, nu=1.0):
        super().__init__(train_x, train_y, likelihood)

        # Mean function
        self.mean_module = gpytorch.means.ConstantMean()

        # Covariance / kernel
        match kernel_type:
            case "RBF":
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
                )
            case "Matern":
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=ard_num_dims)
                )
            case "Linear":
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.LinearKernel(ard_num_dims=ard_num_dims)
                )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def fit_gpr_model(
    df: pd.DataFrame,
    d_type: str,
    kernel_type: str,
    ard_num_dims: int,
    nu: float,
    training_iter: int = 500,
    lr: float = 0.01,
    patience: int = 50,
    min_delta: float = 1e-4,
    device: str = "cpu"
) -> tuple[ExactGPModel, GaussianLikelihood, dict, int]:
    """Train GPR model on CPU or GPU with early stopping.

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    d_type : str
        Validation metric type.
    kernel_type : str
        Kernel to use for GPR model.
    ard_num_dims : int
        Set this if you want a separate lengthscale for each input dimension.
    nu : float
        The smoothness parameter for Matern kernel.
    training_iter : int, optional
        Maximum number of training iterations.
    lr : float, optional
        Adam learning rate.
    patience : int, optional
        Number of consecutive iterations without significant improvement
        before stopping.
    min_delta : float, optional
        Minimum decrease in loss required to count as an improvement.
    device : str, optional
        Device to use for training. E.g. "cpu", "cuda", "cuda:0".
        If "cuda" is requested but CUDA is unavailable, a RuntimeError
        is raised.

    Returns
    -------
    tuple[ExactGPModel, GaussianLikelihood, dict, int]
        Trained GPR model, likelihood, normalisation data, and number
        of iterations completed.
    """

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but CUDA is not available."
        )

    print(f"Training GPR model on: {device}")
    print(
        f"kernel_type: {kernel_type} - "
        f"ard_num_dims: {ard_num_dims} - "
        f"nu: {nu} - "
        f"lr: {lr}"
    )


    # -------------------------------------------------------------------------
    # Training data
    # -------------------------------------------------------------------------

    X_train = df[['x', 'y', 'z']].values
    y_train = df[d_type].values.reshape(-1, 1)

    # -------------------------------------------------------------------------
    # Normalise inputs and outputs
    # -------------------------------------------------------------------------

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std < 1e-12] = 1.0

    X_norm = (X_train - X_mean) / X_std

    y_mean = y_train.mean()
    y_std = y_train.std()

    if y_std < 1e-12:
        y_std = 1.0

    y_norm = (y_train - y_mean) / y_std

    # -------------------------------------------------------------------------
    # Convert to torch tensors and move to selected device
    # -------------------------------------------------------------------------

    train_x = torch.tensor(
        X_norm,
        dtype=torch.float32,
        device=device
    )

    train_y = torch.tensor(
        y_norm.flatten(),
        dtype=torch.float32,
        device=device
    )

    # -------------------------------------------------------------------------
    # Model and likelihood
    # -------------------------------------------------------------------------

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    model = ExactGPModel(
        train_x,
        train_y,
        likelihood,
        kernel_type,
        ard_num_dims,
        nu
    ).to(device)

    # -------------------------------------------------------------------------
    # Train model
    # -------------------------------------------------------------------------

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(
        likelihood,
        model
    )

    # -------------------------------------------------------------------------
    # Early stopping variables
    # -------------------------------------------------------------------------

    best_loss = float("inf")
    best_model_state = None
    best_likelihood_state = None
    iterations_without_improvement = 0

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    for i in range(training_iter):

        optimizer.zero_grad()

        output = model(train_x)
        loss = -mll(output, train_y)

        loss.backward()
        optimizer.step()

        current_loss = loss.item()

        # ---------------------------------------------------------------------
        # Check whether loss has improved
        # ---------------------------------------------------------------------

        if current_loss < best_loss - min_delta:

            best_loss = current_loss
            iterations_without_improvement = 0

            # Save best model state on CPU.
            # This avoids unnecessarily retaining GPU memory for every
            # parameter in the saved state.
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            best_likelihood_state = {
                key: value.detach().cpu().clone()
                for key, value in likelihood.state_dict().items()
            }

        else:
            iterations_without_improvement += 1

        # ---------------------------------------------------------------------
        # Progress information
        # ---------------------------------------------------------------------

        if (i + 1) % 50 == 0:

            if kernel_type != "Linear":
                lengthscale = (
                    model.covar_module
                    .base_kernel
                    .lengthscale
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                )
    
                print(
                    f"Iter {i+1}/{training_iter} - "
                    f"Loss: {current_loss:.6f} - "
                    f"Best: {best_loss:.6f} - "
                    f"Noise: {likelihood.noise.item():.6f} - "
                    f"Lengthscale: {np.round(lengthscale, 3)}"
                )
            else:
    
                print(
                    f"Iter {i+1}/{training_iter} - "
                    f"Loss: {current_loss:.6f} - "
                    f"Best: {best_loss:.6f} - "
                    f"Noise: {likelihood.noise.item():.6f}"
                )

        # ---------------------------------------------------------------------
        # Early stopping
        # ---------------------------------------------------------------------

        if iterations_without_improvement >= patience:

            print(
                f"Early stopping at iteration {i + 1}. "
                f"Loss has not improved by more than {min_delta} "
                f"for {patience} iterations."
            )

            break

    # -------------------------------------------------------------------------
    # Restore best model
    # -------------------------------------------------------------------------

    if best_model_state is not None:

        model.load_state_dict(best_model_state)
        likelihood.load_state_dict(best_likelihood_state)

        model.to("cpu")
        likelihood.to("cpu")

        print(
            f"Restored best model with loss = "
            f"{best_loss:.6f}"
        )

    # -------------------------------------------------------------------------
    # Store normalisation parameters
    # -------------------------------------------------------------------------

    norm_params = {
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std
    }

    return model, likelihood, norm_params, i + 1



def predict_gpr(model_fitted: gpytorch.models.ExactGP,
                likelihood: GaussianLikelihood,
                norm_params: dict,
                X_query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Predict using trained GPR model.

    Parameters
    ----------
    model_fitted : gpytorch.models.ExactGP
        Trained GPR model
    likelihood : GaussianLikelihood
        Trained likelihood
    norm_params : dict
        Data normalisation parameters
    X_query : np.ndarray
        Points where the GPR model is to be evaluated

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Predictions
    """

    X_query_norm = ((X_query - norm_params["X_mean"]) / norm_params["X_std"])
    test_x = torch.tensor(X_query_norm, dtype=torch.float32)

    model_fitted.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():

        observed_pred = likelihood(model_fitted(test_x))
        pred_mean_norm = observed_pred.mean.numpy()
        lower_norm, upper_norm = (observed_pred.confidence_region())

    # Denormalise predictions

    y_mean = norm_params["y_mean"]
    y_std = norm_params["y_std"]

    pred_mean = pred_mean_norm * y_std + y_mean
    lower = lower_norm.numpy() * y_std + y_mean
    upper = upper_norm.numpy() * y_std + y_mean

    return pred_mean, lower, upper


def evaluate_training_points(model_fitted: gpytorch.models.ExactGP,
                             likelihood: GaussianLikelihood,
                             norm_params: dict,
                             df: pd.DataFrame,
                             d_type: str) -> pd.DataFrame:
    """ Evaluate trained GPR model at the training locations

    Parameters
    ----------
    model_fitted : gpytorch.models.ExactGP
        Trained GPR model
    likelihood : GaussianLikelihood
        Trained likelihood
    norm_params : dict
        Data normalisation parameters
    df : pd.DataFrame
        Training data
    d_type : str
        Validation metric type

    Returns
    -------
    pd.DataFrame
        Prediction results
    """

    X_query = df[['x', 'y', 'z']].values

    pred_mean, lower, upper = predict_gpr(model_fitted, likelihood, norm_params, X_query)

    out_df = pd.DataFrame({
        "TC": df.index,
        "measured": df[d_type].values,
        "predicted": pred_mean,
        "lower_95": lower,
        "upper_95": upper
    })

    return out_df


def leave_one_out_ablation(df: pd.DataFrame, 
                           d_type: str,
                           kernel_type: str,
                           ard_num_dims: int,
                           nu: float,
                           training_iter: int = 100,
                           lr: float = 0.01,
                           device: str = "cpu") -> pd.DataFrame:
    """Perform ablation study by excluding one TC data 
    and fitting the model to the rest of the TC data

    Parameters
    ----------
    df : pd.DataFrame
        TC validation data
    d_type : str
        Validation metric type
    kernel_type : str
        Kernel to use for GPR model.
    ard_num_dims : int
        Set this if you want a separate lengthscale for each input dimension.
    nu : float
        The smoothness parameter for Matern kernel.
    training_iter : int, optional
        Number of training iterations
    lr : float, optional
        Adam learning rate.
    device : str, optional
        Device to use for training. E.g. "cpu", "cuda", "cuda:0".
        If "cuda" is requested but CUDA is unavailable, a RuntimeError
        is raised.

    Returns
    -------
    pd.DataFrame
        Ablation study results
    """

    tc_names = list(df.index)

    results_list = []

    for excluded_tc in tc_names:

        print(f"Excluding {excluded_tc}")

        train_df = df.drop(index=excluded_tc)
        test_df = df.loc[[excluded_tc]]

        # Fit GPR model
        model, likelihood, norm_params, iter = fit_gpr_model(train_df, d_type, kernel_type, 
                                                       ard_num_dims, nu, training_iter, lr, device=device)

        # Predict excluded point
        X_test = test_df[['x', 'y', 'z']].values

        pred_mean, lower, upper = predict_gpr(model, likelihood, norm_params, X_test)

        measured = test_df[d_type].values[0]

        predicted = pred_mean[0]
        lower_95 = lower[0]
        upper_95 = upper[0]

        error = predicted - measured
        abs_error = abs(error)

        within_pi = (
            lower_95 <= measured <= upper_95
        )

        if measured < lower_95:
            pi_error = lower_95 - measured
        elif measured > upper_95:
            pi_error = measured - upper_95
        else:
            pi_error = 0.0

        if np.abs(measured) > 1e-12:
            rel_error = abs(error) / abs(measured)
        else:
            rel_error = np.nan

        pi_width = upper_95 - lower_95

        iscore = interval_score(
            measured,
            lower_95,
            upper_95
        )


        results_list.append(
            {
                "excluded_tc": excluded_tc,
                "measured": measured,
                "predicted": predicted,
        
                # mean-based errors
                "error": error,
                "abs_error": abs_error,
                "rel_error": rel_error,
        
                # prediction interval
                "lower_95": lower_95,
                "upper_95": upper_95,
                "pi_width": pi_width,
        
                # interval-based errors
                "within_pi": within_pi,
                "pi_error": pi_error,
                "interval_score": iscore,

                "iteration": iter
        
            }
        )

    return pd.DataFrame(results_list)


def predict_surface(model_fitted: gpytorch.models.ExactGP,
                    likelihood: GaussianLikelihood,
                    norm_params: dict,
                    df: pd.DataFrame,
                    Z_fixed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict the validation metric on a surface with fixed Z.

    Parameters
    ----------
    model_fitted : gpytorch.models.ExactGP
        Trained GPR model
    likelihood : GaussianLikelihood
        Trained likelihood
    norm_params : dict
        Data normalisation parameters
    df : pd.DataFrame
        TC validation data
    Z_fixed : float
        z coordinate at which to calculate the predictions

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Prediction coordinated and prediction data
    """

    x_vals = df['x'].values
    y_vals = df['y'].values

    x_grid = np.linspace(min(x_vals), max(x_vals), 75)
    y_grid = np.linspace(min(y_vals), max(y_vals), 75)
    Xg, Yg = np.meshgrid(x_grid, y_grid)

    X_query = np.column_stack((
        Xg.ravel(),
        Yg.ravel(),
        np.full_like(Xg.ravel(), Z_fixed)
    ))

    pred_mean, pred_lower, pred_upper = predict_gpr(model_fitted, likelihood, norm_params, X_query)

    pred_mean = pred_mean.reshape(Xg.shape)
    pred_lower = pred_lower.reshape(Xg.shape)
    pred_upper = pred_upper.reshape(Xg.shape)

    return Xg, Yg, pred_mean, pred_lower, pred_upper


def plot_surface(Xg: np.ndarray, Yg: np.ndarray, mean: np.ndarray,
                 CI_lower: np.ndarray, CI_upper: np.ndarray, 
                 d_type: str, output_dir: Path) -> None:
    """ Plot prediction surface with CIs.

    Parameters
    ----------
    Xg : np.ndarray
        X coordinate grid
    Yg : np.ndarray
        Y coordinate grid
    mean : np.ndarray
        Mean values grid
    CI_lower : np.ndarray
        Lower CI grid
    CI_upper : np.ndarray
        Upper CI grid
    d_type : str
        Validation metric type
    output_dir : Path
        Output path
    """

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Mean surface
    surf = ax.plot_surface(Xg, Yg, mean, cmap="coolwarm", edgecolor='none')
    # PI surfaces
    interval_alpha = 0.3  # transparency for CI surfaces
    ax.plot_surface(Xg, Yg, CI_lower, color='grey', alpha=interval_alpha, edgecolor='none', label='Obs CI Lower')
    ax.plot_surface(Xg, Yg, CI_upper, color='grey', alpha=interval_alpha, edgecolor='none', label='Obs CI Upper')

    ax.set_xlabel(r'$x$ (m)', labelpad=10)
    ax.set_ylabel(r'$y$ (m)', labelpad=10)
    ax.set_zlabel(r'Predicted $d$ [$^{\circ C}$]', labelpad=15)

    ax.set_title(f'Quadratic fit surface for "{d_type}"', pad=0)

    fig.colorbar(surf, shrink=0.5, aspect=10, label=r'Predicted $d$ [$^{\circ C}$]')

    fig.savefig(output_dir / f"{d_type}_surface.png", dpi=300, bbox_inches="tight")

    plt.close(fig)


def main():

    EXP_DIR.mkdir(exist_ok=True)

    coeff_dir = EXP_DIR / "model_coefficients"
    train_dir = EXP_DIR / "training_predictions"
    ablation_dir = EXP_DIR / "ablation_results"
    surface_dir = EXP_DIR / "surfaces"

    for d in [coeff_dir, train_dir, ablation_dir, surface_dir]:
        d.mkdir(parents=True, exist_ok=True)

    merged_df, d_types = load_data(INPUT_FILE)

    summary_errors = []

    for d_type in d_types:

        print("=" * 80)
        print(f"Processing {d_type}")

        # ---------------------------------------------------------------------
        # Fit model to all TC data
        # ---------------------------------------------------------------------

        model_fitted, likelihood, norm_params, iter = fit_gpr_model(merged_df, d_type, EPOCHS)

        # ---------------------------------------------------------------------
        # Training predictions using the model fitted to all TC data
        # ---------------------------------------------------------------------

        train_pred_df = evaluate_training_points(
            model_fitted,
            likelihood,
            norm_params,
            merged_df,
            d_type
        )
        train_pred_df.to_csv(train_dir / f"{d_type}_training_predictions.csv", index=False)

        # ---------------------------------------------------------------------
        # Ablation study
        # ---------------------------------------------------------------------

        ablation_df = leave_one_out_ablation(merged_df, d_type, EPOCHS)
        ablation_df.to_csv(ablation_dir / f"{d_type}_ablation.csv", index=False)

        mae = mean_absolute_error(ablation_df["measured"], ablation_df["predicted"])
        rmse = np.sqrt(mean_squared_error(ablation_df["measured"], ablation_df["predicted"]))
        # summary_errors.append({
        #     "d_type": d_type,
        #     "MAE": mae,
        #     "RMSE": rmse,
        #     "mean_abs_error": ablation_df["abs_error"].mean(),
        #     "mean_rel_error": ablation_df["rel_error"].mean(skipna=True)
        # })


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
            "accuracy": accuracy,
            "accuracy_matrix": metrics["accuracy"],
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "TN": metrics["TN"],
            "FN": metrics["FN"],
            }
        )

        # ---------------------------------------------------------------------
        # Surface prediction using the model fitted to all TC data
        # ---------------------------------------------------------------------

        Z_fixed = (35 - 15/2 - 5) * 1e-3

        Xg, Yg, pred_mean, pred_lower, pred_upper = predict_surface(
            model_fitted,
            likelihood,
            norm_params,
            merged_df,
            Z_fixed
        )
        
        plot_surface(Xg, Yg, 
                     pred_mean, pred_lower, pred_upper, 
                     d_type, surface_dir)

    # -------------------------------------------------------------------------
    # Save ablation study results
    # -------------------------------------------------------------------------

    summary_df = pd.DataFrame(summary_errors)

    summary_df.to_csv(EXP_DIR / "ablation_summary.csv", index=False)

    print(summary_df)


def test_model(model_type, device):

    lr = model_type["lr"]
    kernel_type = model_type["kernel_type"]
    ard_num_dims = model_type["ard_num_dims"]
    nu = model_type["nu"]
    model_type_print = f"{lr}_{kernel_type}_{ard_num_dims}_{nu}"

    EXP_DIR.mkdir(exist_ok=True)

    coeff_dir = EXP_DIR / "model_coefficients"
    train_dir = EXP_DIR / "training_predictions"
    ablation_dir = EXP_DIR / "ablation_results"

    for d in [coeff_dir, train_dir, ablation_dir]:
        d.mkdir(parents=True, exist_ok=True)

    merged_df, d_types = load_data(INPUT_FILE)

    summary_errors = []

    for d_type in d_types:

        # if d_type != "total_plus" and d_type != "total_minus":
        #     continue

        print("=" * 80)
        print(f"Processing {d_type}")

        # ---------------------------------------------------------------------
        # Fit model to all TC data
        # ---------------------------------------------------------------------

        model_fitted, likelihood, norm_params, iter = fit_gpr_model(merged_df, d_type, kernel_type,
                                                              ard_num_dims, nu, EPOCHS, lr, device=device)

        checkpoint = {
            "model_state_dict": model_fitted.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "norm_params": norm_params,
            "d_type": d_type,
        }
        
        torch.save(checkpoint, coeff_dir / f"{d_type}_gpr_model_{model_type_print}.pth")

        # ---------------------------------------------------------------------
        # Training predictions using the model fitted to all TC data
        # ---------------------------------------------------------------------

        train_pred_df = evaluate_training_points(
            model_fitted,
            likelihood,
            norm_params,
            merged_df,
            d_type
        )
        train_pred_df.to_csv(train_dir / f"{d_type}_training_predictions_{model_type_print}.csv", index=False)

        # ---------------------------------------------------------------------
        # Ablation study
        # ---------------------------------------------------------------------

        ablation_df = leave_one_out_ablation(merged_df, d_type, kernel_type,
                                             ard_num_dims, nu, EPOCHS, lr, device=device)
        ablation_df.to_csv(ablation_dir / f"{d_type}_ablation_{model_type_print}.csv", index=False)

        mae = mean_absolute_error(ablation_df["measured"], ablation_df["predicted"])
        rmse = np.sqrt(mean_squared_error(ablation_df["measured"], ablation_df["predicted"]))
        # summary_errors.append({
        #     "d_type": d_type,
        #     "MAE": mae,
        #     "RMSE": rmse,
        #     "mean_abs_error": ablation_df["abs_error"].mean(),
        #     "mean_rel_error": ablation_df["rel_error"].mean(skipna=True)
        # })


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
            "accuracy": accuracy,
            "accuracy_matrix": metrics["accuracy"],
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "TN": metrics["TN"],
            "FN": metrics["FN"],
            }
        )

    # -------------------------------------------------------------------------
    # Save ablation study results
    # -------------------------------------------------------------------------

    summary_df = pd.DataFrame(summary_errors)

    summary_df.to_csv(EXP_DIR / f"ablation_summary_{model_type_print}.csv", index=False)

    print(summary_df)

# if __name__ == "__main__":
#     main()


lrs = np.array([0.003, 0.01, 0.03, 0.1])
kernel_types = ["RBF", "Matern", "Linear"]
ard_num_dimss = np.array([None, 3])
nus = np.array([2.5, 1.5])

# model_type = {
#   "lr": 0.01,
#   "kernel_type": "RBF",
#   "ard_num_dims": 1,
#   "nu": None
# }

# test_model(model_type, device="cpu")



for lr in lrs:
    for kernel_type in kernel_types:
        for ard_num_dims in ard_num_dimss:
            if kernel_type == "Matern":
                for nu in nus:
                    model_type = {
                      "lr": lr,
                      "kernel_type": kernel_type,
                      "ard_num_dims": ard_num_dims,
                      "nu": nu
                    }
                    test_model(model_type, device="cuda")
            else:
                model_type = {
                      "lr": lr,
                      "kernel_type": kernel_type,
                      "ard_num_dims": ard_num_dims,
                      "nu": None
                }
                test_model(model_type, device="cuda")










# -------------------------------------------------------------------------
# Save ablation study results
# -------------------------------------------------------------------------

""" checkpoint = torch.load(
    "gpr_model.pth",
    map_location="cpu"
)

norm_params = checkpoint["norm_params"]

X_train = df[['x', 'y', 'z']].values
y_train = df[checkpoint["d_type"]].values.reshape(-1, 1)

X_norm = (
    X_train - norm_params["X_mean"]
) / norm_params["X_std"]

y_norm = (
    y_train - norm_params["y_mean"]
) / norm_params["y_std"]

train_x = torch.tensor(
    X_norm,
    dtype=torch.float32
)

train_y = torch.tensor(
    y_norm.flatten(),
    dtype=torch.float32
)

likelihood = gpytorch.likelihoods.GaussianLikelihood()

model = ExactGPModel(
    train_x,
    train_y,
    likelihood
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)
likelihood.load_state_dict(
    checkpoint["likelihood_state_dict"]
)

model.eval()
likelihood.eval()

X_new = np.array([
    [10.0, 20.0, 5.0],
    [15.0, 25.0, 7.0],
    [20.0, 30.0, 8.0],
])

X_new_norm = (
    X_new - norm_params["X_mean"]
) / norm_params["X_std"]

test_x = torch.tensor(
    X_new_norm,
    dtype=torch.float32
)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    prediction = likelihood(model(test_x))

pred_mean_norm = prediction.mean.numpy()
pred_std_norm = prediction.stddev.numpy()

pred_mean = (
    pred_mean_norm * norm_params["y_std"]
    + norm_params["y_mean"]
)

pred_std = (
    pred_std_norm * norm_params["y_std"]
)

print(pred_mean)
print(pred_std) """
