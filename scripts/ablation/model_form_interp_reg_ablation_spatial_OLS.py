import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ablation_funcs import mape_func, interval_score, regression_accuracy_metrics, load_data

# Ablation study for nine models

# -------------------------
# Set random seed
# -------------------------

SEED = 42
np.random.seed(SEED)

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
EXP_DIR = Path(__file__).resolve().parents[1] / "mavm_nuisance_split" / "nen17_net250_na400" / "ablation" / "outputs" / "interp_reg_spatial"


# INPUT_FILE = (
#     Path.cwd()
#     / "images_pointsensors_pulse25X_v4"
#     / "pointsensors_mavm.csv"
# )
# EXP_DIR = Path.cwd() / "quadratic_interp"


TOLERANCE=0.1

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def build_design_matrix(x: np.ndarray, y: np.ndarray, z: np.ndarray, model_type: int) -> pd.DataFrame:
    """Design matrix containing the predictor variables

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinate
    y : np.ndarray
        Spatial coordinate
    z : np.ndarray
        Spatial coordinate
    model_type : int
        Model type 

    Returns
    -------
    pd.DataFrame
        Dataframe with predictor variables
    """

    print(f"Model type selected: {model_type}.")
    match model_type:
        case 1:
            X = pd.DataFrame({
                "x": x,
                "y": y,
                "z": z,
                "x2": x**2
            })
        case 2:
            X = pd.DataFrame({
                "x": x,
                "y": y,
                "z": z,
                "y2": y**2
            })
        case 3:
            X = pd.DataFrame({
                "x": x,
                "y": y,
                "z": z,
                "z2": z**2
            })
        case 4:
            X = pd.DataFrame({
                "x": x,
                "y": y,
                "z": z,
            })
        case 5:
            X = pd.DataFrame({
                "x": x,
                "y": y,
                "z": z,
                "xy": x*y,
            })
        case 6:
            X = pd.DataFrame({
                "x": x,
                "y": y,
                "z": z,
                "yz": y*z,
            })
        case 7:
            X = pd.DataFrame({
                "x": x,
                "y": y,
                "z": z,
                "xz": x*z,
            })
        case 8:
            X = pd.DataFrame({
                "x": x,
                "y": y,
                "z": z,
                "xyz": x*y*z,
            })
        case _:
            raise ValueError(f"Unknown model type: {model_type}.")


    # X = sm.add_constant(X)

    X = sm.add_constant(X, has_constant='add')

    return X

def create_term_names(model_type: int) -> list:

    print(f"Model type selected: {model_type}.")
    match model_type:
        case 1:
            terms = ["const", "x", "y", "z", "x2"]
        case 2:
            terms = ["const", "x", "y", "z", "y2"]
        case 3:
            terms = ["const", "x", "y", "z", "z2"]
        case 4:
            terms = ["const", "x", "y", "z"]
        case _:
            raise ValueError(f"Unknown model type: {model_type}.")



    return terms

def fit_model(df: pd.DataFrame, d_type: str, model_type: int) -> RegressionResultsWrapper:
    """ Fit quadratic model

    Parameters
    ----------
    df : pd.DataFrame
        Data to fit the model (training data)
    d_type : str
        Validation metric type
    model_type : int
        Model type 

    Returns
    -------
    RegressionResultsWrapper
        Fitted model
    """

    x_vals = df['x'].values
    y_vals = df['y'].values
    z_vals = df['z'].values

    d_vals = df[d_type].values

    X = build_design_matrix(x_vals, y_vals, z_vals, model_type)

    model = sm.OLS(d_vals, X)
    model_fitted = model.fit()

    return model_fitted

def evaluate_training_points(model_fitted: RegressionResultsWrapper, 
                             df: pd.DataFrame, 
                             d_type: str,
                             model_type: int) -> pd.DataFrame:
    """Evaluate the model at the data points used for fitting/training

    Parameters
    ----------
    model_fitted : RegressionResultsWrapper
        Model fitted to the training data
    df : pd.DataFrame
        Data used to fit the model (training data)
    d_type : str
        Validation metric type
    model_type : int
        Model type 

    Returns
    -------
    pd.DataFrame
        Prediction results
    """

    X = build_design_matrix(
        df['x'].values,
        df['y'].values,
        df['z'].values,
        model_type
    )

    pred = model_fitted.get_prediction(X)
    pred_summary = pred.summary_frame(alpha=0.05)

    out_df = pd.DataFrame({
        "TC": df.index,
        "measured": df[d_type].values,
        "predicted": pred_summary["mean"].values,
        "lower_95": pred_summary["obs_ci_lower"].values,
        "upper_95": pred_summary["obs_ci_upper"].values
    })

    return out_df

def leave_one_out_ablation(df: pd.DataFrame, d_type: str, model_type: int) -> pd.DataFrame:
    """Perform ablation study by excluding one TC data 
    and fitting the model to the rest of the TC data

    Parameters
    ----------
    df : pd.DataFrame
        TC validation data
    d_type : str
        Validation metric type
    model_type : int
        Model type 

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

        # Fit model
        model_fitted = fit_model(train_df, d_type, model_type)

        # Predict excluded point
        X_test = build_design_matrix(
            test_df['x'].values,
            test_df['y'].values,
            test_df['z'].values,
            model_type
        )

        pred = model_fitted.get_prediction(X_test)
        pred_summary = pred.summary_frame(alpha=0.05)

        measured = test_df[d_type].values[0]
        predicted = pred_summary["mean"].values[0]
        lower_95 = pred_summary['obs_ci_lower'].values[0]
        upper_95 = pred_summary['obs_ci_upper'].values[0]

        error = predicted - measured
        abs_error = abs(error)
        within_pi = lower_95 <= measured <= upper_95
        if measured < lower_95:
            pi_error = lower_95 - measured
        elif measured > upper_95:
            pi_error = measured - upper_95
        else:
            pi_error = 0.0

        if np.abs(measured) > 1e-12:
            rel_error = abs(predicted - measured) / abs(measured)
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
        
                # model fit
                "r_squared": model_fitted.rsquared,
                "aic": model_fitted.aic,
            }
        )

    results_df = pd.DataFrame(results_list)

    return results_df


def predict_surface(model_fitted: RegressionResultsWrapper, 
                    df: pd.DataFrame, 
                    Z_fixed: float,
                    model_type: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict the validation metric on a surface with fixed Z.

    Parameters
    ----------
    model_fitted : RegressionResultsWrapper
        Model fitted to the training data
    df : pd.DataFrame
        TC validation data
    Z_fixed : float
        z coordinate at which to calculate the predictions
    model_type : int
        Model type 

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

    X_query = build_design_matrix(
        Xg.ravel(),
        Yg.ravel(),
        np.full_like(Xg.ravel(), Z_fixed),
        model_type
    )

    pred_summary = (model_fitted.get_prediction(X_query).summary_frame(alpha=0.05))

    pred_mean = pred_summary["mean"].values.reshape(Xg.shape)
    pred_lower = pred_summary['obs_ci_lower'].values.reshape(Xg.shape)
    pred_upper = pred_summary['obs_ci_upper'].values.reshape(Xg.shape)

    return Xg, Yg, pred_mean, pred_lower, pred_upper


def plot_surface(Xg: np.ndarray, Yg: np.ndarray, mean: np.ndarray,
                 PI_lower: np.ndarray, PI_upper: np.ndarray, 
                 d_type: str, output_dir: Path) -> None:
    """ Plot prediction surface with PIs.

    Parameters
    ----------
    Xg : np.ndarray
        X coordinate grid
    Yg : np.ndarray
        Y coordinate grid
    mean : np.ndarray
        Mean values grid
    PI_lower : np.ndarray
        Lower PI grid
    PI_upper : np.ndarray
        Upper PI grid
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
    interval_alpha = 0.3  # transparency for PI surfaces
    ax.plot_surface(Xg, Yg, PI_lower, color='grey', alpha=interval_alpha, edgecolor='none', label='Obs CI Lower')
    ax.plot_surface(Xg, Yg, PI_upper, color='grey', alpha=interval_alpha, edgecolor='none', label='Obs CI Upper')

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

    model_type = 1

    for d_type in d_types:

        print("=" * 80)
        print(f"Processing {d_type}")

        # ---------------------------------------------------------------------
        # Fit model to all TC data
        # ---------------------------------------------------------------------

        model_fitted = fit_model(merged_df, d_type, model_type)

        if hasattr(model_fitted.params, "index"):
            terms = model_fitted.params.index
        else:
            terms = ["const", "x", "y", "z", "x2"]

        coeff_df = pd.DataFrame({
                "term": terms,
                "coefficient": model_fitted.params.values,})

        coeff_df.to_csv(coeff_dir / f"{d_type}_coefficients.csv", index=False)

        # ---------------------------------------------------------------------
        # Training predictions using the model fitted to all TC data
        # ---------------------------------------------------------------------

        train_pred_df = evaluate_training_points(model_fitted, merged_df, d_type, model_type)
        train_pred_df.to_csv(train_dir / f"{d_type}_training_predictions.csv", index=False)

        # ---------------------------------------------------------------------
        # Ablation study
        # ---------------------------------------------------------------------

        ablation_df = leave_one_out_ablation(merged_df, d_type, model_type)
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
        Xg, Yg, pred_mean, pred_lower, pred_upper = predict_surface(model_fitted, merged_df, Z_fixed, model_type)

        plot_surface(Xg, Yg, 
                     pred_mean, pred_lower, pred_upper, 
                     d_type, surface_dir)

    # -------------------------------------------------------------------------
    # Save ablation study results
    # -------------------------------------------------------------------------

    summary_df = pd.DataFrame(summary_errors)

    summary_df.to_csv(EXP_DIR / "ablation_summary.csv", index=False)

    print(summary_df)


def test_model(model_type):

    EXP_DIR.mkdir(exist_ok=True)

    coeff_dir = EXP_DIR / "model_coefficients"
    train_dir = EXP_DIR / "training_predictions"
    ablation_dir = EXP_DIR / "ablation_results"

    for d in [coeff_dir, train_dir, ablation_dir]:
        d.mkdir(parents=True, exist_ok=True)

    merged_df, d_types = load_data(INPUT_FILE)

    summary_errors = []

    for d_type in d_types:

        print("=" * 80)
        print(f"Processing {d_type}")

        # ---------------------------------------------------------------------
        # Fit model to all TC data
        # ---------------------------------------------------------------------

        model_fitted = fit_model(merged_df, d_type, model_type)

        if hasattr(model_fitted.params, "index"):
            terms = model_fitted.params.index
        else:
            terms = create_term_names(model_type)

        coeff_df = pd.DataFrame({
                "term": terms,
                "coefficient": model_fitted.params.values,})

        coeff_df.to_csv(coeff_dir / f"{d_type}_coefficients_{model_type}.csv", index=False)

        # ---------------------------------------------------------------------
        # Training predictions using the model fitted to all TC data
        # ---------------------------------------------------------------------

        train_pred_df = evaluate_training_points(model_fitted, merged_df, d_type, model_type)
        train_pred_df.to_csv(train_dir / f"{d_type}_training_predictions_{model_type}.csv", index=False)

        # ---------------------------------------------------------------------
        # Ablation study
        # ---------------------------------------------------------------------

        ablation_df = leave_one_out_ablation(merged_df, d_type, model_type)
        ablation_df.to_csv(ablation_dir / f"{d_type}_ablation_{model_type}.csv", index=False)

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

    summary_df.to_csv(EXP_DIR / f"ablation_summary_{model_type}.csv", index=False)

    print(summary_df)

# if __name__ == "__main__":
#     main()


# test_model(model_type=5)

for i in range(1, 9):
    test_model(model_type=i)