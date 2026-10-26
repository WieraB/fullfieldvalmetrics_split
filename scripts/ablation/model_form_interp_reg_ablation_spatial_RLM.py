import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
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
INPUT_FILE_TEMP = (
    Path(__file__).resolve().parents[1] / "mavm_nuisance_split" / "nen17_net250_na400" / "ablation" / "inputs"
    / "mean_sim_temperature.csv"
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
        case 9:
            X = pd.DataFrame({
                "x": x,
                "y": y,
                "z": z,
                "x2": x**2
            })
        case 10:
            X = pd.DataFrame({
                "x": x,
                "y": y,
                "z": z,
                "y2": y**2
            })
        case 11:
            X = pd.DataFrame({
                "x": x,
                "y": y,
                "z": z,
                "z2": z**2
            })
        case 12:
            X = pd.DataFrame({
                "x": x,
                "y": y,
                "z": z,
            })
        case 13:
            X = pd.DataFrame({
                "x": x,
                "y": y,
                "z": z,
                "xy": x*y,
            })
        case 14:
            X = pd.DataFrame({
                "x": x,
                "y": y,
                "z": z,
                "yz": y*z,
            })
        case 15:
            X = pd.DataFrame({
                "x": x,
                "y": y,
                "z": z,
                "xz": x*z,
            })
        case 16:
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

    model = sm.RLM(d_vals, X)
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

    pred = model_fitted.predict(X).to_numpy()

    # Covariance matrix of the RLM coefficients
    cov = model_fitted.cov_params()
    
    # Standard error of each prediction
    X_array = np.asarray(X)
    pred_se = np.sqrt(
        np.sum((X_array @ cov) * X_array, axis=1)
    )
    
    z = stats.norm.ppf(0.975) # 95% CI
    lower_95 = pred - z * pred_se
    upper_95 = pred + z * pred_se

    lower_95 = lower_95.iloc[0]
    upper_95 = upper_95.iloc[0]

    out_df = pd.DataFrame({
        "TC": df.index,
        "measured": df[d_type].values,
        "predicted": pred,
        "lower_95": lower_95,
        "upper_95": upper_95
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

        pred = model_fitted.predict(X_test).to_numpy()
        pred = np.float64(pred.item())

        # Covariance matrix of the RLM coefficients
        cov = model_fitted.cov_params()
        
        # Standard error of each prediction
        X_test_array = np.asarray(X_test)

        # pred_se = np.sqrt(
        #     np.sum((X_test_array @ cov) * X_test_array, axis=1)
        # )
        
        # z = stats.norm.ppf(0.975) # 95% CI
        # lower_95 = pred - z * pred_se
        # upper_95 = pred + z * pred_se

        # lower_95 = lower_95.iloc[0]
        # upper_95 = upper_95.iloc[0]



        
        # Variance due to coefficient estimation
        pred_var = np.sum((X_test_array @ cov) * X_test_array, axis=1)
        
        # Residual variance
        resid = np.asarray(model_fitted.resid)
        resid_var = np.sum(resid**2) / model_fitted.df_resid
        
        # Total prediction variance
        pred_se = np.sqrt(pred_var + resid_var)
        
        # 95% prediction interval
        z = stats.norm.ppf(0.975)
        
        lower_95 = pred - z * pred_se
        upper_95 = pred + z * pred_se
        
        lower_95 = lower_95[0]
        upper_95 = upper_95[0]







        measured = test_df[d_type].values[0]
        predicted = pred

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

        ss_res = np.sum((measured - predicted) ** 2)
        ss_tot = np.sum((measured - np.mean(measured)) ** 2)
        
        r_squared = 1 - ss_res / ss_tot

        residuals = measured - predicted
        
        # n = len(measured)
        n=1

        k = X_test.shape[1]
        
        rss = np.sum(residuals ** 2)
        
        aic = n * np.log(rss / n) + 2 * k


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
                "r_squared": r_squared,
                "aic": aic,
            }
        )

    results_df = pd.DataFrame(results_list)

    return results_df

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
            terms = create_term_names(model_type)

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

    print(merged_df)
    print(d_types)

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


# test_model(model_type=1)

for i in range(9, 17):
    test_model(model_type=i)