import numpy as np
import pandas as pd
from pathlib import Path


def mape_func(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def interval_score(y, lower, upper, alpha=0.05):
    """
    Strictly proper interval score.

    Smaller is better.
    """

    width = upper - lower

    below = np.maximum(lower - y, 0)
    above = np.maximum(y - upper, 0)

    score = (
        width
        + (2 / alpha) * below
        + (2 / alpha) * above
    )

    return score

def regression_accuracy_metrics(
    measured,
    predicted,
    tolerance=0.02,
):
    """
    Classification of regression errors using a relative error tolerance.

    rel_error = (predicted - measured) / measured

    Returns counts and common summary statistics.
    """

    measured = np.asarray(measured)
    predicted = np.asarray(predicted)

    rel_error = ((predicted - measured) / np.maximum(measured, 1e-12))

    FP = np.sum(rel_error > tolerance)
    FN = np.sum(rel_error < -tolerance)

    TP = np.sum((rel_error <= tolerance) & (rel_error >= 0.0))
    TN = np.sum((rel_error > -tolerance) & (rel_error < 0.0))

    accuracy = (TP + TN) / len(rel_error)

    precision = TP / (TP + FP) if TP + FP else np.nan

    recall = TP / (TP + FN) if TP + FN else np.nan

    specificity = TN / (TN + FP) if TN + FP else np.nan

    f1 = (
        2 * precision * recall / (precision + recall)
        if (
            not np.isnan(precision)
            and not np.isnan(recall)
            and (precision + recall) > 0
        )
        else np.nan
    )

    return {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "rel_error": rel_error,
    }



def load_data(input_file: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load data and merge it into one dataframe

    Parameters
    ----------
    input_file : Path
        Input file

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Merged dataframe and thermocouple name list
    """

    d_values = pd.read_csv(input_file, index_col=0)

    coords_numpy = np.array([
        [0.0116, -0.0245, 0.0194],
        [0.0138, -0.0245, 0.0013],
        [0.0067, -0.0245, 0.012],
        [0.0110,  0.0245, 0.0031],
        [-0.0105, 0.0245, -0.005],
        [-0.0058, 0.0245, 0.0171],
        [-0.018, -0.0006, 0.0164],
        [-0.018, -0.004, -0.0085],
        [-0.018, -0.0047, 0.0073],
        [-0.018,  0.0124, -0.0032]
    ])

    coords = pd.DataFrame(
        coords_numpy,
        columns=["x", "y", "z"],
        index=[
            "TC1", "TC2", "TC3", "TC4", "TC5",
            "TC6", "TC7", "TC8", "TC9", "TC10"
        ]
    )

    merged_df = coords.join(d_values.T, how='inner')

    return merged_df, list(d_values.index)


def load_data_temp(input_file: Path, input_file_temp: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load data and merge it into one dataframe

    Parameters
    ----------
    input_file : Path
        Input file
    input_file_path : Path
        Input file with mean temperatures

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Merged dataframe and thermocouple name list
    """

    d_values = pd.read_csv(input_file, index_col=0)
    T_values = pd.read_csv(input_file_temp, index_col=0)

    merged_df = T_values.T.join(d_values.T, how='inner')

    merged_df = merged_df.rename(columns={"sim_nom_mean": "T"})

    return merged_df, list(d_values.index)



# def load_data_with_file(input_file: Path, coords_file: Path) -> tuple[pd.DataFrame, list[str]]:
#     """Load data and merge it into one dataframe.

#     Parameters
#     ----------
#     input_file : Path
#         Input data file.
#     coords_file : Path
#         CSV file containing thermocouple coordinates with columns
#         'x', 'y', and 'z'.

#     Returns
#     -------
#     tuple[pd.DataFrame, list[str]]
#         Merged dataframe and thermocouple name list.
#     """

#     d_values = pd.read_csv(input_file, index_col=0)
#     coords = pd.read_csv(coords_file)
#     coords.index = [
#         "TC1", "TC2", "TC3", "TC4", "TC5",
#         "TC6", "TC7"
#     ]
#     print(coords)
#     print(coords.shape)
#     #coords = coords[["x", "y", "z"]].to_numpy()

#     merged_df = coords.join(d_values.T, how="inner")

#     return merged_df, list(d_values.index)



def load_data_with_file(input_file: Path, coords_file: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load data and merge it into one dataframe.

    Parameters
    ----------
    input_file : Path
        Input data file.
    coords_file : Path
        CSV file containing thermocouple coordinates with columns
        'x', 'y', and 'z'.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Merged dataframe and thermocouple name list.
    """

    d_values = pd.read_csv(input_file, index_col=0)
    coords = pd.read_csv(coords_file)
    # coords.index = [
    #     "TC1", "TC2", "TC3", "TC4", "TC5",
    #     "TC6", "TC7"
    # ]
    coords.index = [f"TC{i}" for i in range(1, coords.shape[0]+1)]
    print(coords)
    print(coords.shape)
    #coords = coords[["x", "y", "z"]].to_numpy()

    merged_df = coords.join(d_values.T, how="inner")

    return merged_df, list(d_values.index)


# headers = [f"TC{i}" for i in range(1, pos.shape[0]+1)]