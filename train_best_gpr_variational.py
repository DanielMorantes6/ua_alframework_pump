"""
train_best_gpr_variational.py

This script is provided as a lightweight verification entry point for the best-performing active learning path reported in the study
(GPR + uncertainty-driven sampling).

Its purpose is to:
- verify that the repository environment is correctly configured
- reproduce the final GPR training logic outside the notebook interface
- expose the final fixed training function used for the best path

This script is not intended to replace the original notebook workflow.
The corresponding notebook remains the main record of the full methodology,
and the final artifacts are already generated and stored separately.
Some data file locations must be updated locally before execution.
"""

import joblib
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, WhiteKernel, Matern
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    r2_score,
)

import Functions_PumpAI as fun


def compute_kge(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    r = np.corrcoef(y_true, y_pred)[0, 1]
    sigma_true = np.std(y_true, ddof=0)
    sigma_pred = np.std(y_pred, ddof=0)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    return 1 - np.sqrt(
        (r - 1) ** 2
        + (sigma_pred / sigma_true - 1) ** 2
        + (mean_pred / mean_true - 1) ** 2
    )


def compute_pbias(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    return 100 * np.sum(y_true - y_pred) / np.sum(y_true)


def gpr_fixed_with_predictions(
    kernel,
    X_train,
    y_train,
    X_test,
    y_test,
    best_params,
    random_state=42,
    n_restarts_optimizer=5,
):
    y_train_1d = np.ravel(y_train)
    y_test_1d = np.ravel(y_test)

    best_model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )

    best_model.set_params(**best_params)
    best_model.fit(X_train, y_train_1d)

    print("\nKernel parameters available:")
    print(best_model.kernel_.get_params().keys())

    y_pred_train, ysigma_pred_train = best_model.predict(X_train, return_std=True)
    y_pred_test, ysigma_pred_test = best_model.predict(X_test, return_std=True)

    results = {
        "mse_train": mean_squared_error(y_train_1d, y_pred_train),
        "r2_train": r2_score(y_train_1d, y_pred_train),
        "mae_train": mean_absolute_error(y_train_1d, y_pred_train),
        "ev_train": explained_variance_score(y_train_1d, y_pred_train),
        "kge_train": compute_kge(y_train_1d, y_pred_train),
        "pbias_train": compute_pbias(y_train_1d, y_pred_train),

        "mse_test": mean_squared_error(y_test_1d, y_pred_test),
        "r2_test": r2_score(y_test_1d, y_pred_test),
        "mae_test": mean_absolute_error(y_test_1d, y_pred_test),
        "ev_test": explained_variance_score(y_test_1d, y_pred_test),
        "kge_test": compute_kge(y_test_1d, y_pred_test),
        "pbias_test": compute_pbias(y_test_1d, y_pred_test),

        "best_params": best_params,
    }

    results_uncertainty = {
        "scaled_uncertainty_train": ysigma_pred_train,
        "scaled_uncertainty_test": ysigma_pred_test,
    }

    predictions = {
        "y_train_true": y_train_1d,
        "y_train_pred": y_pred_train,
        "y_test_true": y_test_1d,
        "y_test_pred": y_pred_test,
    }

    return best_model, results, results_uncertainty, predictions


def main():
    print("Loading fixed BEP-scaled test set...")
    testdata_bepscaled = joblib.load("artifacts/bepscaled_testdf.pkl")

    kernelMatern1 = (
        C(1.0, (1e-2, 1e2))
        * Matern([1.0, 1.0, 1.0, 1.0], (1e-1, 1e5), nu=1.5)
        + WhiteKernel(1e-2, (1e-10, 1e1))
    )

    # NOTE:
    # Update this path locally before execution if needed.
    file_path = "/Users/danielmorantes/PythonP/ua_alframework_pump/gpr_variational_data.xlsx"
    sheet_name = "gp_variational"

    feature_cols = ["rho(kg/m3)", "k(Pa*s^n)", "n(-)", "massflow_dim"]
    target_cols = "head_dim"

    print("Loading final training data from Excel...")
    var_rawrev = fun.load_excel_data(file_path, sheet_name)

    print("Applying BEP scaling...")
    entirevarrev, var_BEPscaledrev = fun.apply_bep_scaling(var_rawrev)
    print("samples after the BEP normalization:", len(var_BEPscaledrev), "of", len(entirevarrev))

    print("Scaling train and test data...")
    Xtrain_scaledrev, Xtest_scaledrev, ytrain_scaledrev, ytest_scaledrev, scalerxrev, scaleryrev = fun.scale_data(
        var_BEPscaledrev,
        testdata_bepscaled,
        feature_cols,
        target_cols,
    )

    best_params_gpr_final = {
        "kernel__k2__noise_level": 0.005050000000000001,
        "kernel__k1__k1__constant_value": 2.277777777777778,
        "alpha": 0.21,
    }

    print("Training final verification GPR model...")
    gp_modelrev, gp_scorerev, gp_uncrev, gp_predictionsrev = gpr_fixed_with_predictions(
        kernel=kernelMatern1,
        X_train=Xtrain_scaledrev,
        y_train=ytrain_scaledrev,
        X_test=Xtest_scaledrev,
        y_test=ytest_scaledrev,
        best_params=best_params_gpr_final,
        random_state=42,
        n_restarts_optimizer=5,
    )

    print("\nFinal GPR verification results:")
    for key, val in gp_scorerev.items():
        if key != "best_params":
            print(f"{key}: {val:.4f}")

    print("Best hyperparameters:", gp_scorerev["best_params"])
    print("\nVerification script completed successfully.")


if __name__ == "__main__":
    main()