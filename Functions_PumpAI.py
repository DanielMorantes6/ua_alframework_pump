"""
Functions_PumpAI.py

Core utility functions used in the study:
"Active Learning Framework for Surrogate Modelling of Centrifugal Pumps Performance".

This module centralizes the reusable functions employed across the repository and is intended
to be imported by the Jupyter notebooks included in the project.

Included sections
-----------------
A. Output and persistence utilities
B. Synthetic fluid generation and rheological fitting
C. Data loading, descriptive analysis, and preprocessing
D. Baseline surrogate model training and feature analysis
E. General plotting utilities for model evaluation
F. Active learning sampling and dataset augmentation
G. Multi-training-set preparation and repeated training
H. Query-based / variational GPR candidate generation and optimization
I. Active learning performance and uncertainty visualization
J. Rheology and fluid-behavior visualization
K. Final result visualization blocks

Main capabilities
-----------------
- synthetic fluid generation and rheological fitting
- data loading, preprocessing, and BEP-based scaling
- train/validation/test splitting and feature scaling
- surrogate model training for XGBoost and Gaussian Process Regression
- active learning sampling and fluid selection strategies
- Bayesian optimization and uncertainty-driven candidate refinement
- plotting and visualization of model performance, uncertainty, and rheological behavior

Notes
-----
- Some functions generate figures or serialized objects and should use repository-relative output paths.
- Several workflows depend on the private dataset used in the study, which is not distributed in the public repository.
- This file contains the core implementation used throughout the notebooks and experimental workflow.
"""

#DEPENDENCIES
#Python and file handling
import os
import random as rd
import joblib
#Data handling and manipulation
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, dual_annealing
from scipy.spatial.distance import cdist
#Plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
#DOE/Sampling/Optimization
from pyDOE2 import lhs
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
#Scikit-learn: preprocessing, splitting, metrics, decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    pairwise_distances,
    pairwise_distances_argmin_min,
    silhouette_score
)
from sklearn.decomposition import PCA
#Scikit-learn: models / clustering
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
#XGBoost
import xgboost
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping
print("XGBoost version:", xgboost.__version__)
rd.seed(42) 
#Notebook-only / display helper
from IPython.display import display


#A. Output and persistence utilities
def save_plot(image_name, dpi=300, tight=True, show_path=True):
    folder_path = "/artifacts"
    if folder_path:
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, image_name)
    else:
        file_path = image_name

    if tight:
        plt.tight_layout()
    plt.savefig(file_path, dpi=dpi)

    if show_path:
        print(f"Plot saved to: {file_path}")

def save_object(obj, name):
    folder = 'artifacts'
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{name}.pkl")
    joblib.dump(obj, file_path)
    print(f"{name} saved in {folder}")

#B. Synthetic fluid generation and rheological fitting
def generate_sinthetic_fluids(operative_ranges, num_samples, filename):
    """
    Generate sinthetic fluid properties based on given operative ranges.
    Parameters:
    - operative_ranges (dict): Dictionary containing the min and max values for each parameter.
    - num_samples (int): Number of synthetic fluid samples to generate.
    - filename (str): Name of the file to save the generated data.
    Returns:
    - pd.DataFrame: DataFrame containing the synthetic fluid properties.
    """
    samples_dict = {
        key: [rd.uniform(value[0], value[1]) for _ in range(num_samples)]
        for key, value in operative_ranges.items()
    }
    # Convert dictionary to a Pandas DataFrame for better visualization
    df_samples = pd.DataFrame(samples_dict)
    df_samples.to_excel(f"{filename}.xlsx", index=False)
    print(f"Data saved as {filename}.xlsx")
    return df_samples

def fit_all_cross_fluids_to_powerlaw(file_path: str, sheet_name: str, header_row: int = 0, plot: bool = True):
    """
    Loads a sheet of Cross Fluids, fits each to the Power Law model,
    calculates R², adds fitted K and n values, and optionally plots
    each fit in a grid of subplots with log-log scaling and formatted ticks.
    """
    def cross_model(gamma_dot, mu0, muinf, gamma_c, m):
        return muinf + (mu0 - muinf) / (1 + (gamma_dot / gamma_c) ** m)

    def fit_power_law(gamma_dot, mu):
        log_gamma = np.log(gamma_dot)
        log_mu = np.log(mu)
        coeffs = np.polyfit(log_gamma, log_mu, 1)
        n = coeffs[0] + 1
        K = np.exp(coeffs[1])
        return K, n

    def fit_cross_to_powerlaw(mu0, muinf, gamma_c, m):
        gamma_dot = np.logspace(1, 2.5, 100)
        mu = cross_model(gamma_dot, mu0, muinf, gamma_c, m)
        K, n = fit_power_law(gamma_dot, mu)
        return K, n, gamma_dot, mu

    xls = pd.ExcelFile(file_path)
    df = xls.parse(sheet_name, header=header_row)
    df.columns = df.columns.str.strip()

    df["Fitted_K"] = np.nan
    df["Fitted_n"] = np.nan
    df["R2"] = np.nan
    fit_summary = []
    fitted_curves = []

    for i, row in df.iterrows():
        try:
            fluid_id = row["Sinthetic Fluid"]
            muinf = float(row["muinf (Pa*s)"])
            mu0 = float(row["mu0 (Pa*s)"])
            gamma_c = float(row["gammac (1/s)"])
            m = float(row["m (-)"])

            K, n, gamma_dot, mu_cross = fit_cross_to_powerlaw(mu0, muinf, gamma_c, m)
            mu_powerlaw = K * gamma_dot ** (n - 1)
            r2 = r2_score(mu_cross, mu_powerlaw)

            df.at[i, "Fitted_K"] = K
            df.at[i, "Fitted_n"] = n
            df.at[i, "R2"] = r2
            fit_summary.append({
                "Fluid_ID": fluid_id,
                "Fitted_K": K,
                "Fitted_n": n,
                "R2": r2
            })

            if plot:
                fitted_curves.append({
                    "fluid_id": fluid_id,
                    "gamma_dot": gamma_dot,
                    "mu_cross": mu_cross,
                    "mu_powerlaw": mu_powerlaw,
                    "r2": r2
                })

        except Exception as e:
            print(f"Error processing row {i} ({row.get('Sinthetic Fluid', 'Unknown')}): {e}")

    if plot and fitted_curves:
        n_plots = len(fitted_curves)
        n_cols = 5
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), squeeze=False)

        for idx, curve in enumerate(fitted_curves):
            row_idx, col_idx = divmod(idx, n_cols)
            ax = axes[row_idx][col_idx]

            line1, = ax.plot(curve["gamma_dot"], curve["mu_cross"], label="Cross Model", alpha=0.6)
            line2, = ax.plot(curve["gamma_dot"], curve["mu_powerlaw"], '--', label="Power Law Fit", alpha=0.9)

            ax.set_title(f"{curve['fluid_id']} (R² = {curve['r2']:.4f})", fontsize=14)
            ax.set_xlabel("Shear Rate [s⁻¹]", fontsize=14)
            ax.set_ylabel("Viscosity [Pa·s]", fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=12)

            ax.set_xscale('log')
            ax.set_yscale('log')
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_locator(mticker.LogLocator(base=10.0))
                axis.set_major_formatter(mticker.ScalarFormatter())
                axis.get_major_formatter().set_scientific(False)
                axis.get_major_formatter().set_useOffset(False)

            ax.grid(True, which="both", ls="--")

        for i in range(n_plots, n_rows * n_cols):
            fig.delaxes(axes.flat[i])

        # Shared legend
        fig.legend([line1, line2], ['Cross Model', 'Power Law Fit'],
                loc='upper center', ncol=2, fontsize=14, frameon=False, bbox_to_anchor=(0.5, 1.02))

        plt.tight_layout()
        plt.subplots_adjust(top=0.94)  # Make space for legend
        plt.show()
def generate_sinthetic_fluids_LHS(operative_range, num_samples, filename=None):
    # Generate Latin Hypercube Samples
    lhs_samples = lhs(len(operative_range), samples=num_samples, criterion='maximin')

    # Scale the samples to the parameter ranges
    lhs_scaled = np.zeros_like(lhs_samples)
    for i, key in enumerate(operative_range.keys()):
        lhs_scaled[:, i] = lhs_samples[:, i] * (operative_range[key][1] - operative_range[key][0]) + operative_range[key][0]

    
    lhs_df = pd.DataFrame(lhs_scaled, columns=operative_range.keys())
    lhs_df.to_excel(f"{filename}.xlsx", index=False)
    print(f"Data saved as {filename}.xlsx")
    return lhs_df

def generate_fluid_pool_from_lhs(opran_dict, num_samples, filename=None):
    all_fluid_data = []

    for type_fluid in opran_dict:
        name = type_fluid['name']
        ranges = type_fluid['ranges']
        var_names = list(ranges.keys())
        samples_unit = lhs(len(var_names), samples=num_samples, criterion='maximin')
        # Scale samples to physical ranges
        samples_scaled = np.zeros_like(samples_unit)
        for i, var in enumerate(var_names):
            low, high = ranges[var]
            samples_scaled[:, i] = samples_unit[:, i] * (high - low) + low
        df = pd.DataFrame(samples_scaled, columns=var_names)
        df['FluidType'] = name
        all_fluid_data.append(df)
    df_all = pd.concat(all_fluid_data, ignore_index=True)

    if filename:
        if filename.endswith(".xlsx"):
            df_all.to_excel(filename, index=False)
        else:
            df_all.to_csv(filename, index=False)
        print(f"Fluid pool saved as {filename}")

    return df_all

#C. Data loading, descriptive analysis, and preprocessing
def load_excel_data(filepath, sheet_name):
    """
    Load data from an Excel file.
    Parameters:
        filepath (str): Path to the Excel file.
        sheet_name (str): Name of the sheet to read.
    Returns:
        df (pd.DataFrame): Loaded DataFrame with a liquid number id.
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    print(f" Loaded data with shape: {df.shape}")
    n_fluids = df['Liquid'].nunique()
    fluids_name = df['Liquid'].unique()
    print(f" Loaded Fluids: {n_fluids}")
    fluid_mapping = {name: idx for idx, name in enumerate(fluids_name)}
    df['LiquidNo'] = df['Liquid'].map(fluid_mapping)
    drop_columns = ['Liquid','Concentration', 'Flow Rate(GPM)', 'Flow Rate[m3/h]', 'Omega(rpm)']
    if drop_columns:
        df.drop(columns=drop_columns, inplace=True, errors='ignore')
        print(f"Dropped columns: {drop_columns}")

    return df

def decriptive_stats(data,bins,img_name=None):
    indicators = {
        'Mean': [],
        'Median': [],
        'StdDev': [],
        'Variance': []
    }
    for col in data.columns[1:]:
        mean = data[col].mean()
        median = data[col].median()
        std = data[col].std()
        var = data[col].var()

        indicators['Mean'].append(mean)
        indicators['Median'].append(median)
        indicators['StdDev'].append(std)
        indicators['Variance'].append(var)

    indicators_data = pd.DataFrame(indicators, index=data.columns[1:])
    sns.set_theme(style="white")

    # Define the number of rows and columns for the grid
    num_cols = 3  # 3 variables per row
    num_rows = (len(data.columns[1:]) + num_cols - 1) // num_cols  # Calculate rows needed
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axs = axs.flatten()  # Flatten the axes array for easy indexing

    # Define a list of colors to cycle through
    colors = sns.color_palette("hls", 10)

    # Plot each variable in its respective subplot
    for i, col in enumerate(data.columns[1:]):
        sns.histplot(data=data, x=col, kde=True, color=colors[i % len(colors)], ax=axs[i], bins=bins)
        
        # Increase font size for axis labels and tick labels
        axs[i].set_xlabel(col, fontsize=16)
        axs[i].set_ylabel('Frecuency', fontsize=16)
        axs[i].tick_params(axis='both', which='major', labelsize=16)

        # Calculate statistics
        mean = data[col].mean()
        std = data[col].std()
        var = data[col].var()

        # Add statistics as text to the plot
        stats_text = f"Mean: {mean:.2f}\nStd: {std:.2f}\nVar: {var:.2f}"
        axs[i].text(0.95, 0.95, stats_text, transform=axs[i].transAxes, 
                    fontsize=16, verticalalignment='top', horizontalalignment='right', 
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    # Adjust layout
    plt.tight_layout()

    # Save plot if requested
    if img_name:
        plt.savefig(img_name, bbox_inches='tight')

    return indicators_data

def apply_bep_scaling(df):
    """
    Applies BEP-based normalization to each fluid group using the row with max efficiency.
    Parameters:
        df (pd.DataFrame): Input raw DataFrame.
    Returns:
        model_filer(pd.DataFrame): DataFrame with new dimensionless columns
    """
    scaled_df = pd.DataFrame()
    for fluid_id in df['LiquidNo'].unique():
        group = df[df['LiquidNo'] == fluid_id].copy()
        bep_row = group.loc[group['Efficiency[-]'].idxmax()]

        group['massflow_dim'] = group['Mass Flow[kg/s]'] / bep_row['Mass Flow[kg/s]']
        group['torque_dim'] = group['Torque[N-m]'] / bep_row['Torque[N-m]']
        group['head_dim'] = group['TotalHead[m]'] / bep_row['TotalHead[m]']
        group['breakpower_dim'] = group['Break Power[W]'] / bep_row['Break Power[W]']
        group['hydraulicpower_dim'] = group['Hydraulic Power[W]'] / bep_row['Hydraulic Power[W]']
        group['eff_dim'] = group['Efficiency[-]'] / bep_row['Efficiency[-]']
        scaled_df = pd.concat([scaled_df, group], axis=0)
    scaled_df.reset_index(drop=True, inplace=True)
       # Final modeling dataset
    model_df = scaled_df[[
        'LiquidNo',
        'rho(kg/m3)', 'k(Pa*s^n)', 'n(-)',
        'massflow_dim', 'torque_dim', 'hydraulicpower_dim', 'breakpower_dim', 'eff_dim',
        'head_dim' 
        #'Reynolds Newtonian (-)', 'Reynolds non-Newtonian (-)', 'Aparent Dynamic Viscosity N(Pa*s)',
        #'Aparent Dynamic Viscosity PL(Pa*s)', 'Kinematic Viscosity N(cSt)', 'Kinematic Viscosity PL(cSt)', 'Head Coefficient N (-)',
        #'Head Coefficient PL (-)', 'Flow Coefficient N(-)', 'Flow Coefficient PL(-)', 'Hydraulic Power Coefficient N (-)', 'Hydraulic Power Coefficient PL (-)',
        #'Specific Speed N(-)', 'Specific Speed PL(-)'
    ]].copy()
    # Remove outliers based on TotalHead[m]
    model_filter = model_df[
    (model_df['head_dim'] > 0.001) &
    (model_df['head_dim'] < 3)].reset_index(drop=True)

    return scaled_df, model_filter

def scale_data(df, df_test, feature_cols, target_col):
    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)
    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values.reshape(-1, 1)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X)
    X_test_scaled = scaler_X.transform(X_test)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y)
    y_test_scaled = scaler_y.transform(y_test)
    print(f"Train set size: {X.shape[0]} samples.")
    print(f"Test set size: {X_test.shape[0]} samples (FIXED TEST SET).")
    print("Features and target scaled.")
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y

def scale_split_random(df, df_test, feature_cols, target_col, val_size, random_state=100):
    """
    Scales input features and target using StandardScaler. Also split the data into training and validation sets.
    Parameters:
        df (pd.DataFrame): The input dataset (Bep scaled dataset).
        feature_cols (list): List of columns to use as features.
        target_col (str): Column to use as the target variable.
    Returns:
        X_scaled (np.array): Scaled features.
        y_scaled (np.array): Scaled target.
        scaler_X (StandardScaler): Scaler fitted on features.
        scaler_y (StandardScaler): Scaler fitted on target.
    **First scaling and then split 
    """
    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)
    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values.reshape(-1, 1)
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state)
    print(f"Split completed: {X_train.shape[0]} train samples, {X_val.shape[0]} validation samples.")
    print(f"Test set size: {X_test.shape[0]} samples (FIXED TEST SET).")

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)
    print("Features and target scaled.")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, scaler_X, scaler_y

def splitandscale_byfluid(df_bepscaled, df_testbep, feature_cols, target_col, val_size, random_state=100):
    """"
    **First split and then scaling
    """
    fluid_ids = df_bepscaled['LiquidNo'].unique()
    fluid_train_ids, fluid_val_ids = train_test_split(fluid_ids, test_size=val_size, random_state=random_state)

    train_df = df_bepscaled[df_bepscaled['LiquidNo'].isin(fluid_train_ids)]
    val_df = df_bepscaled[df_bepscaled['LiquidNo'].isin(fluid_val_ids)]

    # Get X and y
    X_train = train_df[feature_cols].values
    y_train = train_df[[target_col]].values.reshape(-1, 1)

    X_val = val_df[feature_cols].values
    y_val = val_df[[target_col]].values.reshape(-1, 1)

    X_test = df_testbep[feature_cols].values
    y_test = df_testbep[[target_col]].values.reshape(-1, 1)
    print(f"Split completed: {X_train.shape[0]} train samples, {X_val.shape[0]} validation samples.")
    print(f"Test set size: {X_test.shape[0]} samples (FIXED TEST SET).")

    # Apply scaling
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)
    print("Features and target scaled.")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, scaler_X, scaler_y
    
#D. Baseline model training and feature analysis
def xgboost_with_cv(X_train, y_train, X_val, y_val, X_test, y_test, param_dist, n_iter, cv, img_name):
    """
    Trains an XGBoost regressor using RandomizedSearchCV with cross-validation.
    Parameters:
        X_train (np.array): Training features.
        y_train (np.array): Training targets.
        X_val (np.array): Validation features.
        y_val (np.array): Validation target.
        X_test (np.array): Test features.
        y_test (np.array): Test target.
        param_dist (dict): Dictionary of hyperparameter search space.
        n_iter (int): Number of random combinations to try.
        cv (int): Number of cross-validation folds.
    Returns:
        best_model (XGBRegressor): Trained model with best hyperparameters.
        results (dict): Dictionary with RMSE, R², and best parameters.
    """
    # Define model
    xgb_model = XGBRegressor(
        objective='reg:squarederror',
        booster='gbtree',
        random_state=42
    )

    # Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    fit_params = {'early_stopping_rounds':10,
                  'eval_set': [(X_val, y_val)],
                  'verbose': False}

    #Run randomized search on training set
    random_search.fit(X_train, y_train.ravel(), **fit_params) 
    best_model = random_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_val = best_model.predict(X_val)
    y_pred_test = best_model.predict(X_test)

    # Evaluate
    results = {
        'mse_train': mean_squared_error(y_train, y_pred_train),
        'r2_train': r2_score(y_train, y_pred_train),
        'mae_train': mean_absolute_error(y_train, y_pred_train),
        'ev_train': explained_variance_score(y_train, y_pred_train),
        'mse_val': mean_squared_error(y_val, y_pred_val),
        'r2_val': r2_score(y_val, y_pred_val),
        'mae_val': mean_absolute_error(y_val, y_pred_val),
        'ev_val': explained_variance_score(y_val, y_pred_val),
        'mse_test': mean_squared_error(y_test, y_pred_test),
        'r2_test': r2_score(y_test, y_pred_test),
        'mae_test': mean_absolute_error(y_test, y_pred_test),
        'ev_test': explained_variance_score(y_test, y_pred_test),
        'best_params': random_search.best_params_
    }

    #Output results
    for k,v in results.items():
        if not k.startswith('best_'):
            print(f"{k}: {v:.4f}")
    print("Best hyperparameters:", results['best_params'])

    #Ploting predicitions vs real values
    plot_predictions_subplots(y_train, y_pred_train, None, y_val, y_pred_val, None, y_test, y_pred_test, None, img_name=img_name)
    plot_hyperparameter_heatmap(random_search, x_param='param_learning_rate', y_param='param_n_estimators', metric='mean_test_score', img_name=img_name)
    
    return best_model, results

def run_pca_and_print_top_features(X, feature_names, n_components=3, top_k=3):
    """
    Run PCA on the dataset and print the top features for each principal component..
    """
    pca = PCA(n_components=n_components)
    pca.fit(X)
    for i, component in enumerate(pca.components_):
        print(f"\n Principal Component {i+1} (Explained Variance: {pca.explained_variance_ratio_[i]:.2%})")
        # Get indices of top absolute values
        top_indices = np.argsort(np.abs(component))[-top_k:][::-1]
        for idx in top_indices:
            feature = feature_names[idx]
            weight = component[idx]
            print(f"  • {feature:20s}: {weight:.4f}")


def gpr_with_cv(kernel, X_train, y_train, X_test, y_test, param_dtst, n_iter, cv, img_name):
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, random_state=42)
    #Set Up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=gpr,
        param_distributions=param_dtst,
        n_iter=n_iter,
        cv=cv,
        scoring= 'neg_mean_squared_error',
        verbose=1,
        random_state=42,
        n_jobs=-1)
    #Run randomized search on training set
    random_search.fit(X_train, y_train.ravel())
    best_params = random_search.best_params_
    #Re-train the best modeL
    best_model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        random_state=42)
    best_model.set_params(**best_params)
    best_model.fit(X_train, y_train.ravel())
    print(best_model.kernel_.get_params().keys())
    # Predictions
    y_pred_train, ysigma_pred_train = best_model.predict(X_train, return_std=True)
    y_pred_test, ysigma_pred_test = best_model.predict(X_test, return_std=True)
    # Evaluate
    results = {
        'mse_train': mean_squared_error(y_train, y_pred_train),
        'r2_train': r2_score(y_train, y_pred_train),
        'mae_train': mean_absolute_error(y_train, y_pred_train),
        'ev_train': explained_variance_score(y_train, y_pred_train),
        'mse_test': mean_squared_error(y_test, y_pred_test),
        'r2_test': r2_score(y_test, y_pred_test),
        'mae_test': mean_absolute_error(y_test, y_pred_test),
        'ev_test': explained_variance_score(y_test, y_pred_test),
        'best_params': best_params,
    }
    results_uncertainty = {
        'scaled_uncertainty_train': ysigma_pred_train,
        'scaled_uncertainty_test': ysigma_pred_test
    }
    #Output results
    for key, val in results.items():
        if 'params' not in key:
            print(f"{key}: {val:.4f}")
    print("Best hyperparameters:", best_params)

    #Ploting predicitions vs real values
    plot_predictions_subplots(y_train, y_pred_train, ysigma_pred_train, None, None, None, y_test, y_pred_test, ysigma_pred_test,img_name=img_name)
    plot_hyperparameter_heatmap(random_search, x_param='param_kernel__k2__noise_level', y_param='param_alpha', metric='mean_test_score', img_name=img_name)
    return best_model, results, results_uncertainty

#E. General plotting utilities for model evaluation
def plot_predictions_subplots(
    y_train, y_train_pred, y_train_std=None,
    y_val=None,  y_val_pred=None,  y_val_std=None,
    y_test=None, y_test_pred=None, y_test_std=None,
    tolerance=0.2,
    img_name: str = 'no'
):
    """
    Plot train, validation, and test predictions in one axes.
    - y_*       : actual values
    - y_*_pred  : predicted values
    - y_*_std   : standard deviation (only for GPR; can be None)
    """

    # Build list of (label, actual, pred, std, color)
    sets = [
        ('Train',      y_train,      y_train_pred,  y_train_std,  'C0'),
    ]
    if y_val is not None and y_val_pred is not None:
        sets.append(('Validation', y_val,       y_val_pred,   y_val_std,   'C1'))
    if y_test is not None and y_test_pred is not None:
        sets.append(('Test',       y_test,      y_test_pred,  y_test_std,  'C2'))

    # Flatten all actuals for plotting the 1:1 and tolerance lines
    all_actuals = np.hstack([s[1].ravel() for s in sets])
    mn, mx = all_actuals.min(), all_actuals.max()
    x_line = np.linspace(mn, mx, 200)

    # White background, no grid
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('white')
    ax.grid(True)

    # Plot each set
    for label, y_true, y_pred, y_std, color in sets:
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        if y_std is not None:
            y_err = 2 * y_std.ravel()
            ax.errorbar(
                y_true, y_pred,
                yerr=y_err,
                fmt='o',
                ecolor=color,
                elinewidth=1,
                capsize=2,
                markerfacecolor=color,
                markeredgecolor='k',
                alpha=0.6,
                label=f'{label} ±2σ'
            )
        else:
            ax.scatter(
                y_true, y_pred,
                c=color,
                alpha=0.6,
                label=label
            )

    # 1:1 ideal line
    ax.plot(x_line, x_line, 'k--', lw=1, label='Ideal')
    # ±20% lines
    ax.plot(x_line, x_line * (1 + tolerance), 'r--', lw=0.8, label='+20%')
    ax.plot(x_line, x_line * (1 - tolerance), 'r--', lw=0.8, label='−20%')

    ax.set_xlabel('Actual Scaled Head', fontsize=16)
    ax.set_ylabel('Predicted Scaled Head', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(loc='best', frameon=False, fontsize=14, ncol=2)

    plt.tight_layout()

    if img_name and img_name != 'no':
        from pathlib import Path
        out_path = Path(f"{img_name}_combined_predictions.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {out_path}")

    plt.show()

def plot_feature_importance(model, feature_names, img_name):
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1][:10]

    # Switch to a white background style
    plt.style.use('default')
    plt.rcParams['axes.grid'] = False
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

    plt.figure(figsize=(10, 6))
    bars = plt.barh(
        np.array(feature_names)[sorted_idx][::-1],
        importance[sorted_idx][::-1],
        color='skyblue'
    )

    plt.xlabel("Feature Importance", fontsize=24)
    plt.ylabel("Features", fontsize=24)
    plt.title("Top 10 Most Important Features", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.grid(True, axis='x', linestyle='--', alpha=0.4)

    plt.tight_layout()
    if img_name != 'no':
        name = img_name + '_feature_importance.png'
        from pathlib import Path
        save_plot(name)
    plt.show()

def plot_hyperparameter_heatmap(random_search, x_param, y_param, metric='mean_squared_error', img_name=str):
    results_df = pd.DataFrame(random_search.cv_results_)

    # Pivot to create matrix for heatmap
    heatmap_data = results_df.pivot_table(
        index=y_param,
        columns=x_param,
        values=metric,
        aggfunc='mean'
    )

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="viridis", cbar_kws={'label': 'Mean Squared Error'})
    plt.title(f"Hyperparameter Tuning Heatmap")
    plt.xlabel(x_param.replace('param_', '').replace('_', ' ').title())
    plt.ylabel(y_param.replace('param_', '').replace('_', ' ').title())
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))   
    if img_name != 'no':
        name = img_name+'_hyperparameter_heatmap.png'
        save_plot(name)
    plt.tight_layout()
    plt.show()

#F. Active learning sampling and dataset augmentation
def greedy_sampling_inputs(df_train, df_pool, n_select):
    input_features = ['rho(kg/m3)', 'k(Pa*s^n)', 'n(-)']
    #Extract the input features
    X_train = df_train[input_features].copy()
    X_pool = df_pool[input_features].copy()
    #Scale the input features
    scaler_inputs = StandardScaler()
    X_train_scaled = scaler_inputs.fit_transform(X_train)
    X_pool_scaled = scaler_inputs.transform(X_pool)
    #Calculate the pairwise distances from pool to train set
    distances = pairwise_distances(X_pool_scaled, X_train_scaled)
    min_distances = distances.min(axis=1)
    #Select the n candidates
    selected_indices = np.argsort(-min_distances)[:n_select]
    selected_df = df_pool.iloc[selected_indices].copy()
    #Remove the selected candidates from the pool
    updated_pool = df_pool.drop(selected_indices).reset_index(drop=True)

    print(f"Selected {n_select} new candidates:")
    print(f"Remaining pool size: {updated_pool.shape[0]}")

    return selected_df, updated_pool

def greedy_sampling_inputs_outputs(model, df_trainBEP, df_poolraw, scaler_X, features_cols, n_select=2):
    #Extract and scale input features
    X_train = df_trainBEP[features_cols].copy()
    X_pool = df_poolraw[features_cols].copy()

    X_train_scaled = scaler_X.transform(X_train)
    X_pool_scaled = scaler_X.transform(X_pool)

    #Using the current model predict the ouputs for both sets
    y_train_pred = model.predict(X_train_scaled).reshape(-1, 1)
    y_pool_pred = model.predict(X_pool_scaled).reshape(-1, 1)

    #Calculate the pairwise distances from pool to train set on both inputs and outputs
    dist_inputs = pairwise_distances(X_pool_scaled, X_train_scaled)
    dist_outputs = pairwise_distances(y_pool_pred, y_train_pred)
    min_distances_inputs = dist_inputs.min(axis=1)
    min_distances_outputs = dist_outputs.min(axis=1)

    #Normalize the distances
    norm_distX = min_distances_inputs / min_distances_inputs.max()
    norm_distY = min_distances_outputs / min_distances_outputs.max()
    score = norm_distX + norm_distY

    #Select the top scoring candidates
    selected_indices = np.argsort(-score)[:n_select]
    selected_df = df_poolraw.iloc[selected_indices].copy()
    updated_pool = df_poolraw.drop(selected_indices).reset_index(drop=True)
    print(f"Selected {n_select} new candidates:")
    print(f"Remaining pool size: {updated_pool.shape[0]}")
    
    return selected_df, updated_pool

def loop_add_fluids_in_order(int_train_df,add_fluids_df, batch_size=2, fluid_id_column='LiquidNo',verbose=True):
    """
    This function takes the initial train dataset and additional fluids dataset, and generates a list of training sets
    combining the intial and additional fluids in batches.
    """
    #making the number of the fluids unique
    add_fluids_df = make_liquidno_unique(int_train_df, add_fluids_df, fluid_id_column)
    current_train = int_train_df.copy()
    new_fluids = add_fluids_df[fluid_id_column].unique()  # In order of appearance

    train_sets = []
    fluids_list = []
    initial_fluids = current_train[fluid_id_column].nunique()
    fluids_list.append(initial_fluids)  

    for i in range(0, len(new_fluids), batch_size):
        batch = new_fluids[i:i + batch_size]
        batch_data = add_fluids_df[add_fluids_df[fluid_id_column].isin(batch)]
        current_train = pd.concat([current_train, batch_data], ignore_index=True)
        train_sets.append(current_train.copy())
        num_fluids = current_train[fluid_id_column].nunique()
        fluids_list.append(num_fluids)        
        if verbose:
            print(f"Iteration {i // batch_size + 1}: New training set includes {num_fluids} unique fluids")
    return train_sets, fluids_list

def make_liquidno_unique(train_df, add_df, fluid_id_column):
    """"
    This function makes the fluid IDs in the additional DataFrame unique by adding an offset"""
    max_train = train_df[fluid_id_column].max()
    if add_df[fluid_id_column].min() <= max_train:
        offset = max_train + 1
        add_df[fluid_id_column] = add_df[fluid_id_column] + offset
    return add_df

#G. Multi-training-set preparation and repeated training
def split_scale_by_fluid_multiple_trainsets(train_sets, test_setBEP, feature_cols, target_col, val_size, random_state=100):
    """
    This function takes a list of training sets and splits and scales each one into training and validation sets.
    """
    X_train_sets = []
    y_train_sets = []
    X_val_sets = []
    y_val_sets = []
    X_test_sets = []
    y_test_sets = []
    for train_set in train_sets:
        X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, _, _=splitandscale_byfluid(train_set, test_setBEP, feature_cols, 
                                                                                                     target_col, val_size, random_state)
        X_train_sets.append(X_train_scaled)
        y_train_sets.append(y_train_scaled)
        X_val_sets.append(X_val_scaled)
        y_val_sets.append(y_val_scaled)
        X_test_sets.append(X_test_scaled)
        y_test_sets.append(y_test_scaled)

    return X_train_sets, y_train_sets, X_val_sets, y_val_sets, X_test_sets, y_test_sets

def scale_multiple_trainsets(train_sets, test_setBEP, feature_cols, target_col):
    """
    This function takes a list of training sets and scales each one.
    """
    X_train_sets = []
    y_train_sets = []
    X_test_sets = []
    y_test_sets = []
    for train_set in train_sets:
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled ,_, _ = scale_data(train_set, test_setBEP, feature_cols, target_col)
        X_train_sets.append(X_train_scaled)
        y_train_sets.append(y_train_scaled)
        X_test_sets.append(X_test_scaled)
        y_test_sets.append(y_test_scaled)
    return X_train_sets, y_train_sets, X_test_sets, y_test_sets

def train_multiple_XGBoost(X_train_sets, y_train_sets, X_val_sets, y_val_sets, X_test_sets, y_test_sets, param_dist, n_iter, cv, img_name='no'):

    """
    Trains and evaluates XGBoost models on multiple training sets with the same hyperparameter tuning config.
    """
    models = []
    results_list = []

    for i in range(len(X_train_sets)):
        print(f"\nTraining iteration {i + 1}/{len(X_train_sets)}")

        X_train = X_train_sets[i]
        y_train = y_train_sets[i]
        X_val = X_val_sets[i]
        y_val = y_val_sets[i]
        X_test = X_test_sets[i]
        y_test = y_test_sets[i]

        # Handle image naming if enabled
        if img_name != 'no':
            img_name_iter = f"{img_name}_iter{i + 1}"
        else:
            img_name_iter = 'no'
        best_model, results = xgboost_with_cv(X_train, y_train, X_val, y_val, X_test, y_test, param_dist, n_iter, cv, img_name=img_name_iter)

        # Store model and results
        models.append(best_model)
        results_list.append(results)

    return models, results_list

def alpha_scheduler_GPR(t, alpha_0, k=0.2, p=0.7):
    return alpha_0/(1+k*t**p)

def train_multiple_GPR(kernel, X_train_sets, y_train_sets, X_test_sets, y_tests_sets, param_dist_NOalpha, alpha_0, n_iter, cv, img_name='no'):
    models = []
    results_list = []
    results_uncertainty_list = []
    for i in range(len(X_train_sets)):
        print(f"\nTraining iteration {i + 1}/{len(X_train_sets)}")
        X_train = X_train_sets[i]
        y_train = y_train_sets[i]
        X_test = X_test_sets[i]
        y_test = y_tests_sets[i]
        # Handle image naming if enabled
        if img_name != 'no':
            img_name_iter = f"{img_name}_iter{i + 1}"
        else:
            img_name_iter = 'no'
        
        #Dynamic alpha
        current_alpha = alpha_scheduler_GPR(i,alpha_0)
        alpha_range = np.linspace(current_alpha-0.005, current_alpha+0.005)
        param_dist = param_dist_NOalpha.copy()
        param_dist['alpha'] = alpha_range
        
        #Train model
        best_model, results, results_uncertainty= gpr_with_cv(kernel, X_train, y_train, X_test, y_test, param_dist, n_iter, cv, img_name=img_name_iter)
        # Store model and results
        models.append(best_model)
        results_list.append(results)
        results_uncertainty_list.append(results_uncertainty)
    
    return models, results_list, results_uncertainty_list

#H. Query-based / variational GPR candidate generation and optimization
def filter_grid(df_grid):
    #Power law invalid fluids 
    #1. low viscosity + low n combinations
    #2. high viscosity + low n combinations
    pl_invalid = (
        (df_grid["FluidType"] == "PowerLaw") & (
            ((df_grid["k(Pa*s^n)"] < 1.0) & (df_grid["n(-)"] < 0.75))
        )
    )             
    print(f"Filtered out {pl_invalid.sum()} invalid Power Law fluid samples.")
    return df_grid[~pl_invalid].reset_index(drop=True)

def vargp_predictions_cluster(model, grid, scaler_X, features_cols, make_plot=None):
    X_grid = grid[features_cols].copy()
    Xgrid_scaled = scaler_X.transform(X_grid)
    #Use the current model to create the field of predictions
    y_pred, ysigma_pred = model.predict(Xgrid_scaled, return_std= True)
    grid['y_pred'] = y_pred
    grid['sigma'] = ysigma_pred

    #Normalize sigma within fluid type for fair selection
    top_uncertain_list = []
    for fluid_type in grid['FluidType'].unique():
        group = grid[grid['FluidType'] == fluid_type].copy()
        group['sigma_norm'] = (group['sigma'] - group['sigma'].min()) / (group['sigma'].max() - group['sigma'].min() + 1e-8)
        top_n = max(1, int(len(group) * 0.03))
        top_uncertain = group.nlargest(top_n, 'sigma_norm')
        top_uncertain_list.append(top_uncertain)

    top_uncertain_df = pd.concat(top_uncertain_list).reset_index(drop=True)
    top_uncertain_df = top_uncertain_df.drop(columns=['sigma_norm'])
    return top_uncertain_df


def vargp_bayesoptimization(model, selected_fluids_df, opran_dict, features_cols, n_calls=25):
    optimized_fluids = []
    # Build a lookup for fluid bounds
    bounds_dict = {entry['name']: entry['ranges'] for entry in opran_dict}

    for _, fluid in selected_fluids_df.iterrows():
        fluid_type = fluid['FluidType']
        bounds = bounds_dict[fluid_type]

        # Define search space based on bounds
        space = []
        static_values = {}
        for key in features_cols:
            lb, ub = bounds[key]
            if lb == ub:
                static_values[key] = lb  # fixed input
            else:
                space.append(Real(lb, ub, name=key))

        # Define the objective function depending on the space
        if space:
            @use_named_args(space)
            def objective(**params):
                full_input = [params[key] if key in params else static_values[key] for key in features_cols]
                _, std = model.predict([full_input], return_std=True)
                return -std[0]  # Maximize uncertainty
        else:
            # All variables are fixed → no optimization
            full_input = [static_values[key] for key in features_cols]
            _, sigma = model.predict([full_input], return_std=True)
            full_input_dict = {k: v for k, v in zip(features_cols, full_input)}
            full_input_dict['sigma'] = sigma[0]
            full_input_dict['FluidType'] = fluid_type
            optimized_fluids.append(full_input_dict)
            continue

        # Run Bayesian Optimization
        res = gp_minimize(objective, space, n_calls=n_calls, random_state=42)
        optimized_input = res.x
        input_dict = {dim.name: val for dim, val in zip(res.space.dimensions, optimized_input)}

        # Safely reconstruct full input
        full_input = []
        for k in features_cols:
            if k in input_dict:
                full_input.append(input_dict[k])
            elif k in static_values:
                full_input.append(static_values[k])
            else:
                raise KeyError(f"'{k}' not found in input_dict or static_values. "
                               f"Debug info → input_dict: {input_dict}, static_values: {static_values}")

        _, sigma = model.predict([full_input], return_std=True)
        full_input_dict = {k: v for k, v in zip(features_cols, full_input)}
        full_input_dict['sigma'] = sigma[0]
        full_input_dict['FluidType'] = fluid_type
        optimized_fluids.append(full_input_dict)

    # Return full results and top 2 by sigma
    optimized_df = pd.DataFrame(optimized_fluids).sort_values(by='sigma', ascending=False).reset_index(drop=True)
    top2_df = optimized_df.head(2)
    return optimized_df, top2_df


def vargp_ucb_optimization(model, selected_fluids_df, grid, opran_dict, features_cols, kappa, uniqueness_tol, prev_samples=None, n_calls=50):
    optimized_fluids = []
    # Bounds dictionary
    bounds_dict = {entry['name']: entry['ranges'] for entry in opran_dict}
    # Initialize previous samples
    if prev_samples is None:
        prev_samples = pd.DataFrame(columns=features_cols)

    # Uniqueness checker with logging
    check_cols = ['k(Pa*s^n)', 'n(-)', 'massflow_dim']
   
    for _, fluid in selected_fluids_df.iterrows():
        fluid_type = fluid['FluidType']
        fluid_density = fluid['rho(kg/m3)']
        bounds = bounds_dict[fluid_type]
        # Set bounds and fixed/static values
        opt_bounds = []
        static_values = {}
        for key in features_cols:
            if key == 'massflow_dim':
                massflow_min = 0
                massflow_max = 25*0.000063090196*fluid_density
                opt_bounds.append((massflow_min,massflow_max))
            else:
                lb, ub = bounds[key]
                if lb == ub:
                    static_values[key] = lb
                else:
                    opt_bounds.append((lb, ub))

        # Objective function
        def ucb_objective(x):
            full_input = []
            var_idx = 0
            for key in features_cols:
                if key in static_values:
                    full_input.append(static_values[key])
                else:
                    full_input.append(float(x[var_idx]))
                    var_idx += 1
            rho, k, n, massflow = full_input

            candidate_important = [k, n, massflow]
            if not prev_samples.empty:
                distances = cdist([candidate_important],prev_samples[check_cols].values)
                if distances.min() < uniqueness_tol:
                    return 1e6
                
            if fluid_type == 'PowerLaw' and k < 1 and n < 0.75:
                return 1e6
            mu, sigma = model.predict([full_input], return_std=True)
            return -(mu[0] + kappa * sigma[0])
     
        result = dual_annealing(ucb_objective, bounds=opt_bounds, maxiter=n_calls, seed=rd.randint(1,10000))
        optimized_input = result.x

        # Reconstruct full input
        full_input = []
        var_idx = 0
        for key in features_cols:
            if key in static_values:
                full_input.append(static_values[key])
            else:
                full_input.append(float(optimized_input[var_idx]))
                var_idx += 1
        # Final prediction
        mu, sigma = model.predict([full_input], return_std=True)
        full_input_dict = {k: v for k, v in zip(features_cols, full_input)}
        full_input_dict['sigma'] = sigma[0]
        full_input_dict['mean_prediction'] = mu[0]
        full_input_dict['UCB'] = mu[0] + kappa * sigma[0]
        full_input_dict['FluidType'] = fluid_type
        optimized_fluids.append(full_input_dict)

    # Format and return
    optimized_df = pd.DataFrame(optimized_fluids).sort_values(by='UCB', ascending=False).reset_index(drop=True)
    top_df = optimized_df.drop_duplicates(subset=['FluidType']).head(1)

    # Add fluid random from the grid
    remain_grid = grid[~grid[features_cols].apply(tuple, axis=1).isin(
        pd.DataFrame(optimized_df)[features_cols].apply(tuple, axis=1)
    )]
    if not remain_grid.empty:
        random_fluid = remain_grid.sample(n=1, random_state= rd.randint(1,10000) ).reset_index(drop=True)

            # Add placeholder columns so it matches the optimized one
        random_fluid['sigma'] = np.nan
        random_fluid['mean_prediction'] = np.nan
        random_fluid['UCB'] = np.nan
    else:
        random_fluid = pd.DataFrame(columns=features_cols + ['FluidType', 'UCB', 'sigma', 'mean_prediction'])

    combined_top2_df = pd.concat([top_df, random_fluid], ignore_index=True)

    print(f"\n Selected by the highest UCB:")
    print(top_df[features_cols + ['FluidType', 'UCB', 'sigma']])

    print(f"\n Selected randomly from the grid:")
    print(random_fluid[features_cols + ['FluidType']])

    return optimized_df, top_df, combined_top2_df

#I. Active learning performance and uncertainty visualizatio
def plot_metrics_vs_fluids(resultsbasemodel, results_list, fluids_list):
    """
    Plots the MSE and R² vs the number of fluids for Train and Test sets.
    Different marker shapes are used per set and legend is shown only once.
    """

    # Initialize lists with base model metrics
    mse_train = [resultsbasemodel['mse_train']]
    r2_train = [resultsbasemodel['r2_train']]
    mse_test = [resultsbasemodel['mse_test']]
    r2_test = [resultsbasemodel['r2_test']]

    for results in results_list:
        mse_train.append(results['mse_train'])
        r2_train.append(results['r2_train'])
        mse_test.append(results['mse_test'])
        r2_test.append(results['r2_test'])

    # Combine data into long-format DataFrame
    df = pd.DataFrame({
        'Number of Fluids': fluids_list * 2,
        'MSE': mse_train + mse_test,
        'R2': r2_train + r2_test,
        'Set': ['Train'] * len(fluids_list) + ['Test'] * len(fluids_list)
    })

    # Plotting setup
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Custom markers
    marker_dict = {'Train': 'o', 'Test': '^'}

    # MSE Plot
    for set_label, marker in marker_dict.items():
        subset = df[df['Set'] == set_label]
        axs[0].plot(subset['Number of Fluids'], subset['MSE'],
                    marker=marker, label=set_label, markersize=6)
    axs[0].set_xlabel("Number of Fluids", fontsize=14)
    axs[0].set_ylabel("MSE", fontsize=14)
    axs[0].tick_params(labelsize=12)

    # R² Plot (no legend here)
    for set_label, marker in marker_dict.items():
        subset = df[df['Set'] == set_label]
        axs[1].plot(subset['Number of Fluids'], subset['R2'],
                    marker=marker, label=set_label, markersize=6)
    axs[1].set_xlabel("Number of Fluids", fontsize=16)
    axs[1].set_ylabel("R² Score", fontsize=16)
    axs[1].tick_params(labelsize=16)

    # Only show legend once
    axs[0].legend(title="Set", fontsize=16, title_fontsize=13)

    plt.tight_layout()
    plt.show()

def plot_uncertainty_dist_weighted(uncertainties):
    train_sigma = np.array(uncertainties['scaled_uncertainty_train'])
    test_sigma = np.array(uncertainties['scaled_uncertainty_test'])

    # Mean values
    train_mean = np.mean(train_sigma)
    test_mean = np.mean(test_sigma)

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # Histograms with KDE
    sns.histplot(train_sigma, bins=30, kde=True, color='steelblue', label='Train σ', alpha=0.6)
    sns.histplot(test_sigma, bins=30, kde=True, color='orange', label='Test σ', alpha=0.6)

    # Mean lines
    plt.axvline(train_mean, color='blue', linestyle='--', lw=2, label=f"Train mean σ: {train_mean:.3f}")
    plt.axvline(test_mean, color='darkorange', linestyle='--', lw=2, label=f"Test mean σ: {test_mean:.3f}")

    # Labels and legend
    plt.xlabel("σ (Standard Deviation)", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(loc='upper right', fontsize=18, frameon=False, ncol=2)
    plt.tight_layout()
    plt.show()

def plot_multiple_testmetrics_vs_fluids(base_model_results, all_results_lists, fluids_list, labels, set_metrcis):
    """
    Plot MSE and R² for different paths vs. number of fluids, with a shared legend below the plots.
    """

    all_data = []
    for path_idx, results_list in enumerate(all_results_lists):
        mse_values = [base_model_results['mse_' + set_metrcis]]
        r2_values = [base_model_results['r2_' + set_metrcis]]

        for res in results_list:
            mse_values.append(res['mse_' + set_metrcis])
            r2_values.append(res['r2_' + set_metrcis])

        for i, fluids in enumerate(fluids_list):
            all_data.append({
                'Number of Fluids': fluids,
                'MSE': mse_values[i],
                'R2': r2_values[i],
                'Path': labels[path_idx]
            })

    df = pd.DataFrame(all_data)

    markers = ['o', 's', 'D', '^', 'v', '<', '>']
    palette = sns.color_palette("tab10", len(labels))
    sns.set(style="whitegrid", font_scale=1.8)

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for i, path in enumerate(labels):
        subset = df[df['Path'] == path]
        axs[0].plot(
            subset['Number of Fluids'], subset['MSE'],
            label=path,
            marker=markers[i % len(markers)],
            linewidth=2, markersize=7,
            color=palette[i]
        )
        axs[1].plot(
            subset['Number of Fluids'], subset['R2'],
            label=path,
            marker=markers[i % len(markers)],
            linewidth=2, markersize=7,
            color=palette[i]
        )

    axs[0].set_ylabel('MSE ' + " " + set_metrcis)
    axs[1].set_ylabel('R² ' + " " + set_metrcis)
    axs[1].set_xlabel('Number of Fluids')

    # Create a shared legend outside the plot
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(labels), frameon=False, fontsize=16)

    plt.tight_layout(rect=[0, -0.05, 1, 1])  # Leave space at the bottom
    plt.show()


def plot_uncertainty_vs_fluids_vertical(all_uncertainty_dicts, fluids_list, labels):
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Marker shapes for up to 10 paths
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']

    for path_idx, uncertainty_path in enumerate(all_uncertainty_dicts):
        mean_train, min_train, max_train = [], [], []
        mean_test, min_test, max_test = [], [], []

        for u_dict in uncertainty_path:
            train = np.array(u_dict['scaled_uncertainty_train'])
            test = np.array(u_dict['scaled_uncertainty_test'])

            mean_train.append(np.mean(train))
            min_train.append(np.min(train))
            max_train.append(np.max(train))

            mean_test.append(np.mean(test))
            min_test.append(np.min(test))
            max_test.append(np.max(test))

        fluids_arr = np.array(fluids_list)
        label = labels[path_idx]
        marker = markers[path_idx % len(markers)]

        axs[0].plot(fluids_arr, mean_train, label=label, marker=marker)
        axs[0].fill_between(fluids_arr, min_train, max_train, alpha=0.2)

        axs[1].plot(fluids_arr, mean_test, label=label, marker=marker)
        axs[1].fill_between(fluids_arr, min_test, max_test, alpha=0.2)

    # Labels and styling
    axs[0].set_ylabel('Train Uncertainty (σ)', fontsize=16)
    axs[1].set_ylabel('Test Uncertainty (σ)', fontsize=16)
    axs[1].set_xlabel('Number of Fluids', fontsize=16)
    axs[0].tick_params(axis='both', labelsize=16)
    axs[1].tick_params(axis='both', labelsize=16)

    axs[0].legend(loc='upper right', fontsize=16, frameon=False, ncols=2)  # Legend only on top subplot

    axs[0].grid(True)
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

#J. Rheology and fluid-behavior visualization
def plot_rheograms_for_group(df, fluid_ids, fluid_id_col='LiquidNo', k_col='k(Pa*sˆn)', n_col='n(-)',
                             γ_min=0.1, γ_max=1e4, γ_pts=300):
    
    figsize = (15,12)
    palette='tab20'
    legend_ncol=7
    legend_bbox=(0.5, -0.1)
    sns.set_style('whitegrid')
    # 3) common shear‐rate axis
    γ = np.logspace(np.log10(γ_min), np.log10(γ_max), γ_pts)

    # 4) get colors
    colors = sns.color_palette(palette, len(fluid_ids))

    # 5) plot each curve
    plt.figure(figsize=figsize)
    for col, fid in zip(colors, sorted(fluid_ids)):
        row = df.loc[df[fluid_id_col] == fid].iloc[0]
        μ = row[k_col] * γ**(row[n_col] - 1)
        label_text = f"{fid} (k={row[k_col]:.2g}, n={row[n_col]:.2f})"

        # Plot the line
        line, = plt.loglog(γ, μ, color=col, lw=1.5, label=label_text)

        # Add label at the end of the line
        plt.text(
            γ[-1] * 1.05,        # shift slightly right
            μ[-1],               # y value at end of curve
            str(fid),            # fluid ID only (or label_text if you want full)
            fontsize=8,
            color=col,
            verticalalignment='center',
            horizontalalignment='left',
            clip_on=True
        )
    plt.xlabel(r"Shear rate [sˆ-1]", fontsize=16)
    plt.ylabel(r'Viscosity [Pa.s]', fontsize=16)
    plt.grid(which='both', ls=':', alpha=0.5)
    plt.legend(
        loc='upper center',
        bbox_to_anchor=legend_bbox,
        ncol=legend_ncol,
        fontsize='small',
        frameon=False
    )
    plt.tight_layout()
    plt.show()

def plot_sparse_and_dense_rheograms(
    df,
    fluid_id_col: str = 'LiquidNo',
    k_col:        str = 'k(Pa*s^n)',
    n_col:        str = 'n(-)',
    min_pts:      int = 4,
    **plot_kwargs
):
    # filter to power‐law only
    pl = df[np.abs(df[n_col] - 1.0) > 1e-3]
    counts = pl[fluid_id_col].value_counts()
    sparse = counts[counts <  min_pts].index.tolist()
    dense  = counts[counts >= min_pts].index.tolist()

    # plot sparse (< min_pts)
    print(f"Power-law fluids with < {min_pts} samples")
    plot_rheograms_for_group(
        pl, sparse,
        fluid_id_col=fluid_id_col,
        k_col=k_col, n_col=n_col,
        **plot_kwargs
    )

    # plot dense (≥ min_pts)
    print(f"Power-law fluids with ≥ {min_pts} samples")
    plot_rheograms_for_group(
        pl, dense,
        fluid_id_col=fluid_id_col,
        k_col=k_col, n_col=n_col,
        **plot_kwargs
    )

def plot_combined_rheograms_by_density(
    df,
    fluid_id_col='LiquidNo',
    k_col='k(Pa*s^n)',
    n_col='n(-)',
    min_pts=4,
    γ_min=0.1,
    γ_max=1e4,
    γ_pts=300
):
    # Define shared gamma range
    γ = np.logspace(np.log10(γ_min), np.log10(γ_max), γ_pts)

    # Filter power-law fluids only (n ≠ 1)
    pl = df[np.abs(df[n_col] - 1.0) > 1e-3]
    counts = pl[fluid_id_col].value_counts()

    sparse_ids = counts[counts < min_pts].index.tolist()
    dense_ids  = counts[counts >= min_pts].index.tolist()

    # Set up figure and styles
    plt.figure(figsize=(12, 10))
    sns.set_style('whitegrid')

    for fid in sorted(sparse_ids):
        row = pl.loc[pl[fluid_id_col] == fid].iloc[0]
        μ = row[k_col] * γ**(row[n_col] - 1)
        plt.loglog(γ, μ, color='orange', lw=1.5, alpha=0.8)

    for fid in sorted(dense_ids):
        row = pl.loc[pl[fluid_id_col] == fid].iloc[0]
        μ = row[k_col] * γ**(row[n_col] - 1)
        plt.loglog(γ, μ, color='steelblue', lw=1.5, alpha=0.8)

    plt.xlabel(r"Shear Rate [s$^{-1}$]", fontsize=18)
    plt.ylabel(r"Viscosity [Pa·s]", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, which='both', ls=':', alpha=0.5)

    # Legend for both classes only
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='steelblue', lw=2, label=' Fluids with ≥ 4 Samples'),
        Line2D([0], [0], color='orange', lw=2, label='Fluids with < 4 Samples')
    ]
    plt.legend(handles=legend_elements, ncol=2, fontsize=18, frameon=False)

    plt.tight_layout()
    plt.show()


#K. Final result visualization blocks
def plot_train_test_metric_with_uncertainty_blocks_split(
    base_model_results,
    all_results_lists,
    fluids_list,
    labels,
    set_metrics,  # e.g., "mse", "mae", or "ev"
    all_uncertainty_dicts
):
    all_train_data = []
    all_test_data = []

    metric_key = set_metrics.lower()
    train_key = f"{metric_key}_train"
    test_key = f"{metric_key}_test"

    for path_idx, results_list in enumerate(all_results_lists):
        path_label = labels[path_idx]
        metric_train_values = [base_model_results[train_key]]
        metric_test_values = [base_model_results[test_key]]
        uncertainty_train_means = []
        uncertainty_test_means = []

        uncertainty_path = all_uncertainty_dicts[path_idx]
        for u_dict in uncertainty_path:
            unc_train = np.array(u_dict['scaled_uncertainty_train'])
            unc_test = np.array(u_dict['scaled_uncertainty_test'])
            uncertainty_train_means.append(np.mean(unc_train))
            uncertainty_test_means.append(np.mean(unc_test))

        for res in results_list:
            metric_train_values.append(res[train_key])
            metric_test_values.append(res[test_key])

        for i, fluids in enumerate(fluids_list):
            all_train_data.append({
                'Number of Fluids': fluids,
                'Metric': metric_train_values[i],
                'Uncertainty': 2 * uncertainty_train_means[i],
                'Path': path_label
            })
            all_test_data.append({
                'Number of Fluids': fluids,
                'Metric': metric_test_values[i],
                'Uncertainty': 2 * uncertainty_test_means[i],
                'Path': path_label
            })

    df_train = pd.DataFrame(all_train_data)
    df_test = pd.DataFrame(all_test_data)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    bar_width = 0.8 / len(labels)
    fluid_indices = np.arange(len(fluids_list))
    palette = sns.color_palette("tab10", len(labels))
    markers = ['o', '^', 's', 'D', 'v', 'P', '*', 'X', '<', '>']

    for i, path in enumerate(labels):
        color = palette[i]
        offset = i * bar_width - ((len(labels) - 1) * bar_width / 2)
        marker_train = markers[i % len(markers)]
        marker_test = markers[(i + 1) % len(markers)]

        # Train plot
        subset_train = df_train[df_train['Path'] == path].sort_values(by='Number of Fluids')
        x_train = fluid_indices + offset
        y_train = subset_train['Metric']
        yerr_train = subset_train['Uncertainty']
        ax1.bar(x_train, yerr_train, bottom=y_train - yerr_train / 2, width=bar_width,
                alpha=0.3, color=color, label=f'{path} ± 2σ')
        ax1.scatter(x_train, y_train, s=60, color=color, marker=marker_train)

        # Test plot
        subset_test = df_test[df_test['Path'] == path].sort_values(by='Number of Fluids')
        x_test = fluid_indices + offset
        y_test = subset_test['Metric']
        yerr_test = subset_test['Uncertainty']
        ax2.bar(x_test, yerr_test, bottom=y_test - yerr_test / 2, width=bar_width,
                alpha=0.6, color=color, label=path)
        ax2.scatter(x_test, y_test, s=60, color=color, marker=marker_test)

    # Font styling and labels
    ax1.set_ylabel(f'Train {set_metrics.upper()}', fontsize=16)
    ax2.set_ylabel(f'Test {set_metrics.upper()}', fontsize=16)
    ax2.set_xlabel('Number of Fluids', fontsize=16)

    ax1.tick_params(axis='both', labelsize=16)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.set_xticks(fluid_indices)
    ax2.set_xticklabels(fluids_list, fontsize=16)

    # Move legend to ax2 in upper left corner inside the plot
    ax2.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), ncol=2, fontsize=16, frameon=False)

    plt.tight_layout()
    plt.show()
