import pandas as pd
import numpy as np
import statsmodels.api as sm
from config import Config

def load_data(path: str)-> pd.DataFrame:
    df = pd.read_csv(f'data/{path}')
    return df

def split_train_test(df: pd.DataFrame)-> tuple:
    X_train = df[df['date'] == '2024-11'][Config.PREDICTOR] .values
    y_train = df[df['date'] == '2024-12'][Config.TARGET]    .values
    X_test  = df[df['date'] == '2024-12'][Config.PREDICTOR] .values
    y_test  = df[df['date'] == '2025-01'][Config.TARGET]    .values
    X_train = sm.add_constant(X_train, has_constant = 'add')
    X_test  = sm.add_constant(X_test,  has_constant = 'add')
    return (X_train, y_train, X_test, y_test)

def regress_model(X_train: pd.DataFrame, y_train: pd.DataFrame)-> tuple:
    beta_hat = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    y_hat = X_train @ beta_hat
    return (beta_hat, y_hat)

def calculate_RMSE(y_train: pd.DataFrame, y_hat: pd.DataFrame)-> float:
    RMSE_train = np.sqrt(np.mean((y_train - y_hat) ** 2))
    print(f'RMSE_train: {RMSE_train}')
    return RMSE_train

def calc_hat_matrix(X_train: np.ndarray):
    H = X_train @ np.linalg.inv(X_train.T @ X_train) @ X_train.T
    h_ii = np.diag(H)
    return H, h_ii

def error_of_leave_one_out_prediction(y_train: np.ndarray, y_hat: np.ndarray, h_ii: np.ndarray):
    errors  = y_train - y_hat
    h_ii    = h_ii.reshape(-1, 1)
    LOO_errors = errors / (1 - h_ii)
    abs_LOO_errors = np.abs(LOO_errors)
    return abs_LOO_errors
 
def influential_observations_report(h_ii: np.ndarray, abs_LOO_errors: np.ndarray)-> pd.DataFrame:
    hii_stats = {
        "min"   : np.min(h_ii),
        "1%"    : np.percentile(h_ii, 1),
        "25%"   : np.percentile(h_ii, 25),
        "median": np.median(h_ii),
        "75%"   : np.percentile(h_ii, 75),
        "99%"   : np.percentile(h_ii, 99),
        "max"   : np.max(h_ii)
    }
    abs_LOO_errors_stats = {
        "min"   : np.min(abs_LOO_errors),
        "1%"    : np.percentile(abs_LOO_errors, 1),
        "25%"   : np.percentile(abs_LOO_errors, 25),
        "median": np.median(abs_LOO_errors),
        "75%"   : np.percentile(abs_LOO_errors, 75),
        "99%"   : np.percentile(abs_LOO_errors, 99),
        "max"   : np.max(abs_LOO_errors)
    }
    report_df = pd.DataFrame([hii_stats, abs_LOO_errors_stats], index = ['h_ii', 'abs_LOO_errors'])
    print(report_df)
    return report_df

def calc_out_sample_RMSE(X_test: np.ndarray, beta_hat: np.ndarray, y_test: np.ndarray)-> float:
    y_hat_test = X_test @ beta_hat
    RMSE_test = np.sqrt(np.mean((y_test - y_hat_test) ** 2))
    print(f'RMSE_test: {RMSE_test}')
    return RMSE_test

def remove_observation_with_high_hii(h_ii: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.DataFrame)-> tuple:
    P = X_train.shape[1]
    N = X_train.shape[0]
    indices = np.where(h_ii > 2 * P / N)
    X_train_filt = np.delete(X_train, indices, axis = 0)
    y_train_filt = np.delete(y_train, indices, axis = 0)
    return X_train_filt, y_train_filt

def remove_three_observation_with_largest_loo_error(abs_LOO_errors: np.ndarray, X_train: pd.DataFrame, y_train: pd.DataFrame)-> tuple:
    indices = np.argsort(abs_LOO_errors, axis = 0)[-3:]
    X_train_filt = np.delete(X_train, indices, axis = 0)
    y_train_filt = np.delete(y_train, indices, axis = 0)
    return X_train_filt, y_train_filt

def run_regression_analysis(X_train, y_train, X_test, y_test, description):
    print(f"\n--- Running Analysis for: {description} ---")
    beta_hat, y_hat = regress_model(X_train, y_train)
    calculate_RMSE(y_train, y_hat)
    H, h_ii = calc_hat_matrix(X_train)
    abs_LOO_errors = error_of_leave_one_out_prediction(y_train, y_hat, h_ii)
    influential_observations_report(h_ii, abs_LOO_errors)
    calc_out_sample_RMSE(X_test, beta_hat, y_test)
    return h_ii, abs_LOO_errors

if __name__ == '__main__':
    df = load_data('data_with_complete_dates.csv')
    X_train, y_train, X_test, y_test = split_train_test(df)
    
    h_ii, abs_LOO_errors = run_regression_analysis(X_train, y_train, X_test, y_test, "Initial Model")

    X_train_filt_h, y_train_filt_h = remove_observation_with_high_hii(h_ii, X_train, y_train)
    run_regression_analysis(X_train_filt_h, y_train_filt_h, X_test, y_test, "After Removing High h_ii Observations")

    X_train_filt_l, y_train_filt_l = remove_three_observation_with_largest_loo_error(abs_LOO_errors, X_train, y_train)
    run_regression_analysis(X_train_filt_l, y_train_filt_l, X_test, y_test, "After Removing 3 Largest LOO Error Observations")