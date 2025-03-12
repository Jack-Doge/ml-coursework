import pandas as pd
import numpy as np
from loguru import logger
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

    return (X_train, y_train, X_test, y_test)

def regress_model(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame)-> tuple:
    X_train = sm.add_constant(X_train, has_constant = 'add') # (612, 11)
    beta_hat = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train # (11, 1)   
    y_hat = X_train @ beta_hat # (612, 1)

    return (beta_hat, y_hat)

def calculate_RMSE(y_train: pd.DataFrame, y_hat: pd.DataFrame)-> float:
    RMSE_train = np.sqrt(np.mean((y_train - y_hat) ** 2))
    print(f'RMSE_train: {RMSE_train}')
    return RMSE_train

def calc_hat_matrix():
    H = X_train @ np.linalg.inv(X_train.T @ X_train) @ X_train.T # (612, 612)
    h_ii = np.diag(H) # (612, ) diagnal entries of H
    return H, h_ii

def error_of_leave_one_out_prediction(h_ii: np.ndarray):
    errors  = y_train - y_hat    # (612, 1)
    h_ii    = h_ii.reshape(-1, 1)  # (612, 1)
    LOO_errors = errors / (h_ii / (1 - h_ii)) # (612, 1)
    abs_LOO_errors = np.abs(LOO_errors) # (612, 1)
    print(f'absolute LOO errors: {abs_LOO_errors}')
    return abs_LOO_errors
 
def influential_observations_report(h_ii: np.ndarray, abs_LOO_errors: np.ndarray)-> pd.DataFrame:
    # print(h_ii.shape) # (612, 1)
    # print(abs_LOO_errors.shape) # (612, )
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
    # print(report_df)
    return report_df

def calc_out_sample_RMSE(X_test: np.ndarray, beta_hat: np.ndarray, y_test: np.ndarray)-> float:
    X_test = sm.add_constant(X_test, has_constant = 'add')
    y_hat_test = X_test @ beta_hat
    RMSE_test = np.sqrt(np.mean((y_test - y_hat_test) ** 2))
    print(f'RMSE_test: {RMSE_test}')
    return RMSE_test


if __name__ == '__main__':

    df = load_data('data_with_complete_dates.csv')
    X_train, y_train, X_test, y_test = split_train_test(df)
    
    # Exercise 1
    beta_hat, y_hat = regress_model(X_train, y_train, X_test, y_test)
    RMSE_train      = calculate_RMSE(y_train, y_hat)
    H, h_ii         = calc_hat_matrix()
    abs_LOO_errors  = error_of_leave_one_out_prediction(h_ii)

    # Exercise 2
    influential_observations_report(h_ii, abs_LOO_errors)

    # Exercise 3
    calc_out_sample_RMSE(X_test, beta_hat, y_test)

    # Exercise 4













































































