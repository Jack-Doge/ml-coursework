import pandas as pd
import numpy as np
import statsmodels.api as sm
from config import Config
from itertools import repeat

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
    # print(f'absolute LOO errors: {abs_LOO_errors}')
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
    y_hat_test = X_test @ beta_hat
    RMSE_test = np.sqrt(np.mean((y_test - y_hat_test) ** 2))
    print(f'RMSE_test: {RMSE_test}')
    return RMSE_test

def remove_observation_with_high_hii(h_ii: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.DataFrame)-> tuple:
    P = 11
    N = 612
    indices = np.where(h_ii > 2 * P / N)
    X_train_filt = np.delete(X_train, indices, axis = 0)
    y_train_filt = np.delete(y_train, indices, axis = 0)
    return X_train_filt, y_train_filt

def repeat_exercsies_1_3(X_train_filt):
    beta_hat_filt, y_hat_filt   = regress_model(X_train_filt, y_train_filt)
    RMSE_train_filt             = calculate_RMSE(y_train_filt, y_hat_filt)
    H_filt, h_ii_filt           = calc_hat_matrix()
    abs_LOO_errors_filt         = error_of_leave_one_out_prediction(h_ii_filt)
    influential_observations_report(h_ii_filt, abs_LOO_errors_filt)
    calc_out_sample_RMSE(X_test, beta_hat_filt, y_test)

def remove_three_observation_with_largest_loo_error(h_ii: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.DataFrame)-> tuple:
    indices = np.argsort(abs_LOO_errors, axis = 0)[-3:]
    X_train_filt = np.delete(X_train, indices, axis = 0)
    y_train_filt = np.delete(y_train, indices, axis = 0)
    return X_train_filt, y_train_filt

def repeat_exercises_1_3_again(X_train_flit):
    beta_hat_filt, y_hat_filt   = regress_model(X_train_filt, y_train_filt)
    RMSE_train_filt             = calculate_RMSE(y_train_filt, y_hat_filt)
    H_filt, h_ii_filt           = calc_hat_matrix()
    abs_LOO_errors_filt         = error_of_leave_one_out_prediction(h_ii_filt)
    influential_observations_report(h_ii_filt, abs_LOO_errors_filt)
    calc_out_sample_RMSE(X_test, beta_hat_filt, y_test)


if __name__ == '__main__':


    """ HW1 """
    df = load_data('data_with_complete_dates.csv')
    X_train, y_train, X_test, y_test = split_train_test(df)
    print(X_train.shape)
    
    # Exercise 1
    beta_hat, y_hat = regress_model(X_train, y_train)
    RMSE_train      = calculate_RMSE(y_train, y_hat)
    H, h_ii         = calc_hat_matrix()
    abs_LOO_errors  = error_of_leave_one_out_prediction(h_ii)

    # Exercise 2
    influential_observations_report(h_ii, abs_LOO_errors)

    # Exercise 3
    calc_out_sample_RMSE(X_test, beta_hat, y_test)

    # Exercise 4
    X_train_filt, y_train_filt = remove_observation_with_high_hii(h_ii, X_train, y_train)
    repeat_exercsies_1_3(X_train_filt)

    # Exercise 5
    X_train_filt, y_train_filt = remove_three_observation_with_largest_loo_error(h_ii, X_train, y_train)
    repeat_exercises_1_3_again(X_train_filt)
