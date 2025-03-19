import pandas as pd
import numpy as np
import statsmodels.api as sm
from config import Config
from itertools import repeat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from HW1 import *


def normalize(X: np.ndarray)-> np.ndarray:
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm

def compute_rmse_after_pcr(X_train_scaled: np.ndarray, y_train: np.ndarray)-> int:

    rmse_list = []
    for K in repeat(1, 11):
        pca = PCA(n_components = K)
        X_train_pca = pca.fit_transform(X_train_scaled)
        
        model = LinearRegression()
        model.fit(X_train_pca, y_train)

        y_train_pred = model.predict(X_train_pca)
        rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_list.append(rmse)
    
    best_k = np.argmin(rmse_list) + 1
    return best_k

def fit_test_data_and_find_the_best_k(X_train_scaled, y_train, X_test_scaled, y_test):

    rmse_list = []
    for K in repeat(1, 11):
        pca = PCA(n_components = K)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        model = LinearRegression()
        model.fit(X_train_pca, y_train)

        y_test_pred = model.predict(X_test_pca)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        rmse_list.append(rmse)
    
    best_k = np.argmin(rmse_list) + 1
    return best_k


if __name__ == '__main__':


    """ HW2 """
    df = load_data('data_with_complete_dates.csv')
    X_train, y_train, X_test, y_test = split_train_test(df)
    print(X_train.shape)
    
    # Exercise 1
    X_train_scaled = normalize(X_train)
    best_k = compute_rmse_after_pcr(X_train_scaled, y_train)
    print(f'bset k in S_train: {best_k}')
    
    # Exercise 2
    X_test_scaled = normalize(X_test)
    best_k = fit_test_data_and_find_the_best_k(X_train_scaled, y_train, X_test_scaled, y_test)
    print(f'bset k in S_test: {best_k}')

