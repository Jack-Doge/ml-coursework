import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from HW1 import load_data, split_train_test

def normalize(X: np.ndarray)-> np.ndarray:
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm

def find_best_k_pcr(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, use_test_set: bool) -> int:
    rmse_list = []
    for k in range(1, X_train.shape[1] + 1):
        pca = PCA(n_components=k)
        X_train_pca = pca.fit_transform(X_train)
        
        model = LinearRegression()
        model.fit(X_train_pca, y_train)

        if use_test_set:
            X_test_pca = pca.transform(X_test)
            y_pred = model.predict(X_test_pca)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        else:
            y_pred = model.predict(X_train_pca)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        
        rmse_list.append(rmse)
    
    best_k = np.argmin(rmse_list) + 1
    return best_k

if __name__ == '__main__':
    df = load_data('data_with_complete_dates.csv')
    X_train, y_train, X_test, y_test = split_train_test(df)

    X_train_scaled = normalize(X_train)
    X_test_scaled = normalize(X_test)

    best_k_train = find_best_k_pcr(X_train_scaled, y_train, X_test_scaled, y_test, use_test_set=False)
    print(f'Best k for S_train: {best_k_train}')

    best_k_test = find_best_k_pcr(X_train_scaled, y_train, X_test_scaled, y_test, use_test_set=True)
    print(f'Best k for S_test: {best_k_test}')