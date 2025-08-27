from config import Config
from HW1 import load_data
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def train_test_split()-> list[pd.DataFrame]:    
    df = load_data('data_with_complete_dates.csv')
    df['date'] = pd.to_datetime(df['date'])
    train_df = df[(df['date'] >= '2015-01-01') & (df['date'] <= '2019-12-31')][['date', 'code'] + Config.TARGET]
    test_df  = df[(df['date'] >= '2020-01-01') & (df['date'] <= '2021-12-31')][['date', 'code'] + Config.TARGET]
    return train_df, test_df

def wide_format_transform(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.pivot(index = 'date', columns = 'code', values = Config.TARGET[0])
    test_df = test_df.pivot(index = 'date', columns = 'code', values = Config.TARGET[0])
    train_return = train_df.to_numpy()
    test_return = test_df.to_numpy()
    return train_return, test_return

def apt_analysis(return_matrix: np.ndarray, K: int, eigenvectors: np.ndarray = None) -> dict:
    if eigenvectors is None:
        pca = PCA(n_components=K)
        factors = pca.fit_transform(return_matrix)
        eigenvectors = pca.components_.T
    else:
        factors = return_matrix @ eigenvectors

    mu_f = np.mean(factors, axis=0)
    sigma_f = np.cov(factors.T)

    if K == 1:
        w_opt = np.array([1.0])
    else:
        inv_Sigma_f = np.linalg.inv(sigma_f)
        w_opt = inv_Sigma_f @ mu_f / (np.ones(K) @ inv_Sigma_f @ mu_f)

    Rp_t = factors @ w_opt
    mean_return = np.mean(Rp_t)
    volatility = np.std(Rp_t)
    mean_to_volatility = mean_return / volatility if volatility != 0 else 0

    return {
        'factors': factors,
        'eigenvectors': eigenvectors,
        'mu_f': mu_f,
        'sigma_f': sigma_f,
        'w_opt': w_opt,
        'Rp_t': Rp_t,
        'mean_to_volatility': mean_to_volatility
    }

if __name__ == '__main__':
    train_df, test_df = train_test_split()
    train_return, test_return = wide_format_transform(train_df, test_df)

    print("--- Solution for Problem 1 (K=8) ---")
    k_problem1 = 8
    result_p1 = apt_analysis(train_return, k_problem1)
    print(f"Mean-to-volatility: {result_p1['mean_to_volatility']:.4f}")

    print("\n--- Solution for Problem 2 (K=8 on test data) ---")
    result_p2 = apt_analysis(test_return, k_problem1, result_p1['eigenvectors'])
    print(f"Mean-to-volatility: {result_p2['mean_to_volatility']:.4f}")

    print("\n--- Solution for Problem 3 (K=1 to 8) ---")
    results_p3 = []
    for k in range(1, 9):
        train_result = apt_analysis(train_return, k)
        test_result = apt_analysis(test_return, k, train_result['eigenvectors'])
        results_p3.append({
            'K': k,
            'mean_to_volatility': test_result['mean_to_volatility']
        })
    
    results_df_p3 = pd.DataFrame(results_p3)
    print(results_df_p3)