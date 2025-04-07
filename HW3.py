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
    
    train_df = train_df.pivot(index = 'date', columns = 'code', values = Config.TARGET)
    test_df = test_df.pivot(index = 'date', columns = 'code', values = Config.TARGET)
    train_df.columns = [f"{col}_{code}" for code, col in train_df.columns]
    test_df.columns = [f"{col}_{code}" for code, col in test_df.columns]
    train_df.reset_index(inplace=True)
    test_df.reset_index(inplace=True)

    # transform the data to ndarray
    train_return = train_df.iloc[:, 2:].to_numpy()
    test_return = test_df.iloc[:, 2:].to_numpy()
    return train_return, test_return

def apt_analysis(return_matrix: np.ndarray, K: int, eigenvectors: None) -> tuple[np.ndarray, np.ndarray]:
    """
    return matrix: T * N ndarray
    K: number of latent factors
    T: number of time periods
    N: number of assets
    eigenvectors: optional, for applying on test data
    """
    if eigenvectors is None:
        pca = PCA(n_components = K)
        factors = pca.fit_transform(return_matrix)
        eigenvectors = pca.components_.T
        print(f'shape of return matrix: {return_matrix.shape}') # T * N
        print(f'shape of factors: {factors.shape}') # T * K
        print(f'shape of eigenvectors: {eigenvectors.shape}') # N * K
    else:
        factors = return_matrix @ eigenvectors


    # calculate the mean of the factors
    mu_f = np.mean(factors, axis = 0)
    print(f'shape of mu_f: {mu_f.shape}') # K * 1
    # calculate the covariance matrix
    sigma_f = np.cov(factors.T)
    print(f'shape of sigma_f: {sigma_f.shape}') # K * K


    if K == 1:
        sigma_scalar = sigma_f.item()  # 從 (1,1) 轉成純 scalar
        w_opt = 1.0  # 單一因子，權重只能是 1（normalize 不變）
        Rp_t = factors.flatten()  # 因為 w_opt = 1
        mean_return = mu_f.item()
        volatility = np.sqrt(sigma_scalar)
        mean_to_volatility = mean_return / volatility
    else:
        inv_Sigma_f = np.linalg.inv(sigma_f)
        w_opt = inv_Sigma_f @ mu_f / (np.ones(K) @ inv_Sigma_f @ mu_f)
        Rp_t = factors @ w_opt
        mean_return = np.mean(Rp_t)
        volatility = np.std(Rp_t)
        mean_to_volatility = mean_return / volatility

    """
    # calculate the optimal portfolio returns
    inv_sigma_f = np.linalg.inv(sigma_f)

    w_opt = (inv_sigma_f @ mu_f) / (np.ones(K) @ inv_sigma_f @ mu_f)

    Rp_t = factors @ w_opt

    # calculate the mean to volatility ratio
    mean_to_volatility = np.mean(Rp_t) / np.std(Rp_t)
    """

    return {
        'factors'            : factors,
        'eigenvectors'       : eigenvectors,
        'mu_f'               : mu_f,
        'sigma_f'            : sigma_f,
        'w_opt'              : w_opt,
        'Rp_t'               : Rp_t,
        'mean_to_volatility' : mean_to_volatility
    }




if __name__ == '__main__':


    train_df, test_df = train_test_split()
    train_return, test_return = wide_format_transform(train_df, test_df)
    
    # solution of problem 1
    K = 8
    result = apt_analysis(train_return, K, None)
    print(f'solution of problem 1')
    print(result)

    # solution of problem 2
    result_test = apt_analysis(test_return, K, result['eigenvectors'])
    print(f'solution of problem 2')
    print(result_test)


    # solution of problem 3
    result = []
    for K in range(1, 9):
        print(f'K = {K}')
        train_result = apt_analysis(train_return, K, None)
        test_result = apt_analysis(test_return, K, train_result['eigenvectors'])
        result.append({
            'K'                 : K,
            'mean_to_volatility': test_result['mean_to_volatility'],
            'Rp_t'              : test_result['Rp_t']
        })
    result = pd.DataFrame(result)
    print(f'solution of problem 3')
    print(result)

