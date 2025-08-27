import pandas as pd
import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

FEATURE = ['turnover', 'pb', 'ps', 'dividend_yield','cash_dividend_yield', 
           'market_risk_premium', 'size_premium', 'value_premium', 
           'momentum_factor', 'risk_free_rate', 'investment_factor', 'profitability_factor']
TARGET = 'ret'

def load_data(file_path: str)-> pd.DataFrame:
    df = pd.read_csv(file_path, index_col=0)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m')
    df.sort_values(by=['date', 'stock_id'], inplace=True)
    return df

def split_data(df: pd.DataFrame, target_date: str)-> dict:
    train_full = df[df['date'] < pd.to_datetime(target_date)].copy()
    sorted_dates = sorted(train_full['date'].unique())
    split_index = int(len(sorted_dates) * (2 / 3))
    train_dates = sorted_dates[:split_index]
    val_dates   = sorted_dates[split_index:]
    train_data  = train_full[train_full['date'].isin(train_dates)].copy()
    val_data    = train_full[train_full['date'].isin(val_dates)]
    test_data   = df[df['date'] == pd.to_datetime(target_date)].copy()
    test_data['true_ret'] = df[df['date'] == pd.to_datetime(target_date) + pd.DateOffset(months=1)][TARGET].values
    return {
        'train_data': train_data,
        'val_data'  : val_data,
        'test_data' : test_data
    }

def standardize_data(train_data : pd.DataFrame, 
                     val_data   : pd.DataFrame, 
                     test_data  : pd.DataFrame ) -> dict:
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_data[FEATURE])
    y_train = train_data[TARGET].values
    X_val   = scaler.transform(val_data[FEATURE])
    y_val   = val_data[TARGET].values
    X_test  = scaler.transform(test_data[FEATURE])
    return {
        'X_train'   : X_train,
        'y_train'   : y_train,
        'X_val'     : X_val,
        'y_val'     : y_val,
        'X_test'    : X_test,
        'scaler'    : scaler
    }

def ridge_regression_tuning(X_train     : np.ndarray, 
                            y_train     : np.ndarray, 
                            X_val       : np.ndarray, 
                            y_val       : np.ndarray, 
                            X_test      : np.ndarray, 
                            train_data  : pd.DataFrame,
                            test_data   : pd.DataFrame, 
                            val_data    : pd.DataFrame, 
                            scaler      : StandardScaler) -> tuple[Ridge, float]:
    alphas = [0.01, 0.1, 1, 10, 100]
    val_error = []
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_val_pred   = model.predict(X_val)
        mse          = mean_squared_error(y_val, y_val_pred)
        val_error.append(mse)
    best_alpha = alphas[np.argmin(val_error)]
    logger.info(f"Best alpha: {best_alpha}")
    X_all = scaler.transform(pd.concat([train_data, val_data])[FEATURE])
    y_all = pd.concat([train_data, val_data])[TARGET].values
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X_all, y_all)
    y_test_pred = final_model.predict(X_test)
    test_data = test_data.copy()
    test_data['ridge_predicted_ret'] = y_test_pred
    return final_model, best_alpha, test_data

def form_portfolios(test_data: pd.DataFrame) -> dict:
    test_data = test_data.sort_values(by='ridge_predicted_ret', ascending=False).reset_index(drop=True).copy()
    test_data['rank']    = test_data['ridge_predicted_ret'].rank(method='first', ascending=False)
    test_data['group']   = pd.qcut(test_data['rank'], q=4, labels=['D1', 'D2', 'D3', 'D4'])
    test_data.to_csv('data/test_data_with_predictions.csv', index=False)
    group_returns = test_data.groupby('group')['true_ret'].mean()
    ridge_rls = group_returns['D1'] - group_returns['D4']
    ridge_rsimple = test_data['true_ret'].mean()
    logger.info(f'Ridge Long-Short Return: {ridge_rls}, Simple Average Return: {ridge_rsimple}')
    return {
        'group_returns': group_returns,
        'ridge_rls': ridge_rls,
        'ridge_rsimple': ridge_rsimple
    }

def ridge_regression_pipeline(file_path: str, target_date: str) -> dict:
    df = load_data(file_path)
    data_splits = split_data(df, target_date)
    standardized_data = standardize_data(data_splits['train_data'], 
                                         data_splits['val_data'], 
                                         data_splits['test_data'])
    final_model, best_alpha, test_data = ridge_regression_tuning(
        standardized_data['X_train'], 
        standardized_data['y_train'], 
        standardized_data['X_val'], 
        standardized_data['y_val'], 
        standardized_data['X_test'], 
        data_splits['train_data'], 
        data_splits['test_data'], 
        data_splits['val_data'], 
        standardized_data['scaler']
    )
    portfolio_returns = form_portfolios(test_data)
    return {
        'model': final_model,
        'best_alpha': best_alpha,
        'test_data': test_data,
        'portfolio_returns': portfolio_returns
    }

if __name__ == '__main__':
    file_path = 'data/processed_data.csv'
    df = load_data(file_path)
    all_dates = df['date'].unique()
    all_dates = sorted(all_dates)
    all_dates = [date for date in all_dates if date >= pd.Timestamp('2019-09-01') and date < pd.Timestamp('2025-02-01')]
    results = []
    for date in all_dates:
        logger.info(f'Processing date: {date}')
        str_date        = pd.Timestamp(date).strftime('%Y-%m-%d')
        result          = ridge_regression_pipeline(file_path, str_date)
        group_returns   = result['portfolio_returns']['group_returns']
        results.append({
            'date'   : str_date, 
            'RD1'    : group_returns['D1'], 
            'RD2'    : group_returns['D2'],
            'RD3'    : group_returns['D3'],
            'RD4'    : group_returns['D4'],
            'RLS'    : result['portfolio_returns']['ridge_rls'],
            'Rsimple': result['portfolio_returns']['ridge_rsimple'], 
            'alpha'  : result['best_alpha']
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/ridge_regression_results.csv', index=False)