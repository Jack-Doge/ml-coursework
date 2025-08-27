import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

FEATURE = ['turnover', 'pb', 'ps', 'dividend_yield', 'cash_dividend_yield', 
              'market_risk_premium', 'size_premium', 'value_premium', 
              'momentum_factor', 'risk_free_rate', 'investment_factor', 
              'profitability_factor']
TARGET = 'ret'

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, index_col=0)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m')
    df = df.sort_values(by=['date', 'stock_id']).reset_index(drop=True)
    return df

def split_data(df: pd.DataFrame, target_date: str) -> dict:
    target_date = pd.to_datetime(target_date)
    train_full = df[df['date'] < target_date].copy()
    sorted_dates = sorted(train_full['date'].unique())
    split_index = int(len(sorted_dates) * (2 / 3))
    train_dates = sorted_dates[:split_index]
    val_dates = sorted_dates[split_index:]
    train_data = train_full[train_full['date'].isin(train_dates)].copy()
    val_data = train_full[train_full['date'].isin(val_dates)].copy()
    test_data = df[df['date'] == target_date].copy()
    test_data['true_ret'] = df[df['date'] == target_date + pd.DateOffset(months=1)][TARGET].values
    return {
        'train_data': train_data,
        'val_data'  : val_data,
        'test_data' : test_data
    }

def prepare_data(train_data : pd.DataFrame, 
                 val_data   : pd.DataFrame, 
                 test_data  : pd.DataFrame) -> dict:
    X_train = train_data[FEATURE].values
    y_train = train_data[TARGET].values
    X_val = val_data[FEATURE].values
    y_val = val_data[TARGET].values
    X_test = test_data[FEATURE].values
    return {
        'X_train': X_train, 
        'y_train': y_train, 
        'X_val': X_val, 
        'y_val': y_val, 
        'X_test': X_test
    }

def xgboost_tuning(X_train      : np.ndarray, 
                   y_train      : np.ndarray, 
                   X_val        : np.ndarray, 
                   y_val        : np.ndarray, 
                   X_test       : np.ndarray, 
                   train_data   : pd.DataFrame, 
                   test_data    : pd.DataFrame, 
                   val_data     : pd.DataFrame
                   ) -> tuple:
    depths = [2, 3, 4, 5, 6]
    val_errors = []
    for d in depths:
        model = xgb.XGBRegressor(max_depth=d, n_estimators=1000, learning_rate=0.1, 
                                 subsample=0.8, colsample_bytree=0.8, random_state=42)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_val_pred)
        val_errors.append(mse)
    best_depth = depths[np.argmin(val_errors)]
    logger.info(f"Best max_depth: {best_depth}")
    X_all = np.vstack((X_train, X_val))
    y_all = np.concatenate((y_train, y_val))
    final_model = xgb.XGBRegressor(max_depth=best_depth, n_estimators=1000, learning_rate=0.1, 
                                   subsample=0.8, colsample_bytree=0.8, random_state=42)
    final_model.fit(X_all, y_all)
    y_test_pred = final_model.predict(test_data[FEATURE].values)
    test_data = test_data.copy()
    test_data['xgb_predicted_return'] = y_test_pred
    return final_model, best_depth, test_data

def form_portfolios(test_data: pd.DataFrame, pred_col: str = 'xgb_predicted_return') -> dict:
    test_data = test_data.sort_values(by=pred_col, ascending=False).reset_index(drop=True).copy()
    test_data['group'] = pd.qcut(test_data[pred_col], q=4, labels=['D1', 'D2', 'D3', 'D4'])
    group_returns = test_data.groupby('group')['true_ret'].mean()
    rls = group_returns['D1'] - group_returns['D4']
    rsimple = test_data['true_ret'].mean()
    return {
        'group_returns' : group_returns, 
        'rls'           : rls, 
        'rsimple'   : rsimple
    }

def xgboost_pipeline(file_path: str, target_date: str) -> tuple:
    df       = load_data(file_path)
    splits   = split_data(df, target_date)
    prepared = prepare_data(splits['train_data'], splits['val_data'], splits['test_data'])
    model, best_depth, test_data = xgboost_tuning(
        X_train=prepared['X_train'], 
        y_train=prepared['y_train'], 
        X_val=prepared['X_val'], 
        y_val=prepared['y_val'], 
        X_test=prepared['X_test'],
        train_data=splits['train_data'], 
        val_data=splits['val_data'], 
        test_data=splits['test_data'] 
    )
    portfolio_returns = form_portfolios(test_data, pred_col='xgb_predicted_return')
    return {
        'model': model, 
        'best_depth': best_depth, 
        'test_data': test_data, 
        'portfolio returns': portfolio_returns
    }

if __name__ == '__main__':
    file_path = 'data/processed_data.csv'
    df = load_data(file_path)
    all_dates = sorted(df['date'].unique())
    rolling_dates = all_dates[66: -1]
    results = []
    for date in rolling_dates:
        print(f'processing date: {date}')
        date_str = date.strftime('%Y-%m-%d')
        result = xgboost_pipeline(file_path, date_str)
        group_returns = result['portfolio returns']['group_returns']
        results.append({
            'date': date_str, 
            'RD1': group_returns['D1'], 
            'RD2': group_returns['D2'], 
            'RD3': group_returns['D3'], 
            'RD4': group_returns['D4'], 
            'RLS': result['portfolio returns']['rls'], 
            'Rsimple': result['portfolio returns']['rsimple'], 
            'best_depth': result['best_depth']
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/xgboost_results.csv', index=False)
