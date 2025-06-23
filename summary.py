import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ridge_summary():
    df = pd.read_csv('data/ridge_regression_results.csv')
    df['RD1_CUM'] = (1 + df['RD1']).cumprod() - 1
    df['RD2_CUM'] = (1 + df['RD2']).cumprod() - 1
    df['RD3_CUM'] = (1 + df['RD3']).cumprod() - 1
    df['RD4_CUM'] = (1 + df['RD4']).cumprod() - 1
    df['RLS_CUM'] = (1 + df['RLS']).cumprod() - 1
    df['Rsimple_CUM'] = (1 + df['Rsimple']).cumprod() - 1

    # visulize the results
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['RD1_CUM'], label='RD1 Cumulative Return', color='blue', marker='o')
    plt.plot(df['date'], df['RD2_CUM'], label='RD2 Cumulative Return', color='orange', marker='o')
    plt.plot(df['date'], df['RD3_CUM'], label='RD3 Cumulative Return', color='red', marker='o')
    plt.plot(df['date'], df['RD4_CUM'], label='RD4 Cumulative Return', color='green', marker='o')
    plt.plot(df['date'], df['RLS_CUM'], label='RLS Cumulative Return', color='purple', marker='o')
    plt.plot(df['date'], df['Rsimple_CUM'], label='Rsimple Cumulative Return', color='brown', marker='o')
    plt.title('Cumulative Returns of Ridge Regression Portfolios')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.xticks(df['date'][::10])
    plt.legend()
    plt.tight_layout()
    plt.savefig('summary/ridge_cumulative_returns.png')
    plt.show()

def xg_summary():
    df = pd.read_csv('data/xgboost_results.csv')
    df['RD1_CUM'] = (1 + df['RD1']).cumprod() - 1
    df['RD2_CUM'] = (1 + df['RD2']).cumprod() - 1
    df['RD3_CUM'] = (1 + df['RD3']).cumprod() - 1
    df['RD4_CUM'] = (1 + df['RD4']).cumprod() - 1
    df['RLS_CUM'] = (1 + df['RLS']).cumprod() - 1
    df['Rsimple_CUM'] = (1 + df['Rsimple']).cumprod() - 1
    # visualize the results
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['RD1_CUM'], label='RD1 Cumulative Return', color='blue', marker='o')
    plt.plot(df['date'], df['RD2_CUM'], label='RD2 Cumulative Return', color='orange', marker='o')
    plt.plot(df['date'], df['RD3_CUM'], label='RD3 Cumulative Return', color='red', marker='o')
    plt.plot(df['date'], df['RD4_CUM'], label='RD4 Cumulative Return', color='green', marker='o')
    plt.plot(df['date'], df['RLS_CUM'], label='RLS Cumulative Return', color='purple', marker='o')
    plt.plot(df['date'], df['Rsimple_CUM'], label='Rsimple Cumulative Return', color='brown', marker='o')
    plt.title('Cumulative Returns of XGBoost Portfolios')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.xticks(df['date'][::10])
    plt.legend()
    plt.tight_layout()
    plt.savefig('summary/xg_cumulative_returns.png')
    plt.show()

def Q1() -> dict:

    ridge_df = pd.read_csv('data/ridge_regression_results.csv')
    xg_df = pd.read_csv('data/xgboost_results.csv')

    print(ridge_df.head())
    ridge_summary = {
        'mean' : [ridge_df['RD1'].mean(), ridge_df['RD2'].mean(), ridge_df['RD3'].mean(), ridge_df['RD4'].mean(), ridge_df['RLS'].mean(), ridge_df['Rsimple'].mean()],
        'std' : [ridge_df['RD1'].std(), ridge_df['RD2'].std(), ridge_df['RD3'].std(), ridge_df['RD4'].std(), ridge_df['RLS'].std(), ridge_df['Rsimple'].std()],
        'mean-vol': [ridge_df['RD1'].mean() / ridge_df['RD1'].std(),
                    ridge_df['RD2'].mean() / ridge_df['RD2'].std(),
                    ridge_df['RD3'].mean() / ridge_df['RD3'].std(),
                    ridge_df['RD4'].mean() / ridge_df['RD4'].std(), 
                    ridge_df['RLS'].mean() / ridge_df['RLS'].std(),
                    ridge_df['Rsimple'].mean() / ridge_df['Rsimple'].std()
                    ],
    }   
    ridge_summary_df = pd.DataFrame(ridge_summary, index=['RD1', 'RD2', 'RD3', 'RD4', 'RLS', 'Rsimple'])
    print(ridge_summary_df)
    xg_summary_df = pd.DataFrame({
        'mean': [xg_df['RD1'].mean(), xg_df['RD2'].mean(), xg_df['RD3'].mean(), xg_df['RD4'].mean(), xg_df['RLS'].mean(), xg_df['Rsimple'].mean()],
        'std': [xg_df['RD1'].std(), xg_df['RD2'].std(), xg_df['RD3'].std(), xg_df['RD4'].std(), xg_df['RLS'].std(), xg_df['Rsimple'].std()],
        'mean-vol': [xg_df['RD1'].mean() / xg_df['RD1'].std(),
                     xg_df['RD2'].mean() / xg_df['RD2'].std(),
                     xg_df['RD3'].mean() / xg_df['RD3'].std(),
                     xg_df['RD4'].mean() / xg_df['RD4'].std(), 
                     xg_df['RLS'].mean() / xg_df['RLS'].std(),
                     xg_df['Rsimple'].mean() / xg_df['Rsimple'].std()
                     ]
    }, index=['RD1', 'RD2', 'RD3', 'RD4', 'RLS', 'Rsimple'])
    print(xg_summary_df)

    fig, ax = plt.subplots(4, 1, figsize=(16, 16))  # 四行一列

    ridge_summary_df['mean'].plot(kind='bar', ax=ax[0], title='Ridge Regression Mean Returns', rot=0)
    ridge_summary_df['mean-vol'].plot(kind='bar', ax=ax[1], title='Ridge Regression Mean-Vol', rot=0)
    xg_summary_df['mean'].plot(kind='bar', ax=ax[2], title='XGBoost Mean Returns', rot=0)
    xg_summary_df['mean-vol'].plot(kind='bar', ax=ax[3], title='XGBoost Mean-Vol', rot=0)

    plt.tight_layout()
    plt.savefig('summary/Q1.png')
    plt.show()

# Q1()

def Q2():
    ridge_df = pd.read_csv('data/ridge_regression_results.csv')
    xg_df = pd.read_csv('data/xgboost_results.csv')

    mean_df = pd.DataFrame({
        'Ridge Regression': [ridge_df['RD1'].mean(), ridge_df['RD2'].mean(), ridge_df['RD3'].mean(), ridge_df['RD4'].mean(), ridge_df['RLS'].mean(), ridge_df['Rsimple'].mean()],
        'XGBoost': [xg_df['RD1'].mean(), xg_df['RD2'].mean(), xg_df['RD3'].mean(), xg_df['RD4'].mean(), xg_df['RLS'].mean(), xg_df['Rsimple'].mean()]
    }, index=['RD1', 'RD2', 'RD3', 'RD4', 'RLS', 'Rsimple'])
    std_df = pd.DataFrame({
        'Ridge Regression': [ridge_df['RD1'].std(), ridge_df['RD2'].std(), ridge_df['RD3'].std(), ridge_df['RD4'].std(), ridge_df['RLS'].std(), ridge_df['Rsimple'].std()],
        'XGBoost': [xg_df['RD1'].std(), xg_df['RD2'].std(), xg_df['RD3'].std(), xg_df['RD4'].std(), xg_df['RLS'].std(), xg_df['Rsimple'].std()]
    }, index=['RD1', 'RD2', 'RD3', 'RD4', 'RLS', 'Rsimple'])
    mean_vol_df = pd.DataFrame({
        'Ridge Regression': [ridge_df['RD1'].mean() / ridge_df['RD1'].std(),
                             ridge_df['RD2'].mean() / ridge_df['RD2'].std(),
                             ridge_df['RD3'].mean() / ridge_df['RD3'].std(),
                             ridge_df['RD4'].mean() / ridge_df['RD4'].std(), 
                             ridge_df['RLS'].mean() / ridge_df['RLS'].std(),
                             ridge_df['Rsimple'].mean() / ridge_df['Rsimple'].std()],
        'XGBoost': [xg_df['RD1'].mean() / xg_df['RD1'].std(),
                    xg_df['RD2'].mean() / xg_df['RD2'].std(),
                    xg_df['RD3'].mean() / xg_df['RD3'].std(),
                    xg_df['RD4'].mean() / xg_df['RD4'].std(), 
                    xg_df['RLS'].mean() / xg_df['RLS'].std(),
                    xg_df['Rsimple'].mean() / xg_df['Rsimple'].std()]
    }, index=['RD1', 'RD2', 'RD3', 'RD4', 'RLS', 'Rsimple'])
    """
    Compute the long-short portfolio returns RLS
    t+1 = RD1
    t+1 −RD4
    t+1. Does the RLS
    t+1 of non
    linear model have greater mean and mean-to-vol ratio than those of linear model?
    """

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))  # Two rows, one column
    mean_df.plot(kind='bar', ax=ax[0], title='Mean Returns Comparison', rot=0)
    mean_vol_df.plot(kind='bar', ax=ax[1], title='Mean-Vol Comparison', rot=0)    
    plt.tight_layout()
    plt.savefig('summary/Q2.png')
    plt.show()
# Q2()

def Q3():
    ridge_df = pd.read_csv('data/ridge_regression_results.csv')
    xg_df = pd.read_csv('data/xgboost_results.csv')

    """
    Do the long-short portfolio (RLS
    t+1) of both models beat the simple average portfolio
    returns (Rsimple
    t+1 = 1
    N 
    N
    i=1ri,t+1)?
    """
    ridge_summary_df = pd.DataFrame({
        'RLS': ridge_df['RLS'].mean(),
        'Rsimple': ridge_df['Rsimple'].mean(),
        'RLS_std': ridge_df['RLS'].std(),
        'Rsimple_std': ridge_df['Rsimple'].std(),
        'RLS_mean_vol': ridge_df['RLS'].mean() / ridge_df['RLS'].std(),
        'Rsimple_mean_vol': ridge_df['Rsimple'].mean() / ridge_df['Rsimple'].std()
    }, index=['ridge'])
    xg_summary_df = pd.DataFrame({
        'RLS': xg_df['RLS'].mean(),
        'Rsimple': xg_df['Rsimple'].mean(),
        'RLS_std': xg_df['RLS'].std(),
        'Rsimple_std': xg_df['Rsimple'].std(),
        'RLS_mean_vol': xg_df['RLS'].mean() / xg_df['RLS'].std(),
        'Rsimple_mean_vol': xg_df['Rsimple'].mean() / xg_df['Rsimple'].std()
    }, index=['xgboost'])

    summary_df = pd.concat([ridge_summary_df, xg_summary_df], axis=0)
    print(summary_df)
    fig, ax = plt.subplots(2, 1, figsize=(8, 10))  # Two rows, one column
    summary_df[['RLS', 'Rsimple']].plot(kind='bar', ax=ax[0], title='RLS vs Rsimple Mean Returns', rot=0)
    summary_df[['RLS_mean_vol', 'Rsimple_mean_vol']].plot(kind='bar', ax=ax[1], title='RLS vs Rsimple Mean-Vol Ratio', rot=0)
    ax[0].set_ylabel('Mean Returns')
    ax[1].set_ylabel('Mean-Vol Ratio')
    ax[0].legend(['RLS', 'Rsimple'])
    ax[1].legend(['RLS Mean-Vol', 'Rsimple Mean-Vol'])
    plt.suptitle('Comparison of Long-Short Portfolio and Simple Average Portfolio Returns')
    plt.xticks(rotation=0)
    plt.xlabel('Model')
    plt.ylabel('Returns / Mean-Vol Ratio')
    plt.title('RLS vs Rsimple Comparison')
    plt.tight_layout()
    plt.savefig('summary/Q3.png')
    plt.show()
# Q3()



def Q4():
    ridge_df = pd.read_csv('data/ridge_regression_results.csv')
    xgb_df = pd.read_csv('data/xgboost_results.csv')
    ridge_df['date'] = pd.to_datetime(ridge_df['date'])
    xgb_df['date'] = pd.to_datetime(xgb_df['date'])

    ridge_tmax = ridge_df.loc[ridge_df['RLS'].idxmax()]
    ridge_tmin = ridge_df.loc[ridge_df['RLS'].idxmin()]
    
    xgb_tmax = xgb_df.loc[xgb_df['RLS'].idxmax()]
    xgb_tmin = xgb_df.loc[xgb_df['RLS'].idxmin()]

    print(f'Ridge Regression RLS Max: {ridge_tmax["date"]} with return {ridge_tmax["RLS"]} and alpha {ridge_tmax["alpha"]}')
    print(f'Ridge Regression RLS Min: {ridge_tmin["date"]} with return {ridge_tmin["RLS"]} and alpha {ridge_tmin["alpha"]}')
    print(f'XGBoost RLS Max: {xgb_tmax["date"]} with return {xgb_tmax["RLS"]} and depth {xgb_tmax["best_depth"]}')
    print(f'XGBoost RLS Min: {xgb_tmin["date"]} with return {xgb_tmin["RLS"]} and depth {xgb_tmin["best_depth"]}')

    tmax_date = ridge_tmax['date']
    print(tmax_date)

Q4()