import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_cumulative_returns(df: pd.DataFrame, model_name: str):
    for col in ["RD1", "RD2", "RD3", "RD4", "RLS", "Rsimple"]:
        df[f"{col}_CUM"] = (1 + df[col]).cumprod() - 1

    plt.figure(figsize=(12, 6))
    for col in ["RD1", "RD2", "RD3", "RD4", "RLS", "Rsimple"]:
        plt.plot(df["date"], df[f"{col}_CUM"], label=f"{col} Cumulative Return", marker='o')
    
    plt.title(f"Cumulative Returns of {model_name} Portfolios")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.xticks(df['date'][::10])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'summary/{model_name.lower()}_cumulative_returns.png')
    plt.close()

def calculate_summary_stats(df: pd.DataFrame):
    stats = {
        'mean': df.mean(),
        'std': df.std(),
        'mean-vol': df.mean() / df.std()
    }
    return pd.DataFrame(stats).T

def analyze_q1(ridge_df, xg_df):
    ridge_summary = calculate_summary_stats(ridge_df[["RD1", "RD2", "RD3", "RD4", "RLS", "Rsimple"]])
    xg_summary = calculate_summary_stats(xg_df[["RD1", "RD2", "RD3", "RD4", "RLS", "Rsimple"]])

    print("--- Q1: Ridge Summary ---")
    print(ridge_summary)
    print("\n--- Q1: XGBoost Summary ---")
    print(xg_summary)

    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    ridge_summary.loc['mean'].plot(kind='bar', ax=ax[0, 0], title='Ridge Mean Returns', rot=0)
    ridge_summary.loc['mean-vol'].plot(kind='bar', ax=ax[0, 1], title='Ridge Mean-to-Vol Ratio', rot=0)
    xg_summary.loc['mean'].plot(kind='bar', ax=ax[1, 0], title='XGBoost Mean Returns', rot=0)
    xg_summary.loc['mean-vol'].plot(kind='bar', ax=ax[1, 1], title='XGBoost Mean-to-Vol Ratio', rot=0)
    plt.tight_layout()
    plt.savefig('summary/Q1_summary_plots.png')
    plt.close()

def analyze_q2(ridge_df, xg_df):
    ridge_rls_mean = ridge_df['RLS'].mean()
    ridge_rls_mean_vol = ridge_df['RLS'].mean() / ridge_df['RLS'].std()
    xg_rls_mean = xg_df['RLS'].mean()
    xg_rls_mean_vol = xg_df['RLS'].mean() / xg_df['RLS'].std()

    print("--- Q2: RLS Comparison ---")
    print(f"Ridge RLS Mean: {ridge_rls_mean:.4f}, Mean-to-Vol: {ridge_rls_mean_vol:.4f}")
    print(f"XGBoost RLS Mean: {xg_rls_mean:.4f}, Mean-to-Vol: {xg_rls_mean_vol:.4f}")
    
    if xg_rls_mean > ridge_rls_mean:
        print("XGBoost RLS has a greater mean.")
    else:
        print("Ridge RLS has a greater or equal mean.")

    if xg_rls_mean_vol > ridge_rls_mean_vol:
        print("XGBoost RLS has a greater mean-to-vol ratio.")
    else:
        print("Ridge RLS has a greater or equal mean-to-vol ratio.")

def analyze_q3(ridge_df, xg_df):
    ridge_rls_beats = ridge_df['RLS'].mean() > ridge_df['Rsimple'].mean()
    xg_rls_beats = xg_df['RLS'].mean() > xg_df['Rsimple'].mean()

    print("\n--- Q3: RLS vs Rsimple ---")
    print(f"Ridge RLS beats Rsimple: {ridge_rls_beats}")
    print(f"XGBoost RLS beats Rsimple: {xg_rls_beats}")

def analyze_q4(ridge_df, xg_df):
    ridge_tmax = ridge_df.loc[ridge_df['RLS'].idxmax()]
    ridge_tmin = ridge_df.loc[ridge_df['RLS'].idxmin()]
    xgb_tmax = xg_df.loc[xg_df['RLS'].idxmax()]
    xgb_tmin = xg_df.loc[xg_df['RLS'].idxmin()]

    print("\n--- Q4: Best and Worst RLS Periods ---")
    print(f"Ridge Max RLS on {ridge_tmax['date']}: {ridge_tmax['RLS']:.4f} (alpha: {ridge_tmax['alpha']})")
    print(f"Ridge Min RLS on {ridge_tmin['date']}: {ridge_tmin['RLS']:.4f} (alpha: {ridge_tmin['alpha']})")
    print(f"XGBoost Max RLS on {xgb_tmax['date']}: {xgb_tmax['RLS']:.4f} (best_depth: {xgb_tmax['best_depth']})")
    print(f"XGBoost Min RLS on {xgb_tmin['date']}: {xgb_tmin['RLS']:.4f} (best_depth: {xgb_tmin['best_depth']})")

if __name__ == '__main__':
    if not os.path.exists('summary'):
        os.makedirs('summary')

    ridge_df = pd.read_csv('data/ridge_regression_results.csv')
    xg_df = pd.read_csv('data/xgboost_results.csv')

    plot_cumulative_returns(ridge_df.copy(), "Ridge")
    plot_cumulative_returns(xg_df.copy(), "XGBoost")

    analyze_q1(ridge_df, xg_df)
    analyze_q2(ridge_df, xg_df)
    analyze_q3(ridge_df, xg_df)
    analyze_q4(ridge_df, xg_df)
