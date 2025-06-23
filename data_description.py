import pandas as pd
import numpy as np


def describe_data(df: pd.DataFrame) -> None:
    df = df.copy()
    print(df.head())
    stocks_group = df.groupby('stock_id')
    print(f"Total number of stocks: {len(stocks_group)}")
    print(f'Total number of dates: {len(df["date"].unique())}')


if __name__ == '__main__':
    df = pd.read_csv('data/processed_data.csv')
    describe_data(df)