import pandas as pd

def describe_data(df: pd.DataFrame) -> None:
    print("Data Head:")
    print(df.head())
    print(f"\nTotal number of unique stocks: {df['stock_id'].nunique()}")
    print(f"Total number of unique dates: {df['date'].nunique()}")

if __name__ == '__main__':
    df = pd.read_csv('data/processed_data.csv')
    describe_data(df)
