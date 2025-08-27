import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path, index_col=0)
    print(data.head(0))
    return data

if __name__ == "__main__":
    df = load_data("data/raw_data.csv")
    df.rename(columns = {
        '證券代碼': 'stock_id', 
        '年月': 'date', 
        '收盤價(元)_月': 'close', 
        '報酬率％_月': 'ret', 
        '週轉率％_月': 'turnover', 
        '股價淨值比-TEJ': 'pb', 
        '股價營收比-TEJ': 'ps',
        '股利殖利率-TSE': 'dividend_yield', 
        '現金股利率': 'cash_dividend_yield', 
        '市場風險溢酬': 'market_risk_premium', 
        '規模溢酬 (5因子)': 'size_premium', 
        '淨值市價比溢酬': 'value_premium', 
        '動能因子': 'momentum_factor',
        '無風險利率': 'risk_free_rate', 
        '投資因子': 'investment_factor', 
        '盈利能力因子': 'profitability_factor'
    }, inplace=True)
    df['ret'] = df['ret'] / 100
    df['ret'] = df['ret'].apply(lambda x: round(x, 5))
    df['date'] = pd.to_datetime(df['date'], format='%Y%m').dt.to_period('M')
    df.sort_values(by=['date', 'stock_id'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv("data/processed_data.csv", index=False)