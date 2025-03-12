import pandas as pd

price_col       = ['code', 'name', 'date', 'close', 'return', 'volume_shares', 'volume_dollar', 'market_value', 'pe', 'pb', 'ps', 'dividend_yield']
institution_col = ['code', 'name', 'date', 'f_net', 'i_net', 'd_net', 't_net']
factor_col      = ['code', 'name', 'date', 'mkt_rp', 'rf', 'mkt_pf', 'size_rp', 'bm_rp', 'div_rp', 'pe_rp']

def rename_and_convert_to_csv(file_name: str, col: list)-> pd.DataFrame:

    data = pd.read_excel(f'data/raw_data/{file_name}.xlsx')
    data.columns = col
    data['date'] = data['date'].apply(lambda x: x[:7].replace('/', '-'))
    data.to_csv(f'data/raw_data/{file_name}.csv', index=False)
    return data

def merge_raw_data(file: list)-> pd.DataFrame:

    price_data = pd.read_csv(f'data/raw_data/{file[0]}.csv')
    institution_data = pd.read_csv(f'data/raw_data/{file[1]}.csv')
    factor_data = pd.read_csv(f'data/raw_data/{file[2]}.csv')
    factor_data.drop(['code', 'name'], axis = 1, inplace = True)

    data = pd.merge(price_data, institution_data, on=['code', 'name', 'date'], how='left')
    data = pd.merge(data, factor_data, on = ['date'], how = 'left')
    data.to_csv('data/data.csv', index=False)
    return data

def drop_companies_with_incomplete_data(df: pd.DataFrame)-> pd.DataFrame:

    # drop companies with incomplete dates
    df = df[(df['date'] > '2005-08') & (df['date'] != '2025-03')]
    df.drop(['pe'], axis = 1, inplace = True)
    required_dates = len(df['date'].unique())
    df = df.groupby(['code', 'name']).filter(lambda x: len(x) == required_dates)
    
    # drop companies with missing values
    df = df.groupby(['code', 'name']).filter(lambda x: not x.isnull().values.any())
    df.to_csv('data/data_with_complete_dates.csv', index = False)
    return df

def check_null(df: pd.DataFrame)-> pd.DataFrame:

    null = df.isnull().sum()
    print(null)

def descriptive_statistics(df: pd.DataFrame)-> pd.DataFrame:

    df = df.drop(['code', 'name'], axis = 1)
    df = df.describe()
    df.to_csv('data/descriptive_statistics.csv')
    return df

def preview_data(df = pd.DataFrame)-> pd.DataFrame:

    df = pd.concat([df.head(50), df.tail(50)], axis = 0)
    df.to_csv('data/preview_data.csv', index = False)
    return df


if __name__ == '__main__':

    """
    # rename columns and convert from xlsx to csv
    for file_name, col in zip(['price', 'institution', 'factor'], [price_col, institution_col, factor_col]):
        rename_and_convert_to_csv(file_name, col)

    # merge three csv files
    merge_raw_data(['price', 'institution', 'factor'])

    # drop companies with incomplete time series data & drop companies with missing values
    drop_companies_with_incomplete_data(pd.read_csv('data/data.csv'))

    # check missing values
    check_null(pd.read_csv('data/data_with_complete_dates.csv'))

    # descriptive statistics
    descriptive_statistics(pd.read_csv('data/data_with_complete_dates.csv'))

    # preview data
    preview_data(pd.DataFrame(pd.read_csv('data/data_with_complete_dates.csv')))
    """
    ...