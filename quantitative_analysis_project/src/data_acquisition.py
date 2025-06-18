import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import time
from fredapi import Fred
from src.utils import load_config

def download_data(config):
    """Downloads daily open prices for a list of ETFs/ETNs from Alpha Vantage.

    Args:
        config (dict): Configuration dictionary containing API key, tickers, and start date.

    Returns:
        pd.DataFrame: DataFrame containing raw open prices, or None if an error occurs.
    """
    api_key = config['api_key']
    tickers = config['tickers']
    start_date_str = config['start_date']

    ts = TimeSeries(key=api_key, output_format='pandas')
    all_data = {}
    print("开始下载数据...")
    for i, ticker in enumerate(tickers):
        try:
            print(f"正在下载 {ticker} ({i+1}/{len(tickers)})...")
            data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
            # Alpha Vantage API returns data with '1. open', '2. high', etc. as column names
            # and dates as index in descending order.
            # We need to sort by date ascending and select the '1. open' column.
            open_prices = data['1. open'].sort_index()
            all_data[ticker] = open_prices
            # Respect API call limits
            if (i + 1) % 5 == 0 and i + 1 < len(tickers):
                print("暂停60秒以避免超出API速率限制...")
                time.sleep(60)
        except Exception as e:
            print(f"下载 {ticker} 数据时出错: {e}")
            # Optionally, decide how to handle errors, e.g., skip ticker or stop execution
            # For now, we'll print the error and continue, resulting in missing data for this ticker
            continue

    if not all_data:
        print("未能下载任何数据。")
        return None

    # Combine all data into a single DataFrame
    df = pd.DataFrame(all_data)

    # Convert index to datetime objects if not already
    df.index = pd.to_datetime(df.index)

    # Filter data from start_date
    df = df[df.index >= pd.to_datetime(start_date_str)]

    # Save to CSV
    raw_data_path = 'data/raw_open_prices.csv'
    try:
        df.to_csv(raw_data_path)
        print(f"原始开盘价数据已保存到 {raw_data_path}")
    except Exception as e:
        print(f"保存原始数据到CSV时出错: {e}")
        return None

    return df


def download_macro_data(config):
    fred = Fred(api_key=config['fred_api_key'])
    macro_vars = {
        'VIXCLS': 'VIX',  # VIX指数
        'DGS10': 'Treasury_10Y'  # 10年期国债收益率
    }
    macro_data = {}
    print("开始下载宏观经济数据...")
    for code, name in macro_vars.items():
        try:
            print(f"正在下载 {name} ({code})...")
            series = fred.get_series(code, observation_start=config['start_date'])
            macro_data[name] = series
            # FRED API 限制较宽松，但短期内大量请求也可能需要延时
            time.sleep(5) # 短暂延时
        except Exception as e:
            print(f"下载宏观数据 {name} ({code}) 时出错: {e}")
            continue

    if not macro_data:
        print("未能下载任何宏观经济数据。")
        return None

    macro_df = pd.concat(macro_data, axis=1)
    # 宏观数据通常是日度，但可能有周末或节假日缺失，需要填充
    # 同时，确保索引是DatetimeIndex以便与ETF数据对齐
    macro_df.index = pd.to_datetime(macro_df.index)
    # 使用前向填充处理缺失值 (例如周末VIX数据缺失)
    macro_df = macro_df.ffill()
    # 有些宏观数据可能在开始日期附近才有，再次筛选确保日期范围
    macro_df = macro_df[macro_df.index >= pd.to_datetime(config['start_date'])]
    # 再次填充，以防筛选后首行是NaN
    macro_df = macro_df.ffill().dropna() # 填充后再移除开始日期前可能产生的全NaN行

    macro_data_path = 'data/macro_data.csv'
    try:
        macro_df.to_csv(macro_data_path)
        print(f"宏观经济数据已保存到 {macro_data_path}")
    except Exception as e:
        print(f"保存宏观数据到CSV时出错: {e}")
        return None
    return macro_df

if __name__ == '__main__':
    # This is an example of how to run this module independently for testing
    # You would typically call download_data from main.py
    example_config = {
        'api_key': 'YOUR_API_KEY',  # Replace with your actual Alpha Vantage API key
        'tickers': ['SPY', 'QQQ', 'DIA'], # Example tickers
        'start_date': '2010-01-01'
    }
    # Make sure to replace 'YOUR_API_KEY' with a valid key to test
    if example_config['api_key'] == 'YOUR_API_KEY':
        print("请在 example_config 中替换 'YOUR_API_KEY' 为您的 Alpha Vantage API 密钥以进行测试。")
    else:
        raw_prices_df = download_data(example_config)
        if raw_prices_df is not None:
            print("\n下载的原始开盘价数据样本:")
            print(raw_prices_df.head())