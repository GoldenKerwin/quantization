import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

def add_technical_indicators(data):
    print("开始计算技术指标...")
    indicators = pd.DataFrame(index=data.index)
    for ticker in data.columns:
        try:
            # Ensure there's enough data for indicator calculation
            if len(data[ticker].dropna()) >= 20: # SMA window is 20, RSI is 14
                indicators[f'{ticker}_RSI'] = RSIIndicator(data[ticker], window=14).rsi()
                indicators[f'{ticker}_SMA'] = SMAIndicator(data[ticker], window=20).sma_indicator()
            else:
                print(f"警告: {ticker} 的数据不足以计算技术指标，跳过。")
                indicators[f'{ticker}_RSI'] = np.nan
                indicators[f'{ticker}_SMA'] = np.nan
        except Exception as e:
            print(f"计算 {ticker} 的技术指标时出错: {e}")
            indicators[f'{ticker}_RSI'] = np.nan
            indicators[f'{ticker}_SMA'] = np.nan
            
    # Technical indicators can produce NaNs at the beginning, fill them
    indicators = indicators.ffill() # Forward fill first
    indicators = indicators.bfill() # Then backward fill for any remaining NaNs at the very beginning
    
    # Save indicators
    indicators_path = 'data/technical_indicators.csv'
    try:
        indicators.to_csv(indicators_path)
        print(f"技术指标数据已保存到 {indicators_path}")
    except Exception as e:
        print(f"保存技术指标到CSV时出错: {e}")
    print("技术指标计算完成。")
    return indicators

def remove_outliers(data, threshold=3):
    print("开始处理异常值...")
    # Calculate Z-scores only for numeric columns
    numeric_cols = data.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        z_scores = np.abs((data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std())
        # Replace outliers with NaN, then forward fill
        data_no_outliers = data.copy()
        data_no_outliers[numeric_cols] = data[numeric_cols].where(z_scores < threshold)
        data_no_outliers = data_no_outliers.ffill() # Fill NaNs created by outlier removal
        print("异常值处理完成。")
        return data_no_outliers
    else:
        print("数据中没有数值列，跳过异常值处理。")
        return data


def process_data(raw_data_df=None, macro_data_df=None, raw_data_path='data/raw_open_prices.csv', macro_data_path='data/macro_data.csv'):
    """Reads raw open price data, calculates daily log returns, and handles missing values.

    Args:
        raw_data_df (pd.DataFrame, optional): DataFrame with raw open prices. 
                                              If None, data is read from raw_data_path.
        raw_data_path (str, optional): Path to the CSV file with raw open prices.
                                         Defaults to 'data/raw_open_prices.csv'.

    Returns:
        pd.DataFrame: DataFrame containing daily log returns, or None if an error occurs.
    """
    print("开始处理数据...")

    # Load raw ETF data if not provided
    if raw_data_df is None:
        try:
            print(f"从 {raw_data_path} 读取原始ETF数据...")
            raw_data_df = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
        except FileNotFoundError:
            print(f"错误: 原始ETF数据文件 {raw_data_path} 未找到。请先运行数据获取模块。")
            return None
        except Exception as e:
            print(f"读取 {raw_data_path} 时出错: {e}")
            return None

    if raw_data_df.empty:
        print("错误: 原始ETF数据为空。")
        return None

    # Load macro data if not provided
    if macro_data_df is None:
        try:
            print(f"从 {macro_data_path} 读取宏观经济数据...")
            macro_data_df = pd.read_csv(macro_data_path, index_col=0, parse_dates=True)
            # Ensure macro data index is datetime, just in case
            macro_data_df.index = pd.to_datetime(macro_data_df.index)
        except FileNotFoundError:
            print(f"信息: 宏观经济数据文件 {macro_data_path} 未找到。将在没有宏观数据的情况下继续。")
            macro_data_df = None # Explicitly set to None if not found
        except Exception as e:
            print(f"读取 {macro_data_path} 时出错: {e}。将在没有宏观数据的情况下继续。")
            macro_data_df = None

    # Calculate log returns from raw ETF open prices
    raw_data_df = raw_data_df.sort_index()
    log_returns = np.log(raw_data_df / raw_data_df.shift(1))
    log_returns = log_returns.iloc[1:] # Remove first NaN row
    log_returns = log_returns.ffill() # Fill any NaNs in returns

    # Save daily returns
    daily_returns_path = 'data/daily_returns.csv'
    try:
        log_returns.to_csv(daily_returns_path)
        print(f"日度对数回报数据已保存到 {daily_returns_path}")
    except Exception as e:
        print(f"保存日度回报到CSV时出错: {e}")

    # Add technical indicators using raw prices (as indicators are typically price-based)
    # Ensure raw_data_df is aligned with log_returns index for consistency if needed later
    # For indicator calculation, use raw prices from the start
    technical_indicators = add_technical_indicators(raw_data_df)

    # Combine returns and technical indicators
    # Ensure indices are aligned before concat. log_returns starts one day later.
    # Align technical_indicators to log_returns' index
    aligned_indicators = technical_indicators.reindex(log_returns.index).ffill().bfill()
    combined_data = pd.concat([log_returns, aligned_indicators], axis=1)

    # Add macro data if available
    if macro_data_df is not None and not macro_data_df.empty:
        # Align macro_data_df to combined_data's index
        aligned_macro_data = macro_data_df.reindex(combined_data.index).ffill().bfill()
        combined_data = pd.concat([combined_data, aligned_macro_data], axis=1)
    
    # Remove outliers from the combined dataset
    combined_data_no_outliers = remove_outliers(combined_data)

    # Final check for NaNs and fill if any (e.g., if all data for a column was outlier)
    combined_data_final = combined_data_no_outliers.ffill().dropna(axis=0, how='any') # Drop rows if any value is still NaN after ffill

    if combined_data_final.isnull().any().any():
        print("警告: 最终合并数据中仍存在NaN值。请检查数据源和处理步骤。")
        print("包含NaN值的列:", combined_data_final.columns[combined_data_final.isnull().any()].tolist())

    # Save the fully processed and combined data
    processed_combined_data_path = 'data/processed_combined_data.csv'
    try:
        combined_data_final.to_csv(processed_combined_data_path)
        print(f"处理后的组合数据已保存到 {processed_combined_data_path}")
    except Exception as e:
        print(f"保存组合数据到CSV时出错: {e}")
        return None
    
    print("数据处理完成。")
    return combined_data_final

    # The old return logic is now part of the new flow above
    print("开始处理数据...")
    if raw_data_df is None:
        try:
            print(f"从 {raw_data_path} 读取原始数据...")
            raw_data_df = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
        except FileNotFoundError:
            print(f"错误: 原始数据文件 {raw_data_path} 未找到。请先运行数据获取模块。")
            return None
        except Exception as e:
            print(f"读取 {raw_data_path} 时出错: {e}")
            return None

    if raw_data_df.empty:
        print("错误: 原始数据为空。")
        return None

    # Calculate log returns: R_t = log(open_t / open_{t-1})
    # Ensure data is sorted by date for correct diff calculation
    raw_data_df = raw_data_df.sort_index()
    log_returns = np.log(raw_data_df / raw_data_df.shift(1))

    # Remove the first row of NaNs produced by .shift(1)
    log_returns = log_returns.iloc[1:]

    # Handle other missing values (e.g., using forward fill)
    # It's important to consider the implications of ffill, especially for financial time series.
    # For some models, interpolation or more sophisticated imputation might be better.
    # Here, we use ffill as specified, but also print a warning if NaNs remain.
    log_returns_filled = log_returns.ffill()

    if log_returns_filled.isnull().any().any():
        print("警告: 前向填充后仍然存在NaN值。考虑检查原始数据或使用不同的填充策略。")
        # Example: print columns with NaNs
        # print("包含NaN值的列:", log_returns_filled.columns[log_returns_filled.isnull().any()].tolist())
        # For simplicity, we proceed with potentially NaN-containing data if ffill wasn't enough.
        # Depending on the VAR model's tolerance for NaNs, this might need further handling.



if __name__ == '__main__':
    # This is an example of how to run this module independently for testing
    # Create dummy raw_open_prices.csv and macro_data.csv for testing
    dummy_raw_data = {
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
        'ETF1': [100, 101, 100.5, 102, 101.5],
        'ETF2': [200, 200.5, np.nan, 201, 202] # Include a NaN to test filling
    }
    dummy_raw_df = pd.DataFrame(dummy_raw_data).set_index('Date')
    try:
        dummy_raw_df.to_csv('data/raw_open_prices.csv')
        print("创建了用于测试的虚拟 raw_open_prices.csv 文件。")
    except Exception as e:
        print(f"创建虚拟ETF文件时出错: {e}")

    dummy_macro_data = {
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
        'VIX': [15, 15.5, 16, 15.8, 16.2],
        'Treasury_10Y': [3.5, 3.52, 3.51, 3.55, 3.53]
    }
    dummy_macro_df = pd.DataFrame(dummy_macro_data).set_index('Date')
    try:
        dummy_macro_df.to_csv('data/macro_data.csv')
        print("创建了用于测试的虚拟 macro_data.csv 文件。")
    except Exception as e:
        print(f"创建虚拟宏观文件时出错: {e}")

    # Test process_data with file paths (it will load them)
    combined_df = process_data() 
    if combined_df is not None:
        print("\n处理后的组合数据样本:")
        print(combined_df.head())
        print("\n组合数据列名:")
        print(combined_df.columns.tolist())
        print("\n组合数据NaN检查:")
        print(combined_df.isnull().sum())