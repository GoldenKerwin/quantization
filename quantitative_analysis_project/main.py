from src.utils import load_config, setup_logging, ensure_dir
from src.data_acquisition import download_data
from src.data_processing import process_data
from src.data_acquisition import download_data, download_macro_data # Added download_macro_data
from src.data_processing import process_data # Will be updated to handle macro_data and indicators
from src.models.var_model import train_var_model
from src.models.dfm_model import train_dfm_model
from src.models.lstm_model import train_lstm_model
from src.model_evaluation import evaluate_model, plot_combined_evaluation_results, plot_model_performance
import pandas as pd # Added pandas for DataFrame creation
import logging
import os

def main():
    # 设置日志记录
    setup_logging(log_file_name='quantitative_analysis.log')
    logging.info("量化分析项目开始执行...")

    # 确保数据和配置目录存在 (config目录应已存在, data目录可能需要创建)
    # utils.py 中的 ensure_dir 也可以在各模块内部按需调用，或在此处统一处理
    ensure_dir('data') 
    # config_dir = 'config' # 假设config目录必须存在，否则load_config会失败

    # 加载配置文件
    # 调整路径以适应从项目根目录运行 main.py 的情况
    config_file_path = os.path.join('config', 'config.yaml')
    config = load_config(config_file_path)
    if config is None:
        logging.error("未能加载配置文件。项目终止。")
        return

    if config.get('api_key') == 'YOUR_API_KEY_HERE' or not config.get('api_key'):
        logging.error("错误: Alpha Vantage API 密钥未在 config/config.yaml 中配置。")
        logging.error("请访问 https://www.alphavantage.co/support/#api-key 获取免费API密钥并更新配置文件。")
        print("错误: Alpha Vantage API 密钥未在 config/config.yaml 中配置。")
        print("请访问 https://www.alphavantage.co/support/#api-key 获取免费API密钥并更新配置文件。")
        return

    # 1. 数据获取
    logging.info("步骤 1: 数据获取")
    raw_etf_data_file = os.path.join('data', 'raw_open_prices.csv')
    macro_data_file = os.path.join('data', 'macro_data.csv')

    if os.path.exists(raw_etf_data_file):
        logging.info(f"ETF原始数据文件 {raw_etf_data_file} 已存在，跳过下载。")
    else:
        logging.info(f"ETF原始数据文件 {raw_etf_data_file} 未找到，开始下载...")
        download_data(config) # Downloads ETF data
        if not os.path.exists(raw_etf_data_file):
            logging.error(f"ETF数据获取未能创建 {raw_etf_data_file}。项目终止。")
            return
        logging.info("ETF数据获取完成。")

    if config.get('fred_api_key') == 'your_fred_api_key' or not config.get('fred_api_key'):
        logging.warning("警告: FRED API 密钥未在 config/config.yaml 中配置。将跳过宏观数据下载。")
        macro_data = None # Initialize macro_data as None if key is missing
    elif os.path.exists(macro_data_file):
        logging.info(f"宏观数据文件 {macro_data_file} 已存在，跳过下载。")
        try:
            macro_data = pd.read_csv(macro_data_file, index_col=0, parse_dates=True)
            logging.info(f"已从 {macro_data_file} 加载宏观数据。")
        except Exception as e:
            logging.error(f"从 {macro_data_file} 加载宏观数据失败: {e}。将尝试重新下载。")
            macro_data = download_macro_data(config) # Downloads macro data
    else:
        logging.info(f"宏观数据文件 {macro_data_file} 未找到，开始下载...")
        macro_data = download_macro_data(config) # Downloads macro data
        if macro_data is None or not os.path.exists(macro_data_file):
            logging.warning(f"宏观数据获取可能未成功创建 {macro_data_file}。后续步骤可能受影响。")
        else:
            logging.info("宏观数据获取完成。")
    logging.info("数据获取阶段完成。")

    # 2. 数据处理
    logging.info("步骤 2: 数据处理")
    # The new processed data file includes returns, indicators, and macro data
    processed_data_file = os.path.join('data', 'processed_combined_data.csv') 
    # daily_returns.csv and technical_indicators.csv are intermediate files created by process_data

    if os.path.exists(processed_data_file):
        logging.info(f"处理后的组合数据文件 {processed_data_file} 已存在，跳过处理步骤。")
        try:
            data_for_models = pd.read_csv(processed_data_file, index_col=0, parse_dates=True)
            logging.info(f"已从 {processed_data_file} 加载组合数据。")
        except Exception as e:
            logging.error(f"从 {processed_data_file} 加载组合数据失败: {e}。将尝试重新处理。")
            # Ensure raw_etf_data_file exists for reprocessing
            if not os.path.exists(raw_etf_data_file):
                logging.error(f"原始ETF数据文件 {raw_etf_data_file} 在尝试重新处理时未找到。")
                return
            # Load raw ETF data for reprocessing
            raw_etf_data = pd.read_csv(raw_etf_data_file, index_col=0, parse_dates=True)
            data_for_models = process_data(raw_etf_data, macro_data) # macro_data might be None
    else:
        logging.info(f"处理后的组合数据文件 {processed_data_file} 未找到，开始处理...")
        # Ensure raw_etf_data_file exists for processing
        if not os.path.exists(raw_etf_data_file):
            logging.error(f"原始ETF数据文件 {raw_etf_data_file} 在开始处理时未找到。")
            return
        raw_etf_data = pd.read_csv(raw_etf_data_file, index_col=0, parse_dates=True)
        data_for_models = process_data(raw_etf_data, macro_data) # macro_data might be None
    
    if data_for_models is None:
        logging.error("数据处理失败。项目终止。")
        return
    logging.info("数据处理完成。")

    # Extract actual returns for evaluation (original ETF tickers)
    # Assuming 'tickers' in config refers to the base ETFs for which returns are predicted and evaluated.
    # data_for_models contains returns for these tickers directly (e.g., 'SPY', 'QQQ' columns)
    # and also technical indicators (e.g., 'SPY_RSI') and macro data (e.g., 'VIX').
    # For evaluation, we need the actual returns of the 'tickers'.
    actual_returns_for_eval = data_for_models[config['tickers']].copy()
    # Ensure actual_returns_for_eval is not all NaN, which can happen if tickers are not in data_for_models
    if actual_returns_for_eval.isnull().all().all():
        logging.error(f"未能从处理后的数据中提取到有效的实际回报数据进行评估。检查config中的tickers是否与processed_combined_data.csv中的列名匹配（不含技术指标后缀）。可用列: {data_for_models.columns.tolist()}")
        return
    # The VAR model in the original plan used daily_returns.csv, which are log returns of ETFs.
    # The new DFM and LSTM models will use the combined data.
    # For VAR, we should pass only the ETF returns part of data_for_models.
    var_input_data = data_for_models[config['tickers']].copy() # VAR model uses only ETF returns

    # 3. 模型开发与评估
    logging.info("步骤 3: 模型开发与评估")
    all_results = {}

    # VAR模型
    logging.info("训练VAR模型...")
    var_in_preds_file = os.path.join('data', 'var_in_sample_predictions.csv')
    var_out_preds_file = os.path.join('data', 'var_out_sample_predictions.csv')
    if os.path.exists(var_in_preds_file) and os.path.exists(var_out_preds_file):
        logging.info("VAR预测文件已存在，加载预测...")
        var_in_preds = pd.read_csv(var_in_preds_file, index_col=0, parse_dates=True)
        var_out_preds = pd.read_csv(var_out_preds_file, index_col=0, parse_dates=True)
    else:
        var_in_preds, var_out_preds = train_var_model(var_input_data, config)
        if var_in_preds is not None and var_out_preds is not None:
            var_in_preds.to_csv(var_in_preds_file)
            var_out_preds.to_csv(var_out_preds_file)
            logging.info("VAR预测已保存。")
        else:
            logging.error("VAR模型训练失败。")
    if var_in_preds is not None and var_out_preds is not None:
        logging.info("评估VAR模型...")
        var_metrics = evaluate_model(actual_returns_df=actual_returns_for_eval, 
                                     in_sample_preds_df=var_in_preds, 
                                     out_sample_preds_df=var_out_preds, 
                                     model_name='VAR', config=config)
        all_results.update(var_metrics)

    # DFM模型
    logging.info("训练DFM模型...")
    dfm_in_preds_file = os.path.join('data', 'dfm_in_sample_predictions.csv')
    dfm_out_preds_file = os.path.join('data', 'dfm_out_sample_predictions.csv')
    if os.path.exists(dfm_in_preds_file) and os.path.exists(dfm_out_preds_file):
        logging.info("DFM预测文件已存在，加载预测...")
        dfm_in_preds = pd.read_csv(dfm_in_preds_file, index_col=0, parse_dates=True)
        dfm_out_preds = pd.read_csv(dfm_out_preds_file, index_col=0, parse_dates=True)
    else:
        # DFM uses the full combined data (returns, indicators, macro)
        dfm_in_preds, dfm_out_preds = train_dfm_model(data_for_models, config)
        if dfm_in_preds is not None and dfm_out_preds is not None:
            dfm_in_preds.to_csv(dfm_in_preds_file)
            dfm_out_preds.to_csv(dfm_out_preds_file)
            logging.info("DFM预测已保存。")
        else:
            logging.error("DFM模型训练失败。")
    if dfm_in_preds is not None and dfm_out_preds is not None:
        logging.info("评估DFM模型...")
        # DFM predictions are for all columns in data_for_models. We need to select only ETF tickers for evaluation.
        dfm_metrics = evaluate_model(actual_returns_df=actual_returns_for_eval, 
                                     in_sample_preds_df=dfm_in_preds[config['tickers']], 
                                     out_sample_preds_df=dfm_out_preds[config['tickers']], 
                                     model_name='DFM', config=config)
        all_results.update(dfm_metrics)

    # LSTM模型
    logging.info("训练LSTM模型...")
    lstm_in_preds_file = os.path.join('data', 'lstm_in_sample_predictions.csv')
    lstm_out_preds_file = os.path.join('data', 'lstm_out_sample_predictions.csv')
    if os.path.exists(lstm_in_preds_file) and os.path.exists(lstm_out_preds_file):
        logging.info("LSTM预测文件已存在，加载预测...")
        lstm_in_preds = pd.read_csv(lstm_in_preds_file, index_col=0, parse_dates=True)
        lstm_out_preds = pd.read_csv(lstm_out_preds_file, index_col=0, parse_dates=True)
    else:
        # LSTM uses the full combined data
        lstm_in_preds, lstm_out_preds = train_lstm_model(data_for_models, config)
        if lstm_in_preds is not None and lstm_out_preds is not None:
            lstm_in_preds.to_csv(lstm_in_preds_file)
            lstm_out_preds.to_csv(lstm_out_preds_file)
            logging.info("LSTM预测已保存。")
        else:
            logging.error("LSTM模型训练失败。")
    if lstm_in_preds is not None and lstm_out_preds is not None:
        logging.info("评估LSTM模型...")
        # LSTM predictions are for ETF tickers only, as per lstm_model.py design
        lstm_metrics = evaluate_model(actual_returns_df=actual_returns_for_eval, 
                                      in_sample_preds_df=lstm_in_preds, 
                                      out_sample_preds_df=lstm_out_preds, 
                                      model_name='LSTM', config=config)
        all_results.update(lstm_metrics)

    logging.info("模型开发与评估完成。")

    # 4. 汇总结果
    logging.info("步骤 4: 汇总结果")
    if all_results:
        results_df = pd.DataFrame([all_results])
        results_file_path = os.path.join('data', 'evaluation_results.csv')
        try:
            results_df.to_csv(results_file_path, index=False)
            logging.info(f"所有模型评估结果已保存到 {results_file_path}")
            print(f"\n--- 所有模型评估结果 ---")
            print(results_df.to_string())
        except Exception as e:
            logging.error(f"保存汇总评估结果时出错: {e}")
    else:
        logging.warning("没有可汇总的评估结果。")
    
    logging.info("量化分析项目执行完毕。")

    # Load configuration
    CONFIG_PATH = 'config.yaml'
    # Ensure plot_path exists
    plot_path = config.get('plot_path', 'plots')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    print(f"Plots will be saved to: {plot_path}")

    # Load and prepare data (assuming this function is robust)
    features_df, returns_df = load_and_prepare_data(config)

    if features_df is None or returns_df is None:
        print("Failed to load or prepare data. Exiting.")
        return

    # --- Model Training and Evaluation ---
    all_model_metrics = {}

    # 1. VAR Model
    if config.get('run_var_model', True):
        print("\n--- Training VAR Model ---")
        # VAR model uses a combined dataframe of features and returns if returns are endogenous
        # Adjust data_for_var based on how train_var_model expects it
        data_for_var = pd.concat([features_df, returns_df], axis=1)
        # Ensure data_for_var has frequency if loaded from CSV and not set
        if data_for_var.index.freq is None:
            print("Warning: Inferring business day frequency for VAR data index.")
            data_for_var = data_for_var.asfreq('B', method='ffill') # Or 'D' or other appropriate freq
        
        var_model_fit, var_in_sample_preds, var_out_sample_preds, var_metrics = train_var_model(data_for_var, returns_df, config)
        if var_metrics:
            all_model_metrics['VAR'] = var_metrics
            print("VAR Model Metrics:", var_metrics)
        else:
            print("VAR model training or evaluation failed.")

    # 2. AR Model
    if config.get('run_ar_model', True):
        print("\n--- Training AR Models ---")
        # AR model typically uses returns_df directly, features_df can be for exogenous vars if supported
        ar_model_fit, ar_in_sample_preds, ar_out_sample_preds, ar_metrics = train_ar_model(features_df, returns_df, config)
        if ar_metrics:
            all_model_metrics['AR'] = ar_metrics
            print("AR Model Metrics:", ar_metrics)
        else:
            print("AR model training or evaluation failed.")

    # 3. LSTM Model
    if config.get('run_lstm_model', True):
        print("\n--- Training LSTM Model ---")
        # LSTM model needs features and returns. train_lstm_model should handle its specific data prep.
        lstm_model_instance, lstm_in_sample_preds, lstm_out_sample_preds, lstm_metrics = train_lstm_model(features_df, returns_df, config)
        if lstm_metrics:
            all_model_metrics['LSTM'] = lstm_metrics
            print("LSTM Model Metrics:", lstm_metrics)
        else:
            print("LSTM model training or evaluation failed.")

    # --- Combined Evaluation Plot ---
    if len(all_model_metrics) > 1:
        print("\n--- Generating Combined Model Evaluation Plot ---")
        # Consolidate metrics into a DataFrame suitable for plot_combined_evaluation_results
        # Expected format: Metrics as index, Models as columns
        consolidated_metrics_df = pd.DataFrame()
        for model_name, metrics_dict in all_model_metrics.items():
            # metrics_dict is like {'IC_in_sample_mean': 0.1, 'MSE_out_sample': 0.002, ...}
            # We want to turn this into a Series where index is metric_name and value is the metric value
            # then append/concat this Series as a new column to consolidated_metrics_df
            model_series = pd.Series(metrics_dict, name=model_name)
            if consolidated_metrics_df.empty:
                consolidated_metrics_df = pd.DataFrame(model_series)
            else:
                consolidated_metrics_df = pd.concat([consolidated_metrics_df, model_series], axis=1)

        if not consolidated_metrics_df.empty:
            plot_combined_evaluation_results(consolidated_metrics_df, config)
            print(f"Combined evaluation plot saved in '{config.get('plot_path', 'plots')}'.")
        else:
            print("No metrics available to generate combined plot.")
    elif len(all_model_metrics) == 1:
        print("Only one model was run. Skipping combined evaluation plot.")
        # Optionally, could still plot individual performance for the single model if desired
        # model_name, metrics_dict = list(all_model_metrics.items())[0]
        # print(f"Individual metrics for {model_name}: {metrics_dict}")
    else:
        print("No models were run or no metrics were generated. Skipping combined plot.")

    print("\n--- Main execution finished ---")

if __name__ == '__main__':
    main()