# 量化分析项目

## 概述
本项目预测15个ETF/ETN的开盘到开盘回报，使用VAR、DFM和LSTM模型，结合技术指标和宏观数据。

## 项目结构

```
quantitative_analysis_project/
├── data/
│   ├── raw_open_prices.csv
│   ├── daily_returns.csv
│   ├── technical_indicators.csv
│   └── macro_data.csv
├── src/
│   ├── data_acquisition.py
│   ├── data_processing.py
│   ├── model_development.py  # 原VAR模型，后续可能移除或重构
│   ├── model_evaluation.py
│   ├── models/               # 新增模型目录
│   │   ├── var_model.py      # 迁移后的VAR模型
│   │   ├── dfm_model.py      # 动态因子模型
│   │   └── lstm_model.py     # LSTM模型
│   └── utils.py
├── config/
│   └── config.yaml
├── logs/
│   └── project.log           # 新增日志目录
├── tests/
│   └── test_functions.py     # 新增测试目录
├── main.py
├── requirements.txt
└── README.md
```

## 安装与配置

1.  **克隆项目**（如果适用）或下载项目文件。
2.  **安装依赖**：
    打开终端或命令提示符，导航到项目根目录，然后运行：
    ```bash
    pip install -r requirements.txt
    ```
3.  **配置API密钥**：
    打开 `config/config.yaml` 文件，将 `api_key` (Alpha Vantage) 和 `fred_api_key` (FRED) 字段的值替换为您自己的API密钥。
    ```yaml
    api_key: 'YOUR_ALPHA_VANTAGE_API_KEY'
    fred_api_key: 'YOUR_FRED_API_KEY'
    # ... 其他配置
    ```
    如果您没有API密钥，可以从以下链接免费获取：
    *   Alpha Vantage: [Alpha Vantage官网](https://www.alphavantage.co/support/#api-key)
    *   FRED: [FRED API](https://fred.stlouisfed.org/docs/api/fred/)

## 使用说明

1.  确保您已完成上述安装与配置步骤。
2.  打开终端或命令提示符，导航到项目根目录。
3.  运行主程序：
    ```bash
    python main.py
    ```
4.  程序将执行以下步骤：
    *   加载配置。
    *   下载原始ETF开盘价数据和宏观经济数据。
    *   处理数据，计算日对数回报率和技术指标，合并数据。
    *   分别训练VAR、DFM和LSTM模型，并进行样本内和样本外预测。
    *   评估各模型表现，计算IC、MSE和Sharpe Ratio。
    *   输出评估结果到 `data/evaluation_results.csv` 和日志文件 `logs/project.log`。

## 数据说明

*   `data/raw_open_prices.csv`: 原始下载的ETF开盘价数据。
*   `data/daily_returns.csv`: 处理后的ETF日对数回报率数据。
*   `data/technical_indicators.csv`: 计算得到的技术指标数据 (如RSI, SMA)。
*   `data/macro_data.csv`: 从FRED获取的宏观经济数据 (如VIX, 10年期国债收益率)。
*   `data/evaluation_results.csv`: 各模型评估指标的汇总结果。

## 日志

详细的运行日志记录在 `logs/project.log` 文件中。

## 测试

可以运行单元测试来验证部分模块功能：
```bash
pytest tests/
```

## 注意事项

*   **API密钥限制**：Alpha Vantage和FRED的免费API密钥有请求频率和次数限制。
*   **数据对齐与处理**：确保ETF、宏观数据和技术指标的日期对齐，项目中已包含对缺失值的基本处理（前向填充）。
*   **模型参数**：各模型的参数可以在 `config/config.yaml` 中调整。
*   **计算资源**：LSTM模型训练可能需要较长时间，如果数据量大或模型复杂，建议使用GPU加速（如在Google Colab或配备GPU的本地环境运行）。
*   **文件编码**：所有文本文件默认使用UTF-8编码。

## 未来可能的改进方向

*   **更复杂的特征工程**：探索更多交互特征、非线性变换等。
*   **超参数调优**：对DFM和LSTM模型进行更细致的超参数搜索。
*   **集成学习**：尝试将多个模型的预测结果进行集成。
*   **动态资产选择**：根据市场状态动态选择要预测的资产。
*   **更全面的回测**：引入更复杂的回测框架，考虑交易成本、滑点等实际因素。
*   **在线学习**：使模型能够根据新流入的数据进行在线更新。