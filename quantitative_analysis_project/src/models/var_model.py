from statsmodels.tsa.vector_ar.var_model import VAR
import pandas as pd

def train_var_model(data, config):
    train_end = config['train_end_date']
    train_data = data.loc[:train_end]
    test_data = data.loc[train_end:]
    model = VAR(train_data)
    # 确保最小滞后阶数为1
    lag_order = max(model.select_order(maxlags=config['max_lag']).aic, 1)
    model_fitted = model.fit(lag_order)
    in_sample_preds = model_fitted.fittedvalues
    out_sample_preds = []
    y = train_data.copy()
    for date in test_data.index:
        if len(y) >= lag_order:
            # 处理可能的零滞后阶数
            valid_lag = max(lag_order, 1)
            model = VAR(y)
            model_fitted = model.fit(valid_lag)
            fc = model_fitted.forecast(y.iloc[-lag_order:].values, steps=1)
            out_sample_preds.append((date, fc[0]))
            y = pd.concat([y, test_data.loc[[date]]])
    out_sample_preds = pd.DataFrame([x[1] for x in out_sample_preds], index=[x[0] for x in out_sample_preds], columns=train_data.columns)
    return in_sample_preds, out_sample_preds