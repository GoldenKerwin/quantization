from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
import pandas as pd
import numpy as np

def train_dfm_model(data, config):
    train_end = config['train_end_date']
    train_data = data.loc[:train_end]
    test_data = data.loc[train_end:]
    model = DynamicFactor(train_data, k_factors=3, factor_order=1)
    results = model.fit(maxiter=1000)
    in_sample_preds = pd.DataFrame(results.fittedvalues, index=train_data.index, columns=train_data.columns)
    out_sample_preds = []
    y = train_data.copy()
    for date in test_data.index:
        model = DynamicFactor(y, k_factors=3, factor_order=1)
        results = model.fit(maxiter=1000)
        fc = results.forecast(steps=1)
        out_sample_preds.append((date, fc.values[0]))
        y = pd.concat([y, test_data.loc[[date]]])
    out_sample_preds = pd.DataFrame([x[1] for x in out_sample_preds], index=[x[0] for x in out_sample_preds], columns=train_data.columns)
    return in_sample_preds, out_sample_preds