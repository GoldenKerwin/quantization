import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import grangercausalitytests, acf
import numpy as np # Added for dummy data in main
from .model_evaluation import evaluate_model, plot_combined_evaluation_results # Ensure correct import

def load_and_prepare_data(features_path='data/transformed_features.csv', returns_path='data/daily_returns.csv'):
    """Loads features and returns, aligns them, and handles missing values."""
    try:
        features_df = pd.read_csv(features_path, index_col=0, parse_dates=True)
        returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

    common_dates = features_df.index.intersection(returns_df.index)
    features_df = features_df.loc[common_dates]
    returns_df = returns_df.loc[common_dates]

    features_df = features_df.ffill().dropna()
    returns_df = returns_df.ffill().dropna()

    common_dates = features_df.index.intersection(returns_df.index)
    features_df = features_df.loc[common_dates]
    returns_df = returns_df.loc[common_dates]

    return features_df, returns_df

def plot_var_model_diagnostics(model_fit, features_df, config):
    """Plots ACF, IRF, and Granger causality for the VAR model."""
    plot_path = config.get('plot_path', 'plots')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Plot ACF of residuals
    try:
        residuals = model_fit.resid
        num_variables = residuals.shape[1]
        fig_acf, axes_acf = plt.subplots(num_variables, 1, figsize=(10, 2 * num_variables), sharex=True)
        if num_variables == 1:
            axes_acf = [axes_acf] # Make it iterable
        for i, ax in enumerate(axes_acf):
            pd.plotting.autocorrelation_plot(residuals.iloc[:, i], ax=ax)
            ax.set_title(f'ACF of Residuals - {residuals.columns[i]}')
        plt.tight_layout()
        acf_path = os.path.join(plot_path, 'var_residuals_acf.png')
        fig_acf.savefig(acf_path)
        plt.close(fig_acf)
        print(f"VAR Residuals ACF plot saved to {acf_path}")
    except Exception as e:
        print(f"Error plotting VAR residuals ACF: {e}")

    # Plot Impulse Response Functions (IRF)
    try:
        irf = model_fit.irf(config.get('var_irf_periods', 10))
        # Plotting all responses to an impulse in the first variable as an example
        fig_irf = irf.plot(impulse=features_df.columns[0]) 
        fig_irf.suptitle(f'Impulse Response from {features_df.columns[0]}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        irf_path = os.path.join(plot_path, 'var_irf_example.png')
        fig_irf.savefig(irf_path)
        plt.close(fig_irf)
        print(f"VAR Impulse Response Function plot saved to {irf_path}")
    except Exception as e:
        print(f"Error plotting VAR IRF: {e}")

    # Granger Causality Tests
    print("\nPerforming Granger Causality Tests...")
    max_lag_granger = model_fit.k_ar
    granger_results_summary = []
    # Consider testing only for returns if they are part of features_df
    # Or a subset of variables to avoid excessive computation/output
    variables_to_test = features_df.columns
    if len(variables_to_test) > 5: # Limit for practical reasons in an example
        print(f"Warning: Limiting Granger causality to first 5 variables due to high number ({len(variables_to_test)}).")
        variables_to_test = features_df.columns[:5]

    for i_col in variables_to_test:
        for j_col in variables_to_test:
            if i_col == j_col:
                continue
            try:
                # Ensure data passed to grangercausalitytests has no NaNs and is 2D
                test_data_granger = features_df[[i_col, j_col]].dropna()
                if test_data_granger.shape[0] < max_lag_granger + 5: # Check for sufficient data
                    print(f"Skipping Granger causality for {j_col} -> {i_col} due to insufficient data after NaN drop.")
                    continue
                test_result = grangercausalitytests(test_data_granger, maxlag=[max_lag_granger], verbose=False)
                p_value = test_result[max_lag_granger][0]['ssr_ftest'][1]
                granger_results_summary.append({'Excluding': j_col, 'Dependent': i_col, 'Lag': max_lag_granger, 'P-value': p_value})
                if p_value < 0.05:
                    print(f"Granger Causality: {j_col} -> {i_col} (p-value: {p_value:.4f}) at lag {max_lag_granger}")
            except Exception as e:
                print(f"Error performing Granger causality test for {j_col} -> {i_col}: {e}")
    
    if granger_results_summary:
        granger_df = pd.DataFrame(granger_results_summary)
        granger_path = os.path.join(plot_path, 'var_granger_causality_summary.csv')
        granger_df.to_csv(granger_path, index=False)
        print(f"Granger causality summary saved to {granger_path}")

def train_var_model(features_df, returns_df, config):
    """Trains a VAR model and generates predictions."""
    print("\n--- Training VAR Model ---")
    model_name = "VAR"
    plot_path = config.get('plot_path', 'plots')

    # For VAR, we typically model the features themselves, or features and returns together.
    # If we want to predict returns, returns should be part of the endogenous variables.
    # Let's assume for now we are modeling the features_df directly to keep it simple,
    # and then we'd need a separate mechanism to map VAR forecasts to return predictions if features don't include returns.
    # Or, more commonly, include returns in the VAR system.
    
    # For simplicity in this example, let's assume features_df contains the variables to model with VAR.
    # If returns_df has multiple tickers, VAR can model them jointly if they are included in features_df.
    # Let's use features_df as the input to VAR. If you want to predict specific returns, ensure they are columns in features_df.
    
    # Ensure data is stationary (e.g., by using returns or differenced series)
    # For this example, we'll assume features_df is already stationary (e.g., it contains log returns or pct changes)
    data_for_var = features_df.copy()
    if data_for_var.empty:
        print("Error: Data for VAR model is empty after preparation.")
        return None, None, None

    # Split data
    # The problem asks to predict R_{t+2} using info up to t.
    # So, if we train up to index `t_train_end`, the first out-of-sample prediction is for `t_train_end + 2`.
    train_end_date_str = config.get('train_end_date', '2022-12-31')
    try:
        train_end_date = pd.to_datetime(train_end_date_str)
    except ValueError:
        print(f"Error: Invalid train_end_date format '{train_end_date_str}'. Using last 80% for training.")
        train_size = int(len(data_for_var) * 0.8)
        train_data = data_for_var.iloc[:train_size]
        test_data = data_for_var.iloc[train_size:]
    else:
        if train_end_date not in data_for_var.index:
            # Find the closest date in index if exact match not found
            closest_date = data_for_var.index[data_for_var.index.get_indexer([train_end_date], method='nearest')[0]]
            if closest_date > train_end_date and data_for_var.index.get_loc(closest_date) > 0:
                # if closest is future, take previous
                train_end_date = data_for_var.index[data_for_var.index.get_loc(closest_date) -1]
            else:
                train_end_date = closest_date
            print(f"Warning: Specified train_end_date {train_end_date_str} not in data. Using {train_end_date.strftime('%Y-%m-%d')} instead.")
        
        train_data = data_for_var.loc[data_for_var.index <= train_end_date]
        test_data = data_for_var.loc[data_for_var.index > train_end_date]

    if train_data.empty or test_data.empty:
        print("Error: Training or testing data is empty after split. Check train_end_date and data availability.")
        return None, None, None

    print(f"Training data from {train_data.index.min()} to {train_data.index.max()} ({len(train_data)} samples)")
    print(f"Test data from {test_data.index.min()} to {test_data.index.max()} ({len(test_data)} samples)")

    # Select VAR model order (lag)
    # This can be time-consuming. For now, let's use a fixed lag or a simple selection.
    # Or use information criteria like AIC, BIC. Maxlags should be reasonable.
    # model_order_selection = VAR(train_data).select_order(maxlags=10)
    # lag_order = model_order_selection.aic
    lag_order = config.get('var_lag_order', 5) # Default to 5 if not in config
    print(f"Using VAR lag order: {lag_order}")

    # Fit the VAR model
    try:
        model = VAR(train_data)
        model_fit = model.fit(lag_order)
        print("VAR model fitted successfully.")
        print(model_fit.summary())
        plot_var_model_diagnostics(model_fit, train_data, config) # Pass config
    except Exception as e:
        print(f"Error fitting VAR model: {e}")
        return None, None, None, None # Added None for metrics

    # In-sample predictions (1-step ahead)
    # The problem asks for R_{t+2} prediction. `fittedvalues` are 1-step ahead.
    # If model is on R_t, `fittedvalues` at t are P(R_t | info_{t-1})
    # To get P(R_{t+1} | info_t) for in-sample, this is tricky with `fittedvalues`.
    # Let's use forecast for 1-step ahead on training data for simplicity, then adjust for R_{t+2}
    # `model_fit.fittedvalues` are one-step-ahead predictions on the training data.
    # If `y_t` is the data at time `t`, `fittedvalues.loc[t]` is the prediction for `y_t` using data up to `t-1`.
    # This is P(y_t | I_{t-1}).
    # If we want to predict R_{t+1} using info up to t (as per problem for in-sample if we interpret it that way)
    # then we need to be careful. The problem statement is: "predict t+1 to t+2 open to open return (i.e. log(open_t+2 / open_t+1)) based on all information up to t."
    # This is R_{t+2}. So, for in-sample, if we are at time `t` in training data, we want to predict R_{t+2}.
    # This means we need a 2-step ahead forecast from `t`.
    # `model_fit.predict(start=train_data.index[lag_order], end=train_data.index[-1], steps=2)` is not how it works.
    # We need to iterate or use `forecast` carefully.

    # For in-sample P(R_{t+2} | I_t):
    # This requires forecasting 2 steps ahead from each point `t` in the training period.
    # This is computationally intensive. A common simplification for in-sample is 1-step ahead.
    # Let's assume for in-sample evaluation, we use 1-step ahead predictions of the modeled variables.
    # If returns_df columns are part of `data_for_var`, then `fittedvalues` will contain their 1-step preds.
    in_sample_preds_raw = model_fit.fittedvalues
    # These are P(y_t | I_{t-1}). If y_t is R_t, this is P(R_t | I_{t-1}).
    # We need to align this with `returns_df` for evaluation.
    # If `in_sample_preds_raw` at index `d` is P(R_d | I_{d-1}), then it aligns with `returns_df.loc[d]`.
    # Let's assume `returns_df` columns are directly in `data_for_var` for now.
    # And that `evaluate_model` expects predictions for R_d to be indexed by d.
    in_sample_preds_df = in_sample_preds_raw[returns_df.columns.intersection(in_sample_preds_raw.columns)]
    print(f"In-sample predictions generated. Shape: {in_sample_preds_df.shape}")

    # Out-of-sample predictions (R_{t+2} using info up to t)
    # We need to forecast 2 steps ahead iteratively.
    print("Generating out-of-sample predictions (2-steps ahead)...")
    out_sample_preds_list = []
    history = train_data.copy()
    
    # The first prediction is for `test_data.index[0]`, which corresponds to `train_data.index[-1] + 1*freq`.
    # If we need to predict R_{t+2} using info up to t (where t is the last day of training data),
    # then the first prediction target date is `train_data.index[-1] + 2*freq`.
    # The `forecast` method takes `y` (history) and `steps`.
    # `model_fit.forecast(y=history_t, steps=h)` forecasts `h` steps from the end of `history_t`.
    # So, if `history_t` ends at `t`, `steps=2` will give P(y_{t+1}|I_t) and P(y_{t+2}|I_t).
    # We need the P(y_{t+2}|I_t) part.

    num_forecast_periods = len(test_data)
    # We need to ensure test_data has enough periods for the 2-step ahead prediction to be meaningful.
    # If test_data.index[k] is the date for which we want R_{test_data.index[k]}, and this is R_{t_k + 2}
    # then info is up to t_k.
    # The loop should go up to `len(test_data) - 1` if we are making predictions for dates *within* test_data.index
    # The target dates for predictions will be `test_data.index` if they represent R_{t+2}.

    # Let `t` be the last day of `history`. We want to predict for `t+2`.
    # The `test_data.index` are the dates for which we want to have predictions.
    # If `test_data.index[i]` is `d`, we want P(R_d | I_{d-2}).
    # So, `history` should go up to `d-2`.

    out_sample_preds_collector = {}
    current_history_df = train_data.copy() # Keep history as DataFrame for easier indexing

    for i in range(num_forecast_periods):
        target_date = test_data.index[i]
        # History for forecasting P(y_{target_date} | I_{target_date - 2 periods})
        # The VAR.forecast method takes the lagged values of y itself.
        # model_fit.forecast(y=history_values, steps=h)
        # history_values should be the last `lag_order` observations.
        
        # To predict for target_date (which is t+2), we need data up to t.
        # So, history should end at target_date - 2 periods.
        history_end_date = target_date - 2 * pd.tseries.frequencies.to_offset(data_for_var.index.freq)
        
        # Ensure history_end_date is in data_for_var.index and >= train_data.index[0]
        if history_end_date < data_for_var.index[0]:
             print(f"Skipping OOS prediction for {target_date} due to insufficient history before train start.")
             # Fill with NaNs or handle as per strategy
             out_sample_preds_collector[target_date] = np.full(data_for_var.shape[1], np.nan)
             continue
        
        history_for_forecast = data_for_var.loc[data_for_var.index <= history_end_date]
        if history_for_forecast.shape[0] < lag_order:
            print(f"Skipping OOS prediction for {target_date} due to insufficient history length ({history_for_forecast.shape[0]} < {lag_order}).")
            out_sample_preds_collector[target_date] = np.full(data_for_var.shape[1], np.nan)
            continue

        # The `forecast` method uses the model's fitted parameters and takes recent `y` values.
        # It needs the last `k_ar` (lag_order) observations as a NumPy array.
        y_prev_for_forecast = history_for_forecast.iloc[-lag_order:].values
        
        forecast_result = model_fit.forecast(y=y_prev_for_forecast, steps=2)
        prediction_for_target_date = forecast_result[1, :] # Index 1 for 2-steps ahead
        out_sample_preds_collector[target_date] = prediction_for_target_date

    out_sample_preds_df = pd.DataFrame.from_dict(out_sample_preds_collector, orient='index', columns=data_for_var.columns)
    out_sample_preds_df.dropna(how='all', inplace=True) # Drop rows that might be all NaN if skipped
    out_sample_preds_df = out_sample_preds_df[returns_df.columns.intersection(out_sample_preds_df.columns)]
    print(f"Out-of-sample predictions generated. Shape: {out_sample_preds_df.shape}")

    var_metrics = evaluate_model(actual_returns_df=returns_df, 
                                 in_sample_preds_df=in_sample_preds_df, 
                                 out_sample_preds_df=out_sample_preds_df,
                                 model_name=model_name,
                                 config=config) # Pass config
    
    return model_fit, in_sample_preds_df, out_sample_preds_df, var_metrics


def train_ar_model(features_df, returns_df, config):
    """Trains an AR model for each ticker in returns_df and generates predictions."""
    print("\n--- Training AR Models ---")
    model_name = "AR"
    plot_path = config.get('plot_path', 'plots') # Already present, ensure used if AR specific plots added
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # AR model is univariate. We train one for each ticker in returns_df.
    # We can use features_df as exogenous variables if desired.
    all_in_sample_preds = []
    all_out_sample_preds = []

    train_end_date_str = config.get('train_end_date', '2022-12-31')
    try:
        train_end_date = pd.to_datetime(train_end_date_str)
    except ValueError:
        print(f"Error: Invalid train_end_date format '{train_end_date_str}'. Using last 80% for training.")
        # Fallback handled by splitting logic if date is bad
        pass # Will be handled per-ticker if needed

    for ticker in returns_df.columns:
        print(f"\nTraining AR model for {ticker}...")
        # Prepare data for this ticker
        # Target variable y_t is R_t (return for ticker at time t)
        # We want to predict R_{t+2} using info up to t.
        # So, the series to model is R_t. Forecasts will be 2-steps ahead.
        y_ticker = returns_df[ticker].dropna()
        if y_ticker.empty:
            print(f"Skipping {ticker} due to no data.")
            continue

        # Exogenous variables (optional)
        # Align features with y_ticker. Features at time `t` are X_t.
        # When predicting R_{t+2} using X_t, we need X_t for the forecast period as well.
        # This means we need to forecast exogenous variables if they are not known in advance.
        # For simplicity, let's use only lagged values of y_ticker (no exogenous for now).
        exog_ticker = None 
        # if not features_df.empty:
        #     exog_ticker = features_df.loc[y_ticker.index].dropna(axis=1, how='all') # Align and remove all-NaN feature columns
        #     # Ensure exog has no NaNs for the training period by aligning again after y_ticker.dropna()
        #     common_idx_ar = y_ticker.index.intersection(exog_ticker.index)
        #     y_ticker = y_ticker.loc[common_idx_ar]
        #     exog_ticker = exog_ticker.loc[common_idx_ar]
        #     if exog_ticker.isnull().any().any():
        #         print(f"Warning: Exogenous features for {ticker} contain NaNs after alignment. Filling with ffill/bfill.")
        #         exog_ticker = exog_ticker.ffill().bfill().dropna() # Handle remaining NaNs robustly
        #         common_idx_ar = y_ticker.index.intersection(exog_ticker.index)
        #         y_ticker = y_ticker.loc[common_idx_ar]
        #         exog_ticker = exog_ticker.loc[common_idx_ar]
        #     if exog_ticker.empty:
        #         exog_ticker = None

        # Split data for this ticker
        try:
            train_end_date_ticker = pd.to_datetime(train_end_date_str) # Re-parse for safety
            if train_end_date_ticker not in y_ticker.index:
                closest_date_ticker = y_ticker.index[y_ticker.index.get_indexer([train_end_date_ticker], method='nearest')[0]]
                if closest_date_ticker > train_end_date_ticker and y_ticker.index.get_loc(closest_date_ticker) > 0:
                    train_end_date_ticker = y_ticker.index[y_ticker.index.get_loc(closest_date_ticker) -1]
                else:
                    train_end_date_ticker = closest_date_ticker
                print(f"Warning for {ticker}: Specified train_end_date {train_end_date_str} not in data. Using {train_end_date_ticker.strftime('%Y-%m-%d')} instead.")
            train_y = y_ticker.loc[y_ticker.index <= train_end_date_ticker]
            test_y = y_ticker.loc[y_ticker.index > train_end_date_ticker]
            # train_exog, test_exog = (None, None)
            # if exog_ticker is not None:
            #     train_exog = exog_ticker.loc[train_y.index]
            #     test_exog = exog_ticker.loc[test_y.index]
        except Exception as e:
            print(f"Error splitting data for {ticker} with date {train_end_date_str}: {e}. Using 80/20 split.")
            train_size_ticker = int(len(y_ticker) * 0.8)
            train_y = y_ticker.iloc[:train_size_ticker]
            test_y = y_ticker.iloc[train_size_ticker:]
            # train_exog, test_exog = (None, None)
            # if exog_ticker is not None:
            #     train_exog = exog_ticker.iloc[:train_size_ticker]
            #     test_exog = exog_ticker.iloc[train_size_ticker:]

        if train_y.empty or test_y.empty:
            print(f"Skipping {ticker} due to empty train/test data after split.")
            continue
        
        print(f"AR training for {ticker}: {len(train_y)} samples, testing: {len(test_y)} samples.")

        # Select AR model order
        # sel = ar_select_order(train_y, maxlag=config.get('ar_max_lag', 10), ic='aic', glob=True, exog=train_exog)
        # ar_lags = sel.ar_lags
        # if not ar_lags: # If ar_select_order returns empty, default to 1
        #     ar_lags = [1]
        ar_lags = config.get('ar_lag_order', 5) # Default to 5 if not in config, can be a list or int
        print(f"Using AR lags for {ticker}: {ar_lags}")

        # Fit AR model
        try:
            # ar_model = AutoReg(train_y, lags=ar_lags, exog=train_exog)
            ar_model = AutoReg(train_y, lags=ar_lags, old_names=False) # exog=train_exog removed for simplicity
            ar_model_fit = ar_model.fit()
            print(f"AR model for {ticker} fitted successfully.")
            print(ar_model_fit.summary())
        except Exception as e:
            print(f"Error fitting AR model for {ticker}: {e}")
            continue

        # In-sample predictions (P(R_t | I_{t-1}))
        # `predict` with `dynamic=False` gives 1-step ahead.
        # We need P(R_{t+2} | I_t). This is complex for in-sample with AutoReg's predict.
        # For simplicity, let's use 1-step ahead predictions for in-sample evaluation.
        # These are P(R_t | I_{t-1}).
        # start_idx = ar_model_fit.k_ar # First index for which prediction is available
        # if train_exog is not None:
        #     start_idx = max(ar_model_fit.k_ar, train_exog.index.get_loc(train_y.index[ar_model_fit.k_ar]))
        
        # AutoReg `predict` method: `start` and `end` are indices relative to the original `endog` series.
        # `fittedvalues` are available from `ar_model_fit.predict(start=train_y.index[0], end=train_y.index[-1], dynamic=False)`
        # but they are P(R_t | I_{t-1}).
        in_sample_preds_ticker = ar_model_fit.predict(start=train_y.index[0], end=train_y.index[-1], dynamic=False)
        in_sample_preds_ticker.name = ticker # Set series name for later concat
        all_in_sample_preds.append(in_sample_preds_ticker)
        print(f"AR in-sample predictions for {ticker} generated.")

        # Out-of-sample predictions (P(R_{t+2} | I_t))
        # We need to forecast 2 steps ahead from end of training data, iteratively.
        # `ar_model_fit.predict(start, end, dynamic=True)` can do multi-step but needs care.
        # `ar_model_fit.forecast(steps=h)` is simpler for fixed model.
        # If `train_y` ends at `t_train_end`, `forecast(steps=2)` gives P(R_{t_train_end+1}|I_{t_train_end}) and P(R_{t_train_end+2}|I_{t_train_end}).
        # We need the second one.

        out_sample_preds_ticker_collector = {}
        current_train_y = train_y.copy()
        # current_train_exog = train_exog.copy() if train_exog is not None else None

        for i in range(len(test_y)):
            # History for AR model should be `current_train_y`
            # We need to predict for `test_y.index[i]` which is R_d.
            # This R_d is R_{t_hist_end + 2} if t_hist_end is d-2.
            # So, the model needs to be refit or use a forecast method that takes history.
            # `AutoReg.predict` can take `endog` and `exog` for the period up to forecast start.
            # `ar_model_fit.predict(start=idx_of_t+1, end=idx_of_t+2, dynamic=True)`
            # This is complex. Let's use `forecast` method by re-instantiating model or ensuring `model_fit` can do it.
            # The `model_fit` from `AutoReg` is fixed. To do iterative 2-step ahead, we'd refit or manage state.
            
            # Simpler: use the last `ar_lags` of `current_train_y` and `current_train_exog` to make a 2-step forecast.
            # `ar_model_fit.forecast(steps=2, exog=future_exog_for_2_steps)`
            # If no exog, just `ar_model_fit.forecast(steps=2)` based on data up to end of `train_y` used in `.fit()`.
            # This means the forecast is always from the end of the initial `train_y`.
            # To do iterative (expanding or rolling window):
            if i == 0:
                # First out-of-sample forecast is from the end of the initial training data
                pred_2_step = ar_model_fit.forecast(steps=2) # Forecasts for t+1, t+2 from end of train_y
            else:
                # Refit model with data up to previous step for true iterative forecasting
                # This is computationally expensive. Alternative: use `predict` with `dynamic` carefully.
                # For now, let's assume a simpler approach: the model is fixed, and we are interested
                # in its performance on `test_y` using its state at the end of `train_y`.
                # This is not a true iterative forecast for R_{t+2} over the whole test period.
                # A proper iterative 2-step ahead forecast would be:
                # history_y = y_ticker.loc[y_ticker.index <= test_y.index[i-2]] if i >=2 else train_y
                # temp_model = AutoReg(history_y, lags=ar_lags, old_names=False).fit()
                # pred_2_step = temp_model.forecast(steps=2)
                # This is too slow for many tickers / long test sets.

                # Let's assume the task implies fixed model, forecasting 2 steps from various points.
                # If we need P(R_d | I_{d-2}), where d is in test_y.index:
                # We need to use `ar_model_fit.predict(start=d-1, end=d, dynamic=True)` but this needs `d-1` to be in the model's memory.
                # The `predict` method of `AutoRegResults` is for in-sample and out-of-sample from end of original sample.
                # To get P(R_d | I_{d-2}) for d in test_y.index:
                # We need to provide `endog` up to `d-2` to a predict/forecast call.
                # `ar_model.predict(params=ar_model_fit.params, start=y_ticker.index.get_loc(d-2_periods_ago), end=y_ticker.index.get_loc(d))`
                # This is getting very complex. Let's use a simplified out-of-sample: fixed model, predict sequence from train_end.
                if i == 0: # Only calculate this once for the fixed model from train_end
                    num_oos_steps = len(test_y)
                    # Forecast `num_oos_steps + 1` to get up to R_{t_train_end + num_oos_steps + 1}
                    # We need R_{t_train_end + 2} ... R_{t_train_end + len(test_y) + 1}
                    # So, forecast `len(test_y)` times, taking the 2nd step each time is not right with fixed model.
                    # Instead, forecast a sequence of `len(test_y)` points, where each is 2-steps ahead of some `t`.
                    # This implies `dynamic=True` from `len(train_y) - lag_order + 2` effectively.
                    # The `predict` method with `start` and `end` in `test_y.index` range, with `dynamic=False` up to `train_y.index[-1]`
                    # and then `dynamic=True` effectively, or just use `forecast` for a block.
                    
                    # We need P(R_{d} | I_{d-2}) for d in test_y.index.
                    # The `ar_model_fit.predict` can take `start` and `end` as dates from `test_y.index`.
                    # However, `dynamic=True` makes it a simulation from that point.
                    # `dynamic=False` would require actuals up to `d-1` to predict `d` (1-step).
                    # For 2-step P(R_d | I_{d-2}), we need to call forecast iteratively.
                    # This was the issue with VAR too. Let's make it consistent.
                    # Iterative 2-step ahead forecast:
                    history_iter_y = train_y.copy()
                    for k_oos in range(len(test_y)):
                        target_oos_date = test_y.index[k_oos]
                        # Model needs to be on history_iter_y to forecast
                        # Ensure history_iter_y has enough data for lags
                        if len(history_iter_y) < ar_lags:
                            print(f"Skipping OOS for {ticker} at {target_oos_date}, not enough history ({len(history_iter_y)} < {ar_lags})")
                            out_sample_preds_ticker_collector[target_oos_date] = np.nan
                            # Update history with actual to proceed if possible
                            if k_oos < len(test_y) -1:
                                # Ensure test_y.iloc[[k_oos]] is a Series with the correct index to concat
                                actual_to_append = test_y.iloc[[k_oos]]
                                if not isinstance(actual_to_append, pd.Series):
                                     actual_to_append = pd.Series(actual_to_append.values.flatten(), index=[test_y.index[k_oos]])
                                history_iter_y = pd.concat([history_iter_y, actual_to_append])
                            continue
                    
                        temp_model_iter = AutoReg(history_iter_y, lags=ar_lags, old_names=False).fit()
                        forecast_iter_2step = temp_model_iter.forecast(steps=2)
                        out_sample_preds_ticker_collector[target_oos_date] = forecast_iter_2step.iloc[1]
                        
                        if k_oos < len(test_y) -1:
                            actual_to_append = test_y.iloc[[k_oos]]
                            if not isinstance(actual_to_append, pd.Series):
                                actual_to_append = pd.Series(actual_to_append.values.flatten(), index=[test_y.index[k_oos]])
                            history_iter_y = pd.concat([history_iter_y, actual_to_append])
                    # Removed break, loop should complete for all OOS points

            # If the loop was broken, this part is reached after all oos preds are collected.
            if not out_sample_preds_ticker_collector and len(test_y) > 0: # Fallback if iterative failed or not run
                 # Fallback: fixed model, forecast sequence from end of training.
                 # This is P(R_{train_end+k} | I_{train_end}) for k=1,2,...
                 # We need P(R_{train_end+2}), P(R_{train_end+3}), ...
                 # So, we need the elements from index 1 upwards from a forecast of `len(test_y)+1` steps.
                print(f"Warning for {ticker}: Using fixed model forecast for out-of-sample due to iterative setup.")
                oos_forecast_block = ar_model_fit.forecast(steps=len(test_y) + 1) # Forecast R_{t+1} ... R_{t+len(test_y)+1}
                if len(oos_forecast_block) > 1:
                    preds_for_R_tplus2_onwards = oos_forecast_block.iloc[1:] # Get R_{t+2}, R_{t+3}, ...
                    for k_oos in range(len(test_y)):
                        if k_oos < len(preds_for_R_tplus2_onwards):
                            out_sample_preds_ticker_collector[test_y.index[k_oos]] = preds_for_R_tplus2_onwards.iloc[k_oos]
                        else: # Should not happen if steps were len(test_y)+1
                            out_sample_preds_ticker_collector[test_y.index[k_oos]] = np.nan
                else: # Handle case where forecast block is too short
                    for k_oos in range(len(test_y)):
                         out_sample_preds_ticker_collector[test_y.index[k_oos]] = np.nan


        out_sample_preds_ticker = pd.Series(out_sample_preds_ticker_collector).sort_index()
        out_sample_preds_ticker.name = ticker
        all_out_sample_preds.append(out_sample_preds_ticker)
        print(f"AR out-of-sample predictions for {ticker} generated.")

    # Combine predictions from all tickers
    if not all_in_sample_preds or not all_out_sample_preds:
        print("Error: No AR predictions were generated for any ticker.")
        return None, None, None, None
        
    ar_in_sample_df = pd.concat(all_in_sample_preds, axis=1).dropna(how='all')
    ar_out_sample_df = pd.concat(all_out_sample_preds, axis=1).dropna(how='all')

    print(f"Combined AR in-sample predictions shape: {ar_in_sample_df.shape}")
    print(f"Combined AR out-of-sample predictions shape: {ar_out_sample_df.shape}")

    # Evaluate AR model (combined for all tickers)
    ar_metrics = evaluate_model(actual_returns_df=returns_df,
                                in_sample_preds_df=ar_in_sample_df,
                                out_sample_preds_df=ar_out_sample_df,
                                model_name=model_name,
                                config=config) # Pass config

    return None, ar_in_sample_df, ar_out_sample_df, ar_metrics


if __name__ == '__main__':
    config = {
        'train_end_date': '2021-12-31',
        'var_lag_order': 3,
        'var_irf_periods': 5, 
        'ar_lag_order': 3, # Can be int for fixed lags or list for specific lags
        'plot_path': 'model_plots_output_dev', # Ensure this path is used by plotting functions
        'risk_free_rate': 0.015 # Example, ensure used by Sharpe ratio in evaluation
    }

    if not os.path.exists(config['plot_path']):
        os.makedirs(config['plot_path'])

    print("Loading and preparing data for __main__ example...")
    # Create dummy data for quick testing
    dates = pd.date_range(start='2021-01-01', end='2022-06-30', freq='B') 
    dummy_feature_data = np.random.randn(len(dates), 2)
    dummy_return_data = np.random.randn(len(dates), 2) / 100
    
    features_df = pd.DataFrame(dummy_feature_data, index=dates, columns=['Feature1', 'Feature2'])
    returns_df_main = pd.DataFrame(dummy_return_data, index=dates, columns=['TickerA_ret', 'TickerB_ret'])
    data_for_var_main = pd.concat([features_df, returns_df_main], axis=1)

    # Ensure index has frequency for VAR OOS prediction logic that uses pd.tseries.frequencies.to_offset
    # This is crucial if data is loaded from CSV without explicit frequency.
    if data_for_var_main.index.freq is None:
        print("Warning: Inferring business day frequency for data_for_var_main index.")
        data_for_var_main = data_for_var_main.asfreq('B', method='ffill')
    if returns_df_main.index.freq is None:
        print("Warning: Inferring business day frequency for returns_df_main index.")
        returns_df_main = returns_df_main.asfreq('B', method='ffill')
    if features_df.index.freq is None:
        print("Warning: Inferring business day frequency for features_df index.")
        features_df = features_df.asfreq('B', method='ffill')


    if features_df is not None and returns_df_main is not None:
        print("\n--- Running VAR Model Example ---")
        var_model_fit, var_in_sample, var_out_sample, var_eval_metrics = train_var_model(data_for_var_main, returns_df_main, config)
        if var_eval_metrics:
            print("\nVAR Model Evaluation Metrics:")
            for k, v in var_eval_metrics.items():
                print(f"{k}: {v}")

        print("\n--- Running AR Model Example ---")
        ar_model_fit, ar_in_sample, ar_out_sample, ar_eval_metrics = train_ar_model(features_df, returns_df_main, config)
        if ar_eval_metrics:
            print("\nAR Model Evaluation Metrics:")
            for k, v in ar_eval_metrics.items():
                print(f"{k}: {v}")
        
        # Combine metrics for plot_combined_evaluation_results
        if var_eval_metrics and ar_eval_metrics:
            all_metrics_data = {}
            # Consolidate metrics from different models
            for model_prefix, metrics_dict in [('VAR', var_eval_metrics), ('AR', ar_eval_metrics)]:
                for metric_name_full, value in metrics_dict.items():
                    # metric_name_full is like 'IC_in_sample_mean', 'MSE_out_sample'
                    # We want the base metric name (e.g. 'IC_in_sample_mean') as the row index
                    # and the model_prefix (e.g. 'VAR') as the column name
                    if metric_name_full not in all_metrics_data:
                        all_metrics_data[metric_name_full] = {}
                    all_metrics_data[metric_name_full][model_prefix] = value
            
            # Convert to DataFrame: Metrics as index, Models as columns
            final_metrics_df_main = pd.DataFrame(all_metrics_data)
            # The plot_combined_evaluation_results expects metrics as index and models as columns.
            # Current structure of all_metrics_data should give Metrics as rows, Models as columns directly.
            # If it's Models x Metrics (e.g., VAR is an index), then transpose: 
            # if 'VAR' in final_metrics_df_main.index: final_metrics_df_main = final_metrics_df_main.T

            print("\n--- Plotting Combined Model Evaluation Results ---")
            # Ensure plot_combined_evaluation_results is correctly imported and called
            plot_combined_evaluation_results(final_metrics_df_main, config=config) # Pass config
            print(f"\nExample run finished. Check '{config['plot_path']}' directory for plots.")

    else:
        print("Failed to load data for __main__ example.")