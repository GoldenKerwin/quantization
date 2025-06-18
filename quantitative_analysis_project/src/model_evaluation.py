import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os

def calculate_ic(predictions, actuals):
    """Calculates the Information Coefficient (Pearson correlation) between predictions and actuals."""
    if predictions.empty or actuals.empty:
        return np.nan
    common_index = predictions.index.intersection(actuals.index)
    if common_index.empty:
        return np.nan
    preds_aligned = predictions.loc[common_index].values.flatten()
    actuals_aligned = actuals.loc[common_index].values.flatten()
    if len(preds_aligned) < 2 or len(actuals_aligned) < 2:
        return np.nan
    # Remove NaNs that might exist after alignment or in original data
    mask = ~np.isnan(preds_aligned) & ~np.isnan(actuals_aligned)
    preds_aligned = preds_aligned[mask]
    actuals_aligned = actuals_aligned[mask]
    if len(preds_aligned) < 2 or len(actuals_aligned) < 2:
        return np.nan
    ic, _ = pearsonr(preds_aligned, actuals_aligned)
    return ic

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculates the Sharpe Ratio for a series of returns."""
    if returns.empty or returns.std() == 0:
        return np.nan
    # Assuming daily returns, annualize by sqrt(252)
    excess_returns = returns - risk_free_rate / 252 # Daily risk-free rate
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    return sharpe

def plot_model_performance(actual_returns_df, in_sample_preds_df, out_sample_preds_df, model_name, plot_path='plots'):
    """Plots actual vs. predicted returns for a single model (in-sample and out-of-sample)."""
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    num_series = actual_returns_df.shape[1]
    if num_series == 0:
        print(f"No series to plot for {model_name}.")
        return

    # Determine a reasonable number of subplots
    # Aim for max 2-3 columns for readability
    if num_series <= 3:
        n_cols = num_series
        n_rows = 1
    elif num_series <= 6:
        n_cols = 3
        n_rows = int(np.ceil(num_series / 3))
    else: # More than 6, maybe just plot first few or a summary
        print(f"Warning: {model_name} has {num_series} series. Plotting first 6 for performance.")
        num_series_to_plot = 6
        n_cols = 3
        n_rows = 2
    
    fig_height_per_row = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * fig_height_per_row), squeeze=False)
    axes_flat = axes.flatten()

    for i, ticker in enumerate(actual_returns_df.columns[:num_series_to_plot if num_series > 6 else num_series]):
        ax = axes_flat[i]
        if ticker in actual_returns_df.columns:
            ax.plot(actual_returns_df.index, actual_returns_df[ticker], label='Actual Returns', alpha=0.7)
        
        if in_sample_preds_df is not None and ticker in in_sample_preds_df.columns:
            common_idx_in = actual_returns_df.index.intersection(in_sample_preds_df.index)
            if not common_idx_in.empty:
                ax.plot(common_idx_in, in_sample_preds_df.loc[common_idx_in, ticker], label='In-sample Preds', linestyle='--')
        
        if out_sample_preds_df is not None and ticker in out_sample_preds_df.columns:
            common_idx_out = actual_returns_df.index.intersection(out_sample_preds_df.index)
            if not common_idx_out.empty:
                ax.plot(common_idx_out, out_sample_preds_df.loc[common_idx_out, ticker], label='Out-of-sample Preds', linestyle=':')
        
        ax.set_title(f'{model_name} - {ticker}')
        ax.legend()
        ax.grid(True)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Hide any unused subplots
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    fig.suptitle(f'{model_name} Performance: Actual vs. Predicted Returns', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    save_path = os.path.join(plot_path, f'{model_name.lower()}_performance_plot.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f"{model_name} performance plot saved to {save_path}")

def evaluate_model(actual_returns_df, in_sample_preds_df, out_sample_preds_df, model_name="UnknownModel", config=None):
    """Evaluates a model's predictions and returns key metrics."""
    print(f"\n--- Evaluating {model_name} ---")
    metrics = {}
    if config is None:
        config = {}
    plot_path = config.get('plot_path', 'plots') # Get plot_path from config
    risk_free_rate = config.get('risk_free_rate', 0.01) # Default risk-free rate if not in config

    # Ensure directories exist
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Plot actual vs predicted for this model
    plot_model_performance(actual_returns_df, in_sample_preds_df, out_sample_preds_df, model_name, plot_path)

    # In-sample IC
    if in_sample_preds_df is not None and not in_sample_preds_df.empty:
        ic_in_sample_list = []
        for col in actual_returns_df.columns.intersection(in_sample_preds_df.columns):
            ic_in_sample_list.append(calculate_ic(in_sample_preds_df[col], actual_returns_df[col]))
        metrics[f'{model_name}_IC_in_sample'] = np.nanmean(ic_in_sample_list) if ic_in_sample_list else np.nan
        print(f"{model_name} In-sample IC: {metrics.get(f'{model_name}_IC_in_sample', 'N/A'):.4f}")
    else:
        metrics[f'{model_name}_IC_in_sample'] = np.nan
        print(f"{model_name} In-sample predictions not available or empty.")

    # Out-of-sample IC
    if out_sample_preds_df is not None and not out_sample_preds_df.empty:
        ic_out_sample_list = []
        for col in actual_returns_df.columns.intersection(out_sample_preds_df.columns):
            ic_out_sample_list.append(calculate_ic(out_sample_preds_df[col], actual_returns_df[col]))
        metrics[f'{model_name}_IC_out_sample'] = np.nanmean(ic_out_sample_list) if ic_out_sample_list else np.nan
        print(f"{model_name} Out-of-sample IC: {metrics.get(f'{model_name}_IC_out_sample', 'N/A'):.4f}")

        # Out-of-sample Sharpe Ratio (based on a simple strategy: long if pred > 0, short if pred < 0)
        # This is a very basic strategy and assumes predictions are directly tradable signals.
        strategy_returns_list = []
        for col in actual_returns_df.columns.intersection(out_sample_preds_df.columns):
            common_idx_strat = out_sample_preds_df.index.intersection(actual_returns_df[col].dropna().index)
            if not common_idx_strat.empty:
                preds_for_strat = out_sample_preds_df.loc[common_idx_strat, col]
                actuals_for_strat = actual_returns_df.loc[common_idx_strat, col]
                # Simple strategy: if prediction is positive, take position in direction of actual return.
                # If prediction is negative, take position opposite to actual return (effectively shorting).
                # This is simplified. A more realistic strategy would use signals to decide long/short/neutral.
                # For now: signal = sign(prediction). Strategy return = signal * actual_return.
                signal = np.sign(preds_for_strat)
                strategy_return_col = signal * actuals_for_strat
                strategy_returns_list.append(strategy_return_col)
        
        if strategy_returns_list:
            combined_strategy_returns = pd.concat(strategy_returns_list, axis=1).mean(axis=1) # Average across assets
            metrics[f'{model_name}_Sharpe_out_sample'] = calculate_sharpe_ratio(combined_strategy_returns, risk_free_rate)
            print(f"{model_name} Out-of-sample Sharpe Ratio (simple strategy): {metrics.get(f'{model_name}_Sharpe_out_sample', 'N/A'):.4f}")
        else:
            metrics[f'{model_name}_Sharpe_out_sample'] = np.nan
            print(f"{model_name} Could not calculate Sharpe Ratio due to lack of aligned data or predictions.")

    else:
        metrics[f'{model_name}_IC_out_sample'] = np.nan
        metrics[f'{model_name}_Sharpe_out_sample'] = np.nan
        print(f"{model_name} Out-of-sample predictions not available or empty.")

    return metrics

def plot_combined_evaluation_results(all_metrics_df, config=None):
    """Plots combined evaluation metrics (like IC, Sharpe) for multiple models."""
    if all_metrics_df.empty:
        print("No combined metrics to plot.")
        return
    
    if config is None:
        config = {}
    plot_path = config.get('plot_path', 'plots')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Example: Plotting IC and Sharpe ratios if available
    # all_metrics_df should have metrics as index (e.g., 'VAR_IC_in_sample', 'AR_Sharpe_out_sample')
    # and potentially a 'Value' column, or be structured with models as columns.
    # Let's assume all_metrics_df is a DataFrame where index = metric_name (e.g. IC_in_sample), columns = model_names (e.g. VAR, AR)
    
    # Transpose if models are in index and metrics in columns for easier plotting by metric type
    # if not any(m in all_metrics_df.index for m in ['IC_in_sample', 'Sharpe_out_sample']): # Basic check
    #    if any(m in all_metrics_df.columns for m in ['IC_in_sample', 'Sharpe_out_sample']):
    #        all_metrics_df = all_metrics_df.T
            
    num_metrics = all_metrics_df.shape[0] # Number of unique metric types (e.g., IC_in, IC_out, Sharpe_out)
    num_models = all_metrics_df.shape[1]  # Number of models (e.g., VAR, AR)

    if num_metrics == 0 or num_models == 0:
        print("Not enough data in metrics DataFrame to plot.")
        return

    # Plot each metric type as a separate bar chart comparing models
    for metric_name in all_metrics_df.index:
        if all_metrics_df.loc[metric_name].isnull().all(): # Skip if all values for this metric are NaN
            print(f"Skipping plot for metric '{metric_name}' as all values are NaN.")
            continue
            
        fig, ax = plt.subplots(figsize=(max(6, num_models * 1.5), 5))
        all_metrics_df.loc[metric_name].plot(kind='bar', ax=ax, rot=0)
        ax.set_title(f'Comparison of Models: {metric_name}')
        ax.set_ylabel('Metric Value')
        ax.set_xlabel('Model')
        ax.grid(axis='y', linestyle='--')
        plt.tight_layout()
        save_path = os.path.join(plot_path, f'combined_metric_{metric_name.lower().replace(" ", "_")}.png')
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Combined metric plot for '{metric_name}' saved to {save_path}")

    print(f"Combined evaluation plots saved to '{plot_path}'.")


if __name__ == '__main__':
    # Create dummy data for testing evaluation functions
    dates1 = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'])
    dates2 = pd.to_datetime(['2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-07'])
    
    actuals = pd.DataFrame({'TickerA': np.random.randn(5)/100, 'TickerB': np.random.randn(5)/100}, index=dates1)
    preds_in = pd.DataFrame({'TickerA': np.random.randn(5)/100, 'TickerB': np.random.randn(5)/100}, index=dates1)
    preds_out = pd.DataFrame({'TickerA': np.random.randn(5)/100, 'TickerB': np.random.randn(5)/100}, index=dates2)

    test_config = {
        'plot_path': 'evaluation_plots_output',
        'risk_free_rate': 0.01
    }
    if not os.path.exists(test_config['plot_path']):
        os.makedirs(test_config['plot_path'])

    print("--- Testing evaluate_model for ModelX ---")
    model_x_metrics = evaluate_model(actual_returns_df=actuals, 
                                     in_sample_preds_df=preds_in, 
                                     out_sample_preds_df=preds_out, 
                                     model_name="ModelX",
                                     config=test_config)
    print("ModelX Metrics:", model_x_metrics)

    print("\n--- Testing evaluate_model for ModelY (fewer predictions) ---")
    preds_in_y = preds_in.iloc[:3, :1] # TickerA only, fewer days
    preds_out_y = preds_out.iloc[1:4, :1] # TickerA only, different fewer days
    model_y_metrics = evaluate_model(actual_returns_df=actuals, 
                                     in_sample_preds_df=preds_in_y, 
                                     out_sample_preds_df=preds_out_y, 
                                     model_name="ModelY",
                                     config=test_config)
    print("ModelY Metrics:", model_y_metrics)

    # Prepare data for plot_combined_evaluation_results
    # It expects a DataFrame where index = metric_type (e.g. IC_in_sample), columns = model_names
    if model_x_metrics and model_y_metrics:
        combined_metrics_data = {}
        for k, v in model_x_metrics.items():
            model_n, metric_n = k.split('_', 1)
            if metric_n not in combined_metrics_data:
                combined_metrics_data[metric_n] = {}
            combined_metrics_data[metric_n][model_n] = v
        
        for k, v in model_y_metrics.items():
            model_n, metric_n = k.split('_', 1)
            if metric_n not in combined_metrics_data:
                combined_metrics_data[metric_n] = {}
            combined_metrics_data[metric_n][model_n] = v
        
        final_metrics_for_plot = pd.DataFrame(combined_metrics_data)
        # Ensure correct orientation: metrics as index, models as columns
        # If DataFrame constructor got it wrong, transpose. Example: if it's Model Name as index.
        # if not any(m in final_metrics_for_plot.index for m in ['IC_in_sample', 'Sharpe_out_sample']):
        #     final_metrics_for_plot = final_metrics_for_plot.T

        print("\n--- Testing plot_combined_evaluation_results ---")
        print("Metrics DataFrame for combined plot:")
        print(final_metrics_for_plot)
        plot_combined_evaluation_results(final_metrics_for_plot, config=test_config)
        print(f"\nCombined evaluation plots test finished. Check '{test_config['plot_path']}' directory.")
    else:
        print("Skipping combined plot test as not all model metrics were generated.")