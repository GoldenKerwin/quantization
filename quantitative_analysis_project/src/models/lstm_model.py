import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split # For validation split
import copy # For saving best model
import matplotlib.pyplot as plt # For plotting

# PyTorch LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # Get output from the last time step
        return out

def prepare_lstm_data(data_values, lookback, n_features_to_predict):
    X, y = [], []
    for i in range(lookback, len(data_values)):
        X.append(data_values[i-lookback:i, :])
        y.append(data_values[i, :n_features_to_predict]) # Predict only the first n_features_to_predict (e.g., ETF returns)
    return np.array(X), np.array(y)

def train_lstm_model(data, config):
    train_end = config['train_end_date']
    val_ratio = config['lstm'].get('validation_ratio', 0.1) # Get validation ratio from config, default 0.1
    patience = config['lstm'].get('early_stopping_patience', 10) # Get patience from config, default 10
    lookback = config['lstm']['lookback']
    epochs = config['lstm']['epochs']
    n_outputs = len(config['tickers']) # Number of ETF tickers to predict
    plot_path = config.get('plot_path', 'plots') # Get plot path from config
    model_name = config.get('model_name', 'lstm') # Get model name for plot titles

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Separate features and target for scaling
    all_features_df = data.copy()
    target_etfs_df = data[config['tickers']].copy()

    # Scale features
    scaler_features = MinMaxScaler(feature_range=(-1, 1))
    scaled_features_values = scaler_features.fit_transform(all_features_df.values)
    
    scaler_targets = MinMaxScaler(feature_range=(-1, 1))
    # Fit scaler_targets on the target columns of the training data portion only
    train_target_etfs_values = target_etfs_df[data.index <= train_end].values
    scaler_targets.fit(train_target_etfs_values)

    # Split data into training and testing based on date
    train_val_data_scaled = scaled_features_values[data.index <= train_end]
    test_scaled_features_values = scaled_features_values[data.index > train_end]

    # Split training data into actual training and validation sets
    # Ensure chronological split for time series data
    split_index = int(len(train_val_data_scaled) * (1 - val_ratio))
    train_data_for_lstm = train_val_data_scaled[:split_index]
    val_data_for_lstm = train_val_data_scaled[split_index:]

    X_train_np, y_train_np = prepare_lstm_data(train_data_for_lstm, lookback, n_outputs)
    X_val_np, y_val_np = prepare_lstm_data(val_data_for_lstm, lookback, n_outputs)
    
    # Ensure validation set is not empty
    if len(X_val_np) == 0 or len(y_val_np) == 0:
        print("Warning: Validation set is empty. Adjust validation_ratio or data size. Skipping validation.")
        # Fallback: train on all train_val_data_scaled and skip early stopping
        X_train_np, y_train_np = prepare_lstm_data(train_val_data_scaled, lookback, n_outputs)
        X_train = torch.from_numpy(X_train_np).float().to(device)
        y_train = torch.from_numpy(y_train_np).float().to(device)
        X_val, y_val = None, None # No validation
    else:
        X_train = torch.from_numpy(X_train_np).float().to(device)
        y_train = torch.from_numpy(y_train_np).float().to(device)
        X_val = torch.from_numpy(X_val_np).float().to(device)
        y_val = torch.from_numpy(y_val_np).float().to(device)

    input_dim = X_train.shape[2]
    hidden_dim = config['lstm'].get('hidden_dim', 50)
    output_dim = n_outputs
    num_layers = config['lstm'].get('num_layers', 1)

    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lstm'].get('learning_rate', 0.001))

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if X_val is not None and y_val is not None:
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_losses.append(val_loss.item())
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
                print(f'Validation loss improved to {val_loss.item():.4f}. Saving model.')
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs. Best val loss: {best_val_loss:.4f}')
                if best_model_state:
                    model.load_state_dict(best_model_state)
                break
        else: # No validation set
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}')
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_losses: # Only plot if validation was performed
        plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{model_name.upper()} Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    # Ensure plot_path directory exists
    import os
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    plt.savefig(os.path.join(plot_path, f'{model_name}_training_loss.png'))
    plt.close()
    print(f"Training loss plot saved to {os.path.join(plot_path, f'{model_name}_training_loss.png')}")

    if X_val is not None and y_val is not None: # Only load best model if validation was performed
        if best_model_state and epochs_no_improve >= patience:
            print("Loading best model weights for final predictions.")
            model.load_state_dict(best_model_state)
        elif not best_model_state and epochs_no_improve < patience: # Training finished before patience met, but there was improvement
             if best_model_state: # This case implies training finished, and best_model_state was set at least once
                print("Training finished. Loading best model weights observed during training.")
                model.load_state_dict(best_model_state)
             else: # Training finished, no improvement at all, or validation was skipped.
                print("Warning: Training finished. No improvement in validation loss or validation skipped. Using last model state.")
        elif not best_model_state: # No improvement at all during training
             print("Warning: No improvement in validation loss during training. Using last model state.")

    # In-sample predictions on the entire period before `train_end_date`
    X_train_full_np, y_train_full_np = prepare_lstm_data(train_val_data_scaled, lookback, n_outputs)
    train_val_indices = data.index[data.index <= train_end][lookback:lookback+len(X_train_full_np)]
    X_train_full_torch = torch.from_numpy(X_train_full_np).float().to(device)

    model.eval()
    with torch.no_grad():
        in_sample_preds_scaled = model(X_train_full_torch).cpu().numpy()
    
    if in_sample_preds_scaled.ndim == 1:
        in_sample_preds_scaled = in_sample_preds_scaled.reshape(-1, 1)
    
    in_sample_preds_unscaled = scaler_targets.inverse_transform(in_sample_preds_scaled)
    in_sample_preds_df = pd.DataFrame(in_sample_preds_unscaled, 
                                      index=train_val_indices, 
                                      columns=config['tickers'])
    
    # Actual in-sample targets for plotting
    actual_in_sample_targets_unscaled = scaler_targets.inverse_transform(y_train_full_np)
    actual_in_sample_df = pd.DataFrame(actual_in_sample_targets_unscaled,
                                       index=train_val_indices,
                                       columns=config['tickers'])

    # Out-of-sample predictions (iterative)
    out_sample_preds_list = []
    current_history_scaled = list(train_val_data_scaled[-lookback:])
    test_data_indices = data.index[data.index > train_end]
    actual_test_scaled_features_values = scaled_features_values[data.index > train_end]
    # Prepare actual out-of-sample targets for plotting
    _, y_test_full_np = prepare_lstm_data(test_scaled_features_values, lookback, n_outputs)
    actual_out_sample_targets_unscaled = scaler_targets.inverse_transform(y_test_full_np)
    actual_out_sample_df = pd.DataFrame(actual_out_sample_targets_unscaled,
                                        index=test_data_indices[lookback:lookback+len(y_test_full_np)],
                                        columns=config['tickers'])


    for i in range(len(actual_test_scaled_features_values)):
        model.eval()
        with torch.no_grad():
            input_seq_np = np.array(current_history_scaled[-lookback:]).reshape(1, lookback, input_dim)
            input_seq_torch = torch.from_numpy(input_seq_np).float().to(device)
            pred_scaled = model(input_seq_torch).cpu().numpy()
        
        if pred_scaled.ndim == 1:
            pred_scaled = pred_scaled.reshape(-1,1)

        pred_unscaled = scaler_targets.inverse_transform(pred_scaled)
        out_sample_preds_list.append(pred_unscaled[0])
        
        # Use actual future values for the next prediction's history (Teacher Forcing like for multi-step if it were generation)
        # For single step prediction, this means using the true observed features for the next step's input history.
        current_history_scaled.append(actual_test_scaled_features_values[i]) 
        if len(current_history_scaled) > lookback: # Maintain fixed window size for history
            current_history_scaled.pop(0)

    out_sample_preds_df = pd.DataFrame(out_sample_preds_list, 
                                       index=test_data_indices[:len(out_sample_preds_list)], 
                                       columns=config['tickers'])

    # Plotting predictions vs actuals for each ticker
    for ticker in config['tickers']:
        plt.figure(figsize=(14, 7))
        
        # In-sample plot
        plt.subplot(2, 1, 1)
        plt.plot(actual_in_sample_df.index, actual_in_sample_df[ticker], label='Actual In-Sample')
        plt.plot(in_sample_preds_df.index, in_sample_preds_df[ticker], label='Predicted In-Sample', linestyle='--')
        plt.title(f'{model_name.upper()} In-Sample Predictions vs Actuals for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True)

        # Out-of-sample plot
        plt.subplot(2, 1, 2)
        # Align actual_out_sample_df with out_sample_preds_df for plotting
        common_out_sample_index = actual_out_sample_df.index.intersection(out_sample_preds_df.index)
        if not common_out_sample_index.empty:
            plt.plot(actual_out_sample_df.loc[common_out_sample_index, ticker], label='Actual Out-of-Sample')
            plt.plot(out_sample_preds_df.loc[common_out_sample_index, ticker], label='Predicted Out-of-Sample', linestyle='--')
        else:
            print(f"Warning: No common index for out-of-sample plotting for ticker {ticker}. Skipping plot.")
        plt.title(f'{model_name.upper()} Out-of-Sample Predictions vs Actuals for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f'{model_name}_predictions_vs_actuals_{ticker}.png'))
        plt.close()
        print(f"Predictions vs actuals plot for {ticker} saved to {os.path.join(plot_path, f'{model_name}_predictions_vs_actuals_{ticker}.png')}")

    return in_sample_preds_df, out_sample_preds_df