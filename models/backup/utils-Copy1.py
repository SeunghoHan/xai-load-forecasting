import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score

from .lstm import LSTMModel
from .gru import GRUModel
from .cnnlstm import CNNLSTMModel
from .lstm_attention import AttentionLSTMModel, LSTMWithAttention
from .ls_cnnlstm import LongShortCNNLSTM


def create_model(model_name, input_size, hidden_size, num_layers, output_size, dropout=0.2, long_term_days=None, short_term_days=None):
    if model_name == 'GRU':
        return GRUModel(input_size, hidden_size, num_layers, output_size, dropout)
    elif model_name == 'LSTM':
        return LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
    elif model_name == 'LSTM-Att':
        return LSTMWithAttention(input_size, hidden_size, num_layers, output_size, dropout)
    elif model_name == 'CNNLSTM':
        return CNNLSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
    elif model_name == 'LS_CNNLSTM': 
        return LongShortCNNLSTM(
            long_input_size=input_size['long'],
            short_input_size=input_size['short'],
            hidden_size=hidden_size,
            num_layers=num_layers,
            long_output_size=output_size['long'],
            short_output_size=output_size['short'],
            dropout=dropout
        )
    else:
        raise ValueError(f"Model {model_name} is not recognized.")

def train_and_evaluate(model, model_name, train_data, train_targets, 
                       eval_data, eval_targets, model_path, num_epochs=100, 
                       batch_size=64, learning_rate=0.001, patience=5):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Determine if the model is Multi-Term or Single-Term
    is_multi_input = isinstance(train_data, tuple)  # Check if input is (long, short)

    if is_multi_input:
        train_dataset = torch.utils.data.TensorDataset(
            train_data[0].clone().detach() if isinstance(train_data[0], torch.Tensor) else torch.tensor(train_data[0], dtype=torch.float32),
            train_data[1].clone().detach() if isinstance(train_data[1], torch.Tensor) else torch.tensor(train_data[1], dtype=torch.float32),
            train_targets.clone().detach() if isinstance(train_targets, torch.Tensor) else torch.tensor(train_targets, dtype=torch.float32)
        )
        eval_dataset = torch.utils.data.TensorDataset(
            eval_data[0].clone().detach() if isinstance(eval_data[0], torch.Tensor) else torch.tensor(eval_data[0], dtype=torch.float32),
            eval_data[1].clone().detach() if isinstance(eval_data[1], torch.Tensor) else torch.tensor(eval_data[1], dtype=torch.float32),
            eval_targets.clone().detach() if isinstance(eval_targets, torch.Tensor) else torch.tensor(eval_targets, dtype=torch.float32)
        )
    else:
        train_dataset = torch.utils.data.TensorDataset(
            train_data.clone().detach() if isinstance(train_data, torch.Tensor) else torch.tensor(train_data, dtype=torch.float32),
            train_targets.clone().detach() if isinstance(train_targets, torch.Tensor) else torch.tensor(train_targets, dtype=torch.float32)
        )
        eval_dataset = torch.utils.data.TensorDataset(
            eval_data.clone().detach() if isinstance(eval_data, torch.Tensor) else torch.tensor(eval_data, dtype=torch.float32),
            eval_targets.clone().detach() if isinstance(eval_targets, torch.Tensor) else torch.tensor(eval_targets, dtype=torch.float32)
        )

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)


    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            if is_multi_input:
                long_batch, short_batch, target_batch = batch
                long_batch, short_batch, target_batch = (
                    long_batch.to(device), short_batch.to(device), target_batch.to(device)
                )
                outputs = model(long_batch, short_batch)
            else:
                sequences_batch, target_batch = batch
                sequences_batch, target_batch = sequences_batch.to(device), target_batch.to(device)
                outputs = model(sequences_batch)

            loss = criterion(outputs.squeeze(), target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Validation loop
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                if is_multi_input:
                    long_batch, short_batch, target_batch = batch
                    long_batch, short_batch, target_batch = (
                        long_batch.to(device), short_batch.to(device), target_batch.to(device)
                    )
                    val_outputs = model(long_batch, short_batch)
                else:
                    sequences_batch, target_batch = batch
                    sequences_batch, target_batch = sequences_batch.to(device), target_batch.to(device)
                    val_outputs = model(sequences_batch)

                loss = criterion(val_outputs.squeeze(), target_batch)
                eval_loss += loss.item()

        eval_loss /= len(eval_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {eval_loss:.4f}')

        # Early stopping
        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            torch.save(model.state_dict(), model_path)
            print(f"Validation loss improved. Model saved at epoch {epoch+1}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


def train_and_evaluate_for_multiterm(model, model_name, train_data, train_targets, 
                       eval_data, eval_targets, model_path, num_epochs=100, 
                       batch_size=64, learning_rate=0.001, patience=5):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Determine if the model is Multi-Term or Single-Term
    is_multi_input = isinstance(train_data, tuple)  # Check if input is (long, short)

    if is_multi_input:
        train_dataset = torch.utils.data.TensorDataset(
            train_data[0], train_data[1], train_targets
        )
        eval_dataset = torch.utils.data.TensorDataset(
            eval_data[0], eval_data[1], eval_targets
        )
    else:
        train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
        eval_dataset = torch.utils.data.TensorDataset(eval_data, eval_targets)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    best_val_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            if is_multi_input:
                long_batch, short_batch, target_batch = (t.to(device) for t in batch)
                outputs = model(long_batch, short_batch)
            else:
                sequences_batch, target_batch = (t.to(device) for t in batch)
                outputs = model(sequences_batch)

            loss = criterion(outputs.squeeze(), target_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Validation Phase
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                if is_multi_input:
                    long_batch, short_batch, target_batch = (t.to(device) for t in batch)
                    val_outputs = model(long_batch, short_batch)
                else:
                    sequences_batch, target_batch = (t.to(device) for t in batch)
                    val_outputs = model(sequences_batch)

                loss = criterion(val_outputs.squeeze(), target_batch)
                eval_loss += loss.item()

        eval_loss /= len(eval_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {eval_loss:.4f}')

        # Early Stopping & Model Saving
        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            torch.save(model.state_dict(), model_path)
            print(f"Validation loss improved. Model saved at epoch {epoch+1}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Reduce Learning Rate on Plateau
        scheduler.step(eval_loss)

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


def evaluate_model(model, eval_data, eval_targets, model_name="", batch_size=64):
    """
    Evaluate a given model with R², Adjusted R², SMAPE, and MASE.
    Supports both Single-Term (e.g., LSTM) and Multi-Term (e.g., MT_CNN_LSTM) models.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Check if the model is Multi-Term or Single-Term
    is_multi_input = isinstance(eval_data, tuple)  # Tuple indicates Multi-Term data

    if is_multi_input:
        eval_dataset = torch.utils.data.TensorDataset(
            torch.tensor(eval_data[0], dtype=torch.float32),
            torch.tensor(eval_data[1], dtype=torch.float32),
            torch.tensor(eval_targets, dtype=torch.float32)
        )
    else:
        eval_dataset = torch.utils.data.TensorDataset(
            torch.tensor(eval_data, dtype=torch.float32),
            torch.tensor(eval_targets, dtype=torch.float32)
        )

    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch in eval_loader:
            if is_multi_input:
                long_batch, short_batch, target_batch = batch
                long_batch, short_batch, target_batch = long_batch.to(device), short_batch.to(device), target_batch.to(device)
                outputs = model(long_batch, short_batch)
            else:
                sequences_batch, target_batch = batch
                sequences_batch, target_batch = sequences_batch.to(device), target_batch.to(device)
                outputs = model(sequences_batch)

            # Handle outputs
            if outputs.shape[-1] == 1:
                outputs = outputs.squeeze(-1)

            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(target_batch.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Metrics calculation
    r2 = r2_score(all_targets, all_outputs)
    n = len(all_targets)
    k = eval_data[0].shape[-1] if is_multi_input else eval_data.shape[-1]
    adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - k - 1)
    smape_score = smape(all_targets, all_outputs)
    mase_score = mase(all_targets, all_outputs)

    # Print results
    print(f"R² Score: {r2:.4f}")
    print(f"Adjusted R²: {adjusted_r2:.4f}")
    print(f"SMAPE: {smape_score:.2f}")
    print(f"MASE: {mase_score:.4f}")

    return {"R2": r2, "Adjusted R2": adjusted_r2, "SMAPE": smape_score, "MASE": mase_score}


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path,  weights_only=True))
    return model

# SMAPE 정의
def smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(numerator / denominator) * 100
    return smape

# MASE 정의
def mase(y_true, y_pred, naive_forecast=None):
    mae_pred = np.mean(np.abs(y_true - y_pred))
    if naive_forecast is None:
        naive_forecast = np.roll(y_true, shift=1)
        naive_forecast[0] = y_true[0]  # Shift로 발생하는 문제 해결
    mae_naive = np.mean(np.abs(y_true - naive_forecast))
    mase = mae_pred / mae_naive
    return mase



def evaluate_ensemble(models, eval_sequences, eval_targets, batch_size=64):
    """
    Evaluate ensemble model using simple averaging.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_models = len(models)

    # DataLoader for evaluation data
    eval_dataset = torch.utils.data.TensorDataset(torch.tensor(eval_sequences, dtype=torch.float32),
                                                  torch.tensor(eval_targets, dtype=torch.float32))
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for sequences_batch, targets_batch in eval_loader:
            sequences_batch, targets_batch = sequences_batch.to(device), targets_batch.to(device)

            # Model predictions
            model_outputs = []
            for model in models:
                model.to(device)
                model.eval()
                if isinstance(model, dict):  # For seasonal models
                    season = determine_season(sequences_batch)  # Custom function to determine season
                    outputs = model[season].predict(sequences_batch)
                else:
                    outputs = model(sequences_batch)
                if outputs.shape[-1] == 1:
                    outputs = outputs.squeeze(-1)
                model_outputs.append(outputs.cpu().numpy())

            # Ensemble prediction: simple average
            ensemble_output = np.mean(np.array(model_outputs), axis=0)

            # Append outputs and targets
            all_outputs.append(ensemble_output)
            all_targets.append(targets_batch.cpu().numpy())

    # Convert lists to numpy arrays
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Evaluate metrics
    r2 = r2_score(all_targets, all_outputs)
    smape_score = smape(all_targets, all_outputs)
    mase_score = mase(all_targets, all_outputs)

    print(f"\tR² Score: {r2:.4f}")
    print(f"\tSMAPE: {smape_score:.2f}%")
    print(f"\tMASE: {mase_score:.4f}")

    return {"R2": r2, "SMAPE": smape_score, "MASE": mase_score}



# #############
# ## 모델별 학습 하이퍼파라미터 세팅 (데이터 24*7, 24*30)
# params = { 
#     'LSTM' : {
#         hidden_size=128, 
#         num_layer=2, 
#         dropout=0.3, 
#         num_epochs=200,
#         batch_size=64, 
#         learning_rate=0.001, 
#         patience=5
#     },
#     'GRU' : {
#             hidden_size=128, 
#             num_layer=2, 
#             dropout=0.3, 
#             num_epochs=200,
#             batch_size=64, 
#             learning_rate=0.001, 
#             patience=5
#         }
# }







