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
from .ls_cnnlstm_attention import LongShortCNNLSTMWithAttention

from .ls_loss import LS_Loss, SimpleMSELoss

from torch.utils.data import DataLoader, TensorDataset


def create_model(model_name, input_size, hidden_size, num_layers, output_size,
                  dropout=0.2, long_term_length=None, short_term_length=None):
    if model_name == 'GRU':
        return GRUModel(input_size['single'], hidden_size, num_layers, output_size['single'], dropout)
    elif model_name == 'LSTM':
        return LSTMModel(input_size['single'], hidden_size, num_layers, output_size['single'], dropout)
    elif model_name == 'LSTM-Att':
        return LSTMWithAttention(input_size['single'], hidden_size, num_layers, output_size['single'], dropout)
    elif model_name == 'CNNLSTM':
        return CNNLSTMModel(input_size['single'], hidden_size, num_layers, output_size['single'], dropout)
    elif model_name == 'LS_CNNLSTM': 
        return LongShortCNNLSTM(
            long_input_size=input_size['long'],
            short_input_size=input_size['short'],
            hidden_size=hidden_size,
            num_layers=num_layers,
            long_output_size=output_size['long'],
            short_output_size=output_size['short'],
            long_term_length=long_term_length,
            short_term_length=short_term_length,
            dropout=dropout
        )
    elif model_name == 'LS_CNNLSTM_Att': 
        return LongShortCNNLSTMWithAttention(
            long_input_size=input_size['long'],
            short_input_size=input_size['short'],
            hidden_size=hidden_size,
            num_layers=num_layers,
            long_output_size=output_size['long'],
            short_output_size=output_size['short'], 
            long_term_length=long_term_length, 
            short_term_length=short_term_length)
    else:
        raise ValueError(f"Model {model_name} is not recognized.")

def oversample_data(long_data, short_data, long_targets, short_targets):
    """
    Oversample short-term data to match the size of long-term data.
    """
    # Check the smaller dataset
    if len(short_data) < len(long_data):
        # Calculate the oversample factor and residual
        oversample_factor = len(long_data) // len(short_data)
        residual_samples = len(long_data) % len(short_data)

        # Oversample short data
        short_data = torch.cat([short_data] * oversample_factor + [short_data[:residual_samples]], dim=0)
        short_targets = torch.cat([short_targets] * oversample_factor + [short_targets[:residual_samples]], dim=0)
    elif len(long_data) < len(short_data):
        # Calculate the oversample factor and residual for long data
        oversample_factor = len(short_data) // len(long_data)
        residual_samples = len(short_data) % len(long_data)

        # Oversample long data
        long_data = torch.cat([long_data] * oversample_factor + [long_data[:residual_samples]], dim=0)
        long_targets = torch.cat([long_targets] * oversample_factor + [long_targets[:residual_samples]], dim=0)
    
    # Ensure shapes match after oversampling
    assert len(long_data) == len(short_data), "Oversampling failed to match dataset sizes!"
    assert len(long_targets) == len(short_targets), "Oversampling failed to match target sizes!"

    return long_data, short_data, long_targets, short_targets
    

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

def train_for_short_term_forecast(model, model_name, train_sequences, train_targets, 
                                         eval_sequences, eval_targets, model_path, num_epochs=100, 
                                         batch_size=64, learning_rate=0.001, patience=5):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_dataset = torch.utils.data.TensorDataset(
        train_sequences.clone().detach().to(torch.float32),
        train_targets.clone().detach().to(torch.float32)
    )
    eval_dataset = torch.utils.data.TensorDataset(
        eval_sequences.clone().detach().to(torch.float32),
        eval_targets.clone().detach().to(torch.float32)
    )
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #, weight_decay=1e-5)

    best_val_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # tqdm을 사용하여 학습 진행도 표시
        with tqdm(train_loader, unit="batch", leave=True) as tepoch:
            for sequences_batch, targets_batch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
                
                # Move batch to GPU
                sequences_batch, targets_batch = sequences_batch.to(device), targets_batch.to(device)
                
                # Forward pass
                if 'Att' in model_name:
                    outputs, _ = model(sequences_batch) 
                else:
                    outputs = model(sequences_batch)

                loss = criterion(outputs.squeeze(), targets_batch)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 현재 배치 손실을 누적
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        # Epoch 단위로 평균 손실 계산
        train_loss = running_loss / len(train_loader)

        # Evaluation on the validation set
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for sequences_batch, targets_batch in eval_loader:
                # Move batch to GPU
                sequences_batch, targets_batch = sequences_batch.to(device), targets_batch.to(device)

                if 'Att' in model_name:
                    val_outputs, _ = model(sequences_batch) 
                else:
                    val_outputs = model(sequences_batch)
                loss = criterion(val_outputs.squeeze(), targets_batch)
                eval_loss += loss.item()

        eval_loss /= len(eval_loader)
        # print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {eval_loss:.7f}')

        # Check if the validation loss improved
        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            torch.save(model.state_dict(), model_path)
            print(f"Validation loss improved. Model saved at epoch {epoch+1}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping applied at epoch {epoch+1}")
            break


def train_for_long_term_forecast(model, model_name, train_data, train_targets, 
                                       eval_data, eval_targets, model_path, num_epochs=100, 
                                       batch_size=64, learning_rate=0.001, patience=5, 
                                       oversample_eval=False, alpha=0.3, beta=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_long, train_short = train_data
    train_targets_long, train_targets_short = train_targets

    eval_long, eval_short = eval_data
    eval_targets_long, eval_targets_short = eval_targets

    # Oversample training data
    print("Perform oversampling on training data...")
    train_long, train_short, train_targets_long, train_targets_short = oversample_data(
        train_long, train_short, train_targets_long, train_targets_short
    )

    # Optionally oversample evaluation data
    if oversample_eval:
        print("Perform oversampling on evaluation data...")
        eval_long, eval_short, eval_targets_long, eval_targets_short = oversample_data(
            eval_long, eval_short, eval_targets_long, eval_targets_short
        )
    else:
        min_eval_size = min(len(eval_long), len(eval_short))
        eval_long = eval_long[:min_eval_size]
        eval_short = eval_short[:min_eval_size]
        eval_targets_long = eval_targets_long[:min_eval_size]
        eval_targets_short = eval_targets_short[:min_eval_size]

    # DataLoaders
    train_dataset = torch.utils.data.TensorDataset(train_long, train_short, train_targets_long, train_targets_short)
    eval_dataset = torch.utils.data.TensorDataset(eval_long, eval_short, eval_targets_long, eval_targets_short)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)

    # Loss function and optimizer
    # criterion = nn.MSELoss()
    criterion = SimpleMSELoss(alpha=alpha, beta=beta)
    # criterion = LS_Loss(alpha=1.0, beta=0.3, gamma=0.2, lambda_=0.1, eta=0.01)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in tqdm(range(num_epochs), leave=True):
        model.train()
        train_loss = 0.0

        for long_batch, short_batch, target_long, target_short in train_loader:
            long_batch, short_batch = long_batch.to(device), short_batch.to(device)
            target_long, target_short = target_long.to(device), target_short.to(device)

            if 'Att' in model_name:
                output_long, output_short, _ = model(long_batch, short_batch)
            else:
                output_long, output_short = model(long_batch, short_batch)
            
            # loss = criterion(output_long, target_long) + criterion(output_short, target_short)
            loss = criterion(output_long, target_long, output_short, target_short)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for long_batch, short_batch, target_long, target_short in eval_loader:
                long_batch, short_batch = long_batch.to(device), short_batch.to(device)
                target_long, target_short = target_long.to(device), target_short.to(device)

                if 'Att' in model_name:
                    output_long, output_short, _ = model(long_batch, short_batch)
                else:
                    output_long, output_short = model(long_batch, short_batch)
                
                # loss = criterion(output_long, target_long) + criterion(output_short, target_short)
                loss = criterion(output_long, target_long, output_short, target_short)
                
                eval_loss += loss.item()

        eval_loss /= len(eval_loader)
        scheduler.step(eval_loss)
        
        # print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {eval_loss:.4f}")

        if not(criterion.alpha == 1.0 and criterion.beta == 1.0):
            criterion.alpha = min(1.0, criterion.alpha + 0.05)  # Increase alpha
            criterion.beta = max(0.1, criterion.beta - 0.05)   # Decrease beta
        
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


def evaluate_for_short_term_forecast(model, eval_sequences, eval_targets, model_name="", batch_size=64):
    """
    Evaluate a given model with R², SMAPE, and MASE.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    eval_dataset = torch.utils.data.TensorDataset(torch.tensor(eval_sequences, dtype=torch.float32),
                                                  torch.tensor(eval_targets, dtype=torch.float32))
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for sequences_batch, targets_batch in eval_loader:
            sequences_batch, targets_batch = sequences_batch.to(device), targets_batch.to(device)
            
            if 'Att' in model_name:
                outputs, _ = model(sequences_batch)
            else:
                outputs = model(sequences_batch)

            # 필요에 따라 squeeze() 적용
            if outputs.shape[-1] == 1:
                outputs = outputs.squeeze(-1)

            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets_batch.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    if np.isnan(all_outputs).any() or np.isnan(all_targets).any():
        print("Warning: NaN detected in evaluation data. Returning default scores.")
        return {"R2": 0.0, "Adjusted R2": 0.0, "SMAPE": 0.0, "MASE": 0.0}

    try:
        r2 = r2_score(all_targets, all_outputs)
        n = len(all_targets)
        k = eval_sequences.shape[-1]
        adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - k - 1)
        smape_score = smape(all_targets, all_outputs)
        mase_score = mase(all_targets, all_outputs)
    except Exception as e:
        print(f"Error during metrics calculation: {e}. Returning default scores.")
        return {"R2": 0.0, "Adjusted R2": 0.0, "SMAPE": 0.0, "MASE": 0.0}

    # # Metrics calculation
    # r2 = r2_score(all_targets, all_outputs)
    # n = len(all_targets)
    # k = eval_sequences.shape[-1]
    # adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - k - 1)
    # smape_score = smape(all_targets, all_outputs)
    # mase_score = mase(all_targets, all_outputs)

    # Print results
    print(f"R² Score: {r2:.4f}")
    print(f"Adjusted R²: {adjusted_r2:.4f}")
    print(f"SMAPE: {smape_score:.2f}")
    print(f"MASE: {mase_score:.4f}")

    return {"R2": r2, "Adjusted R2": adjusted_r2, "SMAPE": smape_score, "MASE": mase_score}


def evaluate_for_long_term_forecast(model, eval_data, eval_targets, model_name="", 
                    batch_size=64, oversample_eval=False):
    """
    Evaluate the model using R², SMAPE, and MASE metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    eval_long, eval_short = eval_data
    eval_targets_long, eval_targets_short = eval_targets

    # Optionally oversample evaluation data
    if oversample_eval:
        eval_long, eval_short, eval_targets_long, eval_targets_short = oversample_data(
            eval_long, eval_short, eval_targets_long, eval_targets_short
        )

    # Adjust evaluation data sizes if oversample is not applied
    if not oversample_eval:
        min_eval_size = min(len(eval_long), len(eval_short))
        eval_long = eval_long[:min_eval_size]
        eval_short = eval_short[:min_eval_size]
        eval_targets_long = eval_targets_long[:min_eval_size]
        eval_targets_short = eval_targets_short[:min_eval_size]

    eval_dataset =  torch.utils.data.TensorDataset(eval_long, eval_short, eval_targets_long, eval_targets_short)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)

    all_outputs_long = []
    all_outputs_short = []
    all_targets_long = []
    all_targets_short = []

    with torch.no_grad():
        for long_batch, short_batch, long_target, short_target in eval_loader:
            long_batch, short_batch = long_batch.to(device), short_batch.to(device)
            long_target, short_target = long_target.to(device), short_target.to(device)

            if 'Att' in model_name:
                output_long, output_short, attention_weights = model(long_batch, short_batch)
            else:
                output_long, output_short = model(long_batch, short_batch)

            all_outputs_long.append(output_long.cpu().numpy())
            all_outputs_short.append(output_short.cpu().numpy())
            all_targets_long.append(long_target.cpu().numpy())
            all_targets_short.append(short_target.cpu().numpy())

    all_outputs_long = np.concatenate(all_outputs_long, axis=0)
    all_outputs_short = np.concatenate(all_outputs_short, axis=0)
    all_targets_long = np.concatenate(all_targets_long, axis=0)
    all_targets_short = np.concatenate(all_targets_short, axis=0)

    # Metrics calculation for long-term and short-term predictions
    r2_short = r2_score(all_targets_short, all_outputs_short)
    n = len(all_targets_short)
    k = eval_data[0].shape[-1]
    adjusted_r2 = 1 - ((1 - r2_short) * (n - 1)) / (n - k - 1)
    smape_short = smape(all_targets_short, all_outputs_short)
    mase_short = mase(all_targets_short, all_outputs_short)


    # Print results
    print(f"R² Score: {r2_short:.4f}")
    print(f"Adjusted R²: {adjusted_r2:.4f}")
    print(f"SMAPE: {smape_short:.2f}")
    print(f"MASE: {mase_short:.4f}")
    
    return {
        "R2 Short": r2_short,
        "Adjusted R2": adjusted_r2,
        "SMAPE Short": smape_short,
        "MASE Short": mase_short
    }


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







