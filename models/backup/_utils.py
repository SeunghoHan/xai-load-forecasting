import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score

from .lstm import LSTMModel
from .gru import GRUModel
from .cnnlstm import CNNLSTMModel
from .lstm_attention import AttentionLSTMModel, LSTMWithAttention

def create_model(model_name, input_size, hidden_size, num_layers, output_size, dropout=0.2, long_term_rows=None, long_term_cols=None, short_term_length=None):
    if model_name == 'GRU':
        return GRUModel(input_size, hidden_size, num_layers, output_size, dropout)
    elif model_name == 'LSTM':
        return LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
    elif model_name == 'LSTM-Att':
        return LSTMWithAttention(input_size, hidden_size, num_layers, output_size, dropout)
    elif model_name == 'CNNLSTM':
        return CNNLSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
    elif model_name == 'MT_CNN_LSTM': 
        return MultiTermCNNLSTM(input_size=input_size, 
                                long_term_rows=long_term_rows, 
                                long_term_cols=long_term_cols, 
                                short_term_length=short_term_length, 
                                hidden_size=hidden_size, 
                                output_size=output_size)
    else:
        raise ValueError(f"Model {model_name} is not recognized. Please use 'GRU' or 'LSTM'.")
        

def train_and_evaluate(model, model_name, train_sequences, train_targets, 
                       eval_sequences, eval_targets, model_path, num_epochs=100, 
                       batch_size=64, learning_rate=0.001, patience=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Data loader
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_sequences, dtype=torch.float32), torch.tensor(train_targets, dtype=torch.float32))
    eval_dataset = torch.utils.data.TensorDataset(torch.tensor(eval_sequences, dtype=torch.float32), torch.tensor(eval_targets, dtype=torch.float32))
    
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
        with tqdm(train_loader, unit="batch") as tepoch:
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {eval_loss:.7f}')

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
            
# def evaluate_r2_score(model, eval_sequences, eval_targets, model_name, batch_size=64):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     model.eval()  # 평가 모드로 전환

#     eval_dataset = torch.utils.data.TensorDataset(torch.tensor(eval_sequences, dtype=torch.float32), 
#                                                   torch.tensor(eval_targets, dtype=torch.float32))
#     eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)

#     all_outputs = []
#     all_targets = []

#     with torch.no_grad():
#         for sequences_batch, targets_batch in eval_loader:
#             sequences_batch, targets_batch = sequences_batch.to(device), targets_batch.to(device)
            
#             if 'Att' in model_name:
#                 outputs, _ = model(sequences_batch)
#             else:
#                 outputs = model(sequences_batch)

#             # 필요에 따라 squeeze() 적용
#             if outputs.shape[-1] == 1:
#                 outputs = outputs.squeeze(-1)

#             # GPU에서 CPU로 이동 및 리스트에 저장
#             all_outputs.append(outputs.cpu().numpy())
#             all_targets.append(targets_batch.cpu().numpy())

#     # 리스트를 numpy 배열로 변환
#     all_outputs = np.concatenate(all_outputs, axis=0)
#     all_targets = np.concatenate(all_targets, axis=0)

#     # R² 스코어 계산
#     r2 = r2_score(all_targets, all_outputs)
#     print(f"R² Score: {r2:.4f}")

#     return r2

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

# 모델 평가 함수
def evaluate_model(model, eval_sequences, eval_targets, model_name="", batch_size=64):
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

    # 평가지표 계산
    r2 = r2_score(all_targets, all_outputs)
    smape_score = smape(all_targets, all_outputs)
    mase_score = mase(all_targets, all_outputs)

    # 결과 출력
    print(f"\tR² Score: {r2:.4f}")
    print(f"\tSMAPE: {smape_score:.2f}")
    print(f"\tMASE: {mase_score:.4f}")

    return {"R2": r2, "SMAPE": smape_score, "MASE": mase_score}


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







