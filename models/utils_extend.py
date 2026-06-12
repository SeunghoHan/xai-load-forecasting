import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score

from .lstm_extend import LSTMModel
from .gru_extend import GRUModel
from .cnnlstm_extend import CNNLSTMModel
from .ls_cnnlstm_extend import LongShortCNNLSTM
from .ls_cnnlstm_attn_extend import LongShortCNNLSTMWithAttention
from .transformer import SinglePatchTST, LongShortDualTST_Hier

from .ls_loss import LS_Loss, SimpleMSELoss

from torch.utils.data import DataLoader, TensorDataset

def create_model(model_name,
                 # 단일 스트림용
                 input_size_num=None,          # int, e.g. X.shape[2]
                 output_size=None,             # int, prediction_length
                 # 장·단기용
                 long_input_size_num=None,     # int
                 short_input_size_num=None,    # int
                 long_output_size=None,        # int
                 short_output_size=None,       # int
                 # 공통 하이퍼파라미터
                 hidden_size=128,
                 num_layers=2,
                 dropout=0.2,
                 # UMass(아이콘) 옵션
                 icon_vocab_size=None,
                 icon_embed_dim=8,
                 # 기타 옵션
                 bidirectional=False,
                 long_term_length=None,
                 short_term_length=None):
    """
    단일 스트림 모델: (model_name in {'GRU','LSTM','LSTM-Att','CNNLSTM'})
        - input_size_num (int), output_size (int) 필수
    장·단기 모델: (model_name in {'LS_CNNLSTM','LS_CNNLSTM_Att'})
        - long_input_size_num, short_input_size_num, long_output_size, short_output_size 필수
    """
    # 단일 스트림
    if model_name == 'GRU':
        return GRUModel(
            input_size_num=input_size_num,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            icon_vocab_size=icon_vocab_size,
            icon_embed_dim=icon_embed_dim
        )
    elif model_name == 'LSTM':
        return LSTMModel(
            input_size_num=input_size_num,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            icon_vocab_size=icon_vocab_size,
            icon_embed_dim=icon_embed_dim,
            bidirectional=bidirectional
        )
    elif model_name == 'CNNLSTM':
        return CNNLSTMModel(
            input_size_num=input_size_num,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            icon_vocab_size=icon_vocab_size,
            icon_embed_dim=icon_embed_dim
        )
    elif model_name in 'Transformer':
        # 단일 스트림(단기) 트랜스포머를 SinglePatchTST로 매핑
        # input_size_num, output_size는 load_trained_model()에서 이미 전달됨
        return SinglePatchTST(
            num_features=input_size_num,
            pred_len=output_size,
            target_idx=0,                   # 보통 use_total 채널 index
            d_model=hidden_size,            # = cfg hidden_size
            nhead=8,                        # d_model은 8의 배수 권장
            d_ff=hidden_size * 4 if hidden_size < 192 else hidden_size * 3,
            depth=max(1, num_layers),
            patch_len=24,                   # 24(1일) 패치 권장
            stride=12,                      # 절반 겹침
            dropout=dropout if dropout is not None else 0.10,
            use_ci=True                     # EPC 안정성↑; UMass는 False도 실험 가치
        )


    # 장·단기 모델
    elif model_name == 'LS_CNNLSTM':
        return LongShortCNNLSTM(
            long_input_size_num=long_input_size_num,
            short_input_size_num=short_input_size_num,
            long_output_size=long_output_size,
            short_output_size=short_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            icon_vocab_size=icon_vocab_size,   # 모델 내부에서 (아이콘 사용 여부) 분기 지원한다고 가정
            icon_embed_dim=icon_embed_dim,
            long_term_length=long_term_length,
            short_term_length=short_term_length
        )
    elif model_name == 'LS_CNNLSTM_Att':
        return LongShortCNNLSTMWithAttention(
            long_input_size_num=long_input_size_num,
            short_input_size_num=short_input_size_num,
            long_output_size=long_output_size,
            short_output_size=short_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            icon_vocab_size=icon_vocab_size,
            icon_embed_dim=icon_embed_dim,
            long_term_length=long_term_length,
            short_term_length=short_term_length
        )
    elif model_name == 'LS_Transformer':
        # 장·단기 트랜스포머 (두 개의 단일 스트림을 래핑)
        assert long_input_size_num is not None and short_input_size_num is not None
        assert long_term_length is not None and short_term_length is not None
        assert long_output_size is not None and short_output_size is not None
    
        return LongShortDualTST_Hier(
            long_term_length=long_term_length,
            short_term_length=short_term_length,
            short_term_pred_length=short_output_size,     # 최종 7일
            long_term_pred_length=long_output_size,       # 보조 256 등
            num_features_long=long_input_size_num,
            num_features_short=short_input_size_num,
            target_idx=0,
            d_model=hidden_size,
            nhead=8,
            d_ff=hidden_size * 4 if hidden_size < 192 else hidden_size * 3,
            depth=max(1, num_layers),
            patch_len_long=24, stride_long=12,
            patch_len_short=16, stride_short=8,
            dropout=dropout if dropout is not None else 0.15,
            use_ci=True   # EPC 안정성↑; UMass면 False로도 실험 가능
        )
            
    else:
        raise ValueError(f"Model {model_name} is not recognized.")
        

def _repeat_like(x, times, residual):
    parts = [x] * times
    if residual > 0:
        parts.append(x[:residual])
    return torch.cat(parts, dim=0)

def oversample_align(long_pack, short_pack):
    """
    두 팩의 배치수를 큰 쪽에 맞춰 오버샘플.

    long_pack  = {'Xn': Float [N, T_l, D_l], 'Xi': Long [N, T_l] or None, 'y': Float [N, P_l]}
    short_pack = {'Xn': Float [M, T_s, D_s], 'Xi': Long [M, T_s] or None, 'y': Float [M, P_s]}
    """
    nL = long_pack['Xn'].shape[0]
    nS = short_pack['Xn'].shape[0]
    if nL == nS:
        return long_pack, short_pack

    # 작은 쪽을 큰 쪽에 맞춘다
    if nL > nS:
        times, residual = divmod(nL, nS)
        short_pack['Xn'] = _repeat_like(short_pack['Xn'], times, residual)
        short_pack['y']  = _repeat_like(short_pack['y'],  times, residual)
        if short_pack['Xi'] is not None:
            short_pack['Xi'] = _repeat_like(short_pack['Xi'], times, residual)
    else:
        times, residual = divmod(nS, nL)
        long_pack['Xn'] = _repeat_like(long_pack['Xn'], times, residual)
        long_pack['y']  = _repeat_like(long_pack['y'],  times, residual)
        if long_pack['Xi'] is not None:
            long_pack['Xi'] = _repeat_like(long_pack['Xi'], times, residual)

    # 안전 체크
    assert long_pack['Xn'].shape[0] == short_pack['Xn'].shape[0]
    assert long_pack['y' ].shape[0] == short_pack['y' ].shape[0]
    if (long_pack['Xi'] is not None) and (short_pack['Xi'] is not None):
        assert long_pack['Xi'].shape[0] == short_pack['Xi'].shape[0]

    return long_pack, short_pack

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path,  weights_only=True))
    return model

# SMAPE 정의
# def smape(y_true, y_pred):
#     numerator = np.abs(y_true - y_pred)
#     denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
#     smape = np.mean(numerator / denominator) * 100
#     return smape

def smape(y_true, y_pred, eps=1e-6):
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    return 100.0 * np.mean(np.abs(y_pred - y_true) / denom)

# MASE 정의
def mase(y_true, y_pred, naive_forecast=None):
    mae_pred = np.mean(np.abs(y_true - y_pred))
    if naive_forecast is None:
        naive_forecast = np.roll(y_true, shift=1)
        naive_forecast[0] = y_true[0]  # Shift로 발생하는 문제 해결
    mae_naive = np.mean(np.abs(y_true - naive_forecast))
    mase = mae_pred / mae_naive
    return mase

# 꼭 상단 어딘가에 추가

def train_for_short_term_forecast(model, model_name,
                                  train_sequences, train_targets,
                                  eval_sequences, eval_targets,
                                  model_path, num_epochs=100,
                                  batch_size=64, learning_rate=0.001, patience=5):
    """
    EPC:
      - train_sequences: FloatTensor [N, T, D]
      - train_targets  : FloatTensor [N, P]
    UMass:
      - train_sequences: (Xn, Xi)
          Xn: FloatTensor [N, T, D]
          Xi: LongTensor  [N, T]
      - train_targets  : FloatTensor [N, P]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # --- 데이터셋 구성 ---
    is_umass = isinstance(train_sequences, (tuple, list))
    if is_umass:
        Xn_tr, Xi_tr = train_sequences
        Xn_ev, Xi_ev = eval_sequences
        train_dataset = torch.utils.data.TensorDataset(Xn_tr.float(), Xi_tr.long(), train_targets.float())
        eval_dataset  = torch.utils.data.TensorDataset(Xn_ev.float(), Xi_ev.long(), eval_targets.float())
    else:
        train_dataset = torch.utils.data.TensorDataset(train_sequences.float(), train_targets.float())
        eval_dataset  = torch.utils.data.TensorDataset(eval_sequences.float(),  eval_targets.float())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader  = torch.utils.data.DataLoader(eval_dataset,  batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # 🔹 에폭 내 배치 진행도 표시
        with tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False) as tepoch:
            for i, batch in enumerate(tepoch, 1):
                optimizer.zero_grad()
                if is_umass:
                    x_num, x_icon, y = [t.to(device) for t in batch]
                    out = model(x_num, x_icon)  # (숫자, 아이콘)
                else:
                    x_num, y = [t.to(device) for t in batch]
                    out = model(x_num) if 'Att' not in model_name else model(x_num)[0]

                loss = criterion(out.squeeze(), y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                avg_loss = running_loss / i
                tepoch.set_postfix(batch_loss=f"{loss.item():.6f}", avg=f"{avg_loss:.6f}")

        train_loss = running_loss / max(1, len(train_loader))

        # ---- eval ----
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            # 🔹 검증도 짧은 진행바로 표시
            with tqdm(eval_loader, desc=f"[Eval ] Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False) as eepoch:
                for i, batch in enumerate(eepoch, 1):
                    if is_umass:
                        x_num, x_icon, y = [t.to(device) for t in batch]
                        val_out = model(x_num, x_icon)
                    else:
                        x_num, y = [t.to(device) for t in batch]
                        val_out = model(x_num) if 'Att' not in model_name else model(x_num)[0]

                    loss = criterion(val_out.squeeze(), y).item()
                    eval_loss += loss
                    eepoch.set_postfix(avg=f"{(eval_loss/i):.6f}")

        eval_loss /= max(1, len(eval_loader))

        # 에폭 요약 로그
        tqdm.write(f"Epoch {epoch+1:03d}/{num_epochs} | Train {train_loss:.6f} | Val {eval_loss:.6f}")

        # ---- checkpoint & early stop ----
        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            torch.save(model.state_dict(), model_path)
            tqdm.write(f"[Short] ✓ Epoch {epoch+1} improved → saved (val={eval_loss:.8f}, train={train_loss:.8f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            tqdm.write(f"[Short] no improvement at epoch {epoch+1} (best={best_val_loss:.8f})")
            if epochs_no_improve >= patience:
                tqdm.write(f"[Short] Early stopping at epoch {epoch+1} (best val={best_val_loss:.8f})")
                break


def train_for_long_term_forecast(model, model_name, train_data, train_targets, 
                                 eval_data, eval_targets, model_path, num_epochs=100, 
                                 batch_size=64, learning_rate=0.001, patience=5, 
                                 oversample_eval=False, alpha=0.3, beta=1.0):
    """
    EPC:
      train_data      = (XL, XS)                       Float [N, T_l, D_l], [M, T_s, D_s]
      train_targets   = (yL, yS)                       Float [N, P_l],      [M, P_s]
      eval_data       = (XLe, XSe)
      eval_targets    = (yLe, ySe)
    UMass:
      train_data      = ((XnL, XiL), (XnS, XiS))       Xn*: Float [N, T, D], Xi*: Long [N, T]
      train_targets   = (yL, yS)
      eval_data       = ((XnLe, XiLe), (XnSe, XiSe))
      eval_targets    = (yLe, ySe)

    모델 forward 가정:
      EPC   -> model(XL, XS) 혹은 model(XL, XS) → (outL, outS) / (outL, outS, _)
      UMass -> model(XnL, XnS, XiL, XiS) → (outL, outS) / (outL, outS, _)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # -------- 데이터 언패킹 & 유형 판별 --------
    train_long, train_short = train_data
    train_targets_long, train_targets_short = train_targets
    eval_long, eval_short = eval_data
    eval_targets_long, eval_targets_short = eval_targets

    is_umass = isinstance(train_long, (tuple, list)) and isinstance(train_short, (tuple, list))

    # -------- 오버샘플링 (학습) --------
    print("Perform oversampling on training data...")
    if is_umass:
        XnL, XiL = train_long
        XnS, XiS = train_short
        yL,  yS  = train_targets_long, train_targets_short
    
        packL = {'Xn': XnL.float(), 'Xi': XiL.long(), 'y': yL.float()}
        packS = {'Xn': XnS.float(), 'Xi': XiS.long(), 'y': yS.float()}
    
        packL, packS = oversample_align(packL, packS)
    
        # 언팩 후 Dataset
        XnL, XiL, yL = packL['Xn'], packL['Xi'], packL['y']
        XnS, XiS, yS = packS['Xn'], packS['Xi'], packS['y']
        train_dataset = TensorDataset(XnL, XiL, XnS, XiS, yL, yS)
    else:
        XL, XS = train_long.float(), train_short.float()
        yL, yS = train_targets_long.float(), train_targets_short.float()

            
        minN = min(len(XL), len(XS), len(yL), len(yS))
        XL, XS, yL, yS = XL[:minN], XS[:minN], yL[:minN], yS[:minN]
        train_dataset = TensorDataset(XL, XS, yL, yS)
    
        # packL = {'Xn': XL, 'Xi': None, 'y': yL}
        # packS = {'Xn': XS, 'Xi': None, 'y': yS}
    
        # packL, packS = oversample_align(packL, packS)
    
        # XL, yL = packL['Xn'], packL['y']
        # XS, yS = packS['Xn'], packS['y']
        # train_dataset = TensorDataset(XL, XS, yL, yS)

    # -------- 평가셋 정렬/오버샘플 --------
    if oversample_eval:
        print("Perform oversampling on evaluation data...")
        if is_umass:
            XnLe, XiLe = eval_long
            XnSe, XiSe = eval_short
            yLe,  ySe  = eval_targets_long, eval_targets_short
    
            packLe = {'Xn': XnLe.float(), 'Xi': XiLe.long(), 'y': yLe.float()}
            packSe = {'Xn': XnSe.float(), 'Xi': XiSe.long(), 'y': ySe.float()}
    
            packLe, packSe = oversample_align(packLe, packSe)
    
            XnLe, XiLe, yLe = packLe['Xn'], packLe['Xi'], packLe['y']
            XnSe, XiSe, ySe = packSe['Xn'], packSe['Xi'], packSe['y']
            eval_dataset = TensorDataset(XnLe, XiLe, XnSe, XiSe, yLe, ySe)
        else:
            
            XLe, XSe = eval_long.float(),  eval_short.float()
            yLe, ySe = eval_targets_long.float(), eval_targets_short.float()
            minN = min(len(XLe), len(XSe), len(yLe), len(ySe))
            XLe, XSe, yLe, ySe = XLe[:minN], XSe[:minN], yLe[:minN], ySe[:minN]
            eval_dataset = TensorDataset(XLe, XSe, yLe, ySe)
            
            # XLe, XSe = eval_long.float(),  eval_short.float()
            # yLe, ySe = eval_targets_long.float(), eval_targets_short.float()
    
            # packLe = {'Xn': XLe, 'Xi': None, 'y': yLe}
            # packSe = {'Xn': XSe, 'Xi': None, 'y': ySe}
    
            # packLe, packSe = oversample_align(packLe, packSe)
    
            # XLe, yLe = packLe['Xn'], packLe['y']
            # XSe, ySe = packSe['Xn'], packSe['y']
            # eval_dataset = TensorDataset(XLe, XSe, yLe, ySe)
    else:
        # 작은 쪽에 맞춰 자르기
        if is_umass:
            XnLe, XiLe = eval_long
            XnSe, XiSe = eval_short
            yLe, ySe   = eval_targets_long, eval_targets_short
            minN = min(XnLe.shape[0], XnSe.shape[0])
            XnLe, XiLe, yLe = XnLe[:minN].float(), XiLe[:minN].long(), yLe[:minN].float()
            XnSe, XiSe, ySe = XnSe[:minN].float(), XiSe[:minN].long(), ySe[:minN].float()
            eval_dataset = TensorDataset(XnLe, XiLe, XnSe, XiSe, yLe, ySe)
        else:
            XLe, XSe = eval_long.float(), eval_short.float()
            yLe, ySe = eval_targets_long.float(), eval_targets_short.float()
            minN = min(XLe.shape[0], XSe.shape[0])
            XLe, XSe, yLe, ySe = XLe[:minN], XSe[:minN], yLe[:minN], ySe[:minN]
            eval_dataset = TensorDataset(XLe, XSe, yLe, ySe)

                

    # -------- DataLoaders --------
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader  = DataLoader(dataset=eval_dataset,  batch_size=batch_size, shuffle=False)

    # -------- 손실/옵티마이저/스케줄러 --------
    # 사용자 정의 가중 손실(원래 코드와 동일 시그니처):
    criterion = SimpleMSELoss(alpha=alpha, beta=beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # for epoch in tqdm(range(num_epochs), desc="Training Epochs", unit="epoch"):
    for epoch in range(num_epochs):
        model.train()
        train_loss_acc = 0.0

        # 🔹 에폭 내 배치 진행도 표시
        with tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False) as tepoch:
            for i, batch in enumerate(tepoch, 1):
                optimizer.zero_grad()
            
                if is_umass:
                    XnL_b, XiL_b, XnS_b, XiS_b, yL_b, yS_b = [t.to(device) for t in batch]
                    # 모델 forward: (long_num, short_num, long_icon, short_icon)
                    if 'Att' in model_name:
                        outL, outS, *_ = model(XnL_b, XnS_b, XiL_b, XiS_b)
                    else:
                        outL, outS = model(XnL_b, XnS_b, XiL_b, XiS_b)
                else:
                    XL_b, XS_b, yL_b, yS_b = [t.to(device) for t in batch]
                    if 'Att' in model_name:
                        outL, outS, *_ = model(XL_b, XS_b)
                    else:
                        outL, outS = model(XL_b, XS_b)
    
                loss = criterion(outL, yL_b, outS, yS_b)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss_acc += loss.item()

        train_loss = train_loss_acc / len(train_loader)

        # ===================== Eval =====================
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                if is_umass:
                    XnL_b, XiL_b, XnS_b, XiS_b, yL_b, yS_b = [t.to(device) for t in batch]
                    if 'Att' in model_name:
                        vL, vS, *_ = model(XnL_b, XnS_b, XiL_b, XiS_b)
                    else:
                        vL, vS = model(XnL_b, XnS_b, XiL_b, XiS_b)
                        
                    eval_loss += criterion(vL, yL_b, vS, yS_b).item()
                else:
                    XLe_b, XSe_b, yLe_b, ySe_b = [t.to(device) for t in batch]
                    if 'Att' in model_name:
                        vL, vS, *_ = model(XLe_b, XSe_b)
                    else:
                        vL, vS = model(XLe_b, XSe_b)

                    # ✅ 반드시 같은 배치에서 꺼낸 yLe_b, ySe_b를 사용
                    eval_loss += criterion(vL, yLe_b, vS, ySe_b).item()

        eval_loss /= len(eval_loader)
        scheduler.step(eval_loss)

        # ---- alpha/beta 점진 조정 (원본 로직 유지) ----
        if not (criterion.alpha == 1.0 and criterion.beta == 1.0):
            criterion.alpha = min(1.0, criterion.alpha + 0.05)  # Increase alpha
            criterion.beta  = max(0.1, criterion.beta - 0.05)   # Decrease beta

        # ---- early stopping & checkpoint ----
        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            torch.save(model.state_dict(), model_path)
            print(f"[Long/Short] ✓ Epoch {epoch+1} improved: val_loss={eval_loss:.8f} (train={train_loss:.8f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"[Long/Short] noe improved at {epoch+1}")
            
            if epochs_no_improve >= patience:
                print(f"[Short] Early stopping at epoch {epoch} (best val={best_val_loss:.8f})")
                break



def evaluate_for_short_term_forecast(model, eval_sequences, eval_targets, model_name="", batch_size=64):
    """
    EPC:
      - eval_sequences: FloatTensor/ndarray [N, T, D]
      - eval_targets  : FloatTensor/ndarray [N, P]
    UMass:
      - eval_sequences: (Xn, Xi)
          Xn: FloatTensor/ndarray [N, T, D]
          Xi: LongTensor/ndarray  [N, T]
      - eval_targets  : Float/ndarray [N, P]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    is_umass = isinstance(eval_sequences, (tuple, list))

    if is_umass:
        Xn = _as_float_tensor(eval_sequences[0])
        Xi = _as_long_tensor(eval_sequences[1])
        y  = _as_float_tensor(eval_targets)
        eval_dataset = TensorDataset(Xn, Xi, y)
        k = Xn.shape[-1]  # numeric feature dimension only
    else:
        X  = _as_float_tensor(eval_sequences)
        y  = _as_float_tensor(eval_targets)
        eval_dataset = TensorDataset(X, y)
        k = X.shape[-1]

    eval_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)

    outs, tgts = [], []
    with torch.no_grad():
        for batch in eval_loader:
            if is_umass:
                x_num, x_icon, t = [b.to(device) for b in batch]
                out = model(x_num, x_icon) if 'Att' not in model_name else model(x_num, x_icon)[0]
            else:
                x, t = [b.to(device) for b in batch]
                out = model(x) if 'Att' not in model_name else model(x)[0]

            out = _maybe_squeeze_last(out)
            outs.append(out.detach().cpu().numpy())
            tgts.append(t.detach().cpu().numpy())

    all_outputs = np.concatenate(outs, axis=0)
    all_targets = np.concatenate(tgts, axis=0)

    if np.isnan(all_outputs).any() or np.isnan(all_targets).any():
        print("Warning: NaN detected in evaluation data. Returning default scores.")
        return {"R2": 0.0, "Adjusted R2": 0.0, "SMAPE": 0.0, "MASE": 0.0}

    try:
        r2 = r2_score(all_targets, all_outputs)
        n  = len(all_targets)
        adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - k - 1) if (n - k - 1) != 0 else r2
        smape_score = smape(all_targets, all_outputs)
        mase_score  = mase(all_targets, all_outputs)
    except Exception as e:
        print(f"Error during metrics calculation: {e}. Returning default scores.")
        return {"R2": 0.0, "Adjusted R2": 0.0, "SMAPE": 0.0, "MASE": 0.0}

    print(f"R² Score: {r2:.4f}")
    print(f"Adjusted R²: {adjusted_r2:.4f}")
    print(f"SMAPE: {smape_score:.2f}")
    print(f"MASE: {mase_score:.4f}")

    return {"R2": r2, "Adjusted R2": adjusted_r2, "SMAPE": smape_score, "MASE": mase_score}

# ---------------------------
# Long-term Evaluation (EPC/UMass 겸용)
# ---------------------------
def evaluate_for_long_term_forecast(model, eval_data, eval_targets, model_name="", 
                                    batch_size=64, oversample_eval=False):
    """
    EPC:
      eval_data    = (XL, XS)
      eval_targets = (yL, yS)
    UMass:
      eval_data    = ((XnL, XiL), (XnS, XiS))
      eval_targets = (yL, yS)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    eval_long, eval_short = eval_data
    yL, yS = eval_targets

    is_umass = isinstance(eval_long, (tuple, list)) and isinstance(eval_short, (tuple, list))

    # ----- Pack & dtype -----
    if is_umass:
        XnL, XiL = eval_long
        XnS, XiS = eval_short
        packL = {'Xn': _as_float_tensor(XnL), 'Xi': _as_long_tensor(XiL), 'y': _as_float_tensor(yL)}
        packS = {'Xn': _as_float_tensor(XnS), 'Xi': _as_long_tensor(XiS), 'y': _as_float_tensor(yS)}
        k_short = packS['Xn'].shape[-1]
    else:
        XL, XS = eval_long, eval_short
        packL = {'Xn': _as_float_tensor(XL), 'Xi': None, 'y': _as_float_tensor(yL)}
        packS = {'Xn': _as_float_tensor(XS), 'Xi': None, 'y': _as_float_tensor(yS)}
        k_short = packS['Xn'].shape[-1]

    # ----- Oversample or Truncate -----
    if oversample_eval:
        packL, packS = oversample_align(packL, packS)
    else:
        n_min = min(packL['Xn'].shape[0], packS['Xn'].shape[0])
        for key in ['Xn', 'y']:
            packL[key] = packL[key][:n_min]
            packS[key] = packS[key][:n_min]
        if (packL['Xi'] is not None) and (packS['Xi'] is not None):
            packL['Xi'] = packL['Xi'][:n_min]
            packS['Xi'] = packS['Xi'][:n_min]

    # ----- Dataset/Loader -----
    if is_umass:
        eval_dataset = TensorDataset(packL['Xn'], packL['Xi'], packL['y'],
                                     packS['Xn'], packS['Xi'], packS['y'])
    else:
        eval_dataset = TensorDataset(packL['Xn'], packL['y'], packS['Xn'], packS['y'])

    eval_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)

    outs_L, outs_S, tgts_L, tgts_S = [], [], [], []

    with torch.no_grad():
        for batch in eval_loader:
            if is_umass:
                XnL_b, XiL_b, yL_b, XnS_b, XiS_b, yS_b = [t.to(device) for t in batch]
                outL, outS = model(XnL_b, XnS_b, XiL_b, XiS_b) if 'Att' not in model_name else model(XnL_b, XnS_b, XiL_b, XiS_b)[:2]
            else:
                XLe_b, yLe_b, XSe_b, ySe_b = [t.to(device) for t in batch]
                outL, outS = model(XLe_b, XSe_b) if 'Att' not in model_name else model(XLe_b, XSe_b)[:2]

            outL = _maybe_squeeze_last(outL)
            outS = _maybe_squeeze_last(outS)

            outs_L.append(outL.detach().cpu().numpy())
            outs_S.append(outS.detach().cpu().numpy())
            tgts_L.append((yL_b if is_umass else yLe_b).detach().cpu().numpy())
            tgts_S.append((yS_b if is_umass else ySe_b).detach().cpu().numpy())

    outL = np.concatenate(outs_L, axis=0)
    outS = np.concatenate(outs_S, axis=0)
    tgtL = np.concatenate(tgts_L, axis=0)
    tgtS = np.concatenate(tgts_S, axis=0)

    if np.isnan(outS).any() or np.isnan(tgtS).any() or np.isnan(outL).any() or np.isnan(tgtL).any():
        print("Warning: NaN detected in evaluation data. Returning default scores.")
        return {
            "R2 Short": 0.0, "Adjusted R2 Short": 0.0, "SMAPE Short": 0.0, "MASE Short": 0.0,
            "R2 Long": 0.0,  "Adjusted R2 Long": 0.0,  "SMAPE Long": 0.0,  "MASE Long": 0.0
        }

    # ----- Metrics (Short/Long 각각 산출) -----
    try:
        # Short
        r2_s  = r2_score(tgtS, outS)
        n_s   = len(tgtS)
        adj_s = 1 - ((1 - r2_s) * (n_s - 1)) / (n_s - k_short - 1) if (n_s - k_short - 1) != 0 else r2_s
        sm_s  = smape(tgtS, outS)
        ma_s  = mase(tgtS, outS)

        # Long (k_long은 long numeric feature 수)
        k_long = packL['Xn'].shape[-1]
        r2_l  = r2_score(tgtL, outL)
        n_l   = len(tgtL)
        adj_l = 1 - ((1 - r2_l) * (n_l - 1)) / (n_l - k_long - 1) if (n_l - k_long - 1) != 0 else r2_l
        sm_l  = smape(tgtL, outL)
        ma_l  = mase(tgtL, outL)
    except Exception as e:
        print(f"Error during metrics calculation: {e}. Returning default scores.")
        return {
            "R2 Short": 0.0, "Adjusted R2 Short": 0.0, "SMAPE Short": 0.0, "MASE Short": 0.0,
            "R2 Long": 0.0,  "Adjusted R2 Long": 0.0,  "SMAPE Long": 0.0,  "MASE Long": 0.0
        }

    # ----- Print -----
    print("[Short] R²: {:.4f}, Adj.R²: {:.4f}, SMAPE: {:.2f}, MASE: {:.4f}".format(r2_s, adj_s, sm_s, ma_s))
    print("[Long ] R²: {:.4f}, Adj.R²: {:.4f}, SMAPE: {:.2f}, MASE: {:.4f}".format(r2_l, adj_l, sm_l, ma_l))

    return {
        "R2 Short": r2_s, "Adjusted R2 Short": adj_s, "SMAPE Short": sm_s, "MASE Short": ma_s,
        "R2 Long":  r2_l, "Adjusted R2 Long":  adj_l, "SMAPE Long":  sm_l, "MASE Long":  ma_l
    }

def _as_float_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)

def _as_long_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.long)

def _maybe_squeeze_last(outputs):
    # [N, P, 1] 혹은 [N, 1] 같은 경우를 안전하게 제거
    if outputs.ndim >= 2 and outputs.shape[-1] == 1:
        return outputs.squeeze(-1)
    return outputs


