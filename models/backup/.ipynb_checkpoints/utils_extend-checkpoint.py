import os
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

from .ls_loss import LS_Loss, SimpleMSELoss

from torch.utils.data import DataLoader, TensorDataset


def create_model(model_name, 
                 input_size,                 # dict 또는 int (이전 호환)
                 hidden_size,
                 num_layers, 
                 output_size,                # dict 또는 int (이전 호환)
                 dropout=0.2, 
                 long_term_length=None, 
                 short_term_length=None, 
                 # ===== UMass/EPC 공통 =====
                 input_size_num=None,        # 단기/단일에서 수치형 feature 수
                 icon_vocab_size=None,       # UMass: len(ds.icon_vocab); EPC: None 또는 0
                 icon_embed_dim=8            # 임베딩 차원
                ):
    """
    - 단기/단일: LSTM/GRU/CNNLSTM/LSTM-Att 등은 (x_num[, x_icon]) 입력을 받는 통합형 모델 사용
    - 장기: LS_CNNLSTM / LS_CNNLSTM_Att 은 (long_num, short_num[, long_icon, short_icon]) 입력
    - EPC: icon_vocab_size=None → 임베딩 비활성, 기존과 동일 동작/성능
    """

    # --- 출력 크기 정규화 ---
    if isinstance(output_size, dict):
        out_single = output_size.get('single')
        out_long   = output_size.get('long')
        out_short  = output_size.get('short')
    else:
        out_single = output_size
        out_long = out_short = None

    # --- 입력 크기 정규화 ---
    def _resolve_single_in():
        if input_size_num is not None:  # 권장 경로(UMass/공용)
            return input_size_num
        if isinstance(input_size, dict) and 'single' in input_size:
            return input_size['single']
        if isinstance(input_size, int):
            return input_size
        raise ValueError("input_size_num 또는 input_size['single'] 중 하나를 제공하세요.")

    def _resolve_long_short_in():
        if isinstance(input_size, dict) and ('long' in input_size and 'short' in input_size):
            return input_size['long'], input_size['short']
        # 공용 파이프라인에서 long/short 모두 같은 D를 쓸 때
        if input_size_num is not None:
            return input_size_num, input_size_num
        raise ValueError("input_size에 {'long','short'} 또는 input_size_num을 제공하세요.")

    # --- 단기/단일 ---
    if model_name in ['GRU', 'LSTM', 'LSTM-Att', 'CNNLSTM']:
        in_num = _resolve_single_in()
        out_sz = out_single

        if model_name == 'GRU':
            return GRUModel(
                input_size=in_num,
                output_size=out_sz,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                icon_vocab_size=icon_vocab_size,
                icon_embed_dim=icon_embed_dim
            )
        elif model_name == 'LSTM':
            return LSTMModel(
                input_size_num=in_num,
                output_size=out_sz,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                icon_vocab_size=icon_vocab_size,
                icon_embed_dim=icon_embed_dim
            )
        elif model_name == 'CNNLSTM':
            return CNNLSTMModel(
                input_size_num=in_num,
                output_size=out_sz,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                icon_vocab_size=icon_vocab_size,
                icon_embed_dim=icon_embed_dim
            )

    # --- 장기(dual-term) ---
    elif model_name in ['LS_CNNLSTM', 'LS_CNNLSTM_Att']:
        in_long, in_short = _resolve_long_short_in()
        if model_name == 'LS_CNNLSTM':
            return LongShortCNNLSTM(
                long_input_size_num=in_long,
                short_input_size_num=in_short,
                hidden_size=hidden_size,
                num_layers=num_layers,
                long_output_size=out_long,
                short_output_size=out_short,
                long_term_length=long_term_length,
                short_term_length=short_term_length,
                dropout=dropout,
                icon_vocab_size=icon_vocab_size,
                icon_embed_dim=icon_embed_dim
            )
        else:
            return LongShortCNNLSTMWithAttention(
                long_input_size_num=in_long,
                short_input_size_num=in_short,
                hidden_size=hidden_size,
                num_layers=num_layers,
                long_output_size=out_long,
                short_output_size=out_short,
                long_term_length=long_term_length,
                short_term_length=short_term_length,
                dropout=dropout,
                icon_vocab_size=icon_vocab_size,
                icon_embed_dim=icon_embed_dim
            )

    else:
        raise ValueError(f"Model {model_name} is not recognized.")

def oversample_any(data, targets):
    """
    data:
      - 단기(EPC): X
      - 단기(UMass): (Xn, Xi)
      - 장기(EPC):   (X_long, X_short)
      - 장기(UMass): ((XnL, XiL), (XnS, XiS))
    targets:
      - 단기: y
      - 장기: (yL, yS)

    반환: 입력과 동일한 구조에서 작은 쪽을 큰 쪽 길이에 맞춰 복제.
    """
    def _len(x):
        if isinstance(x, tuple):  # (Xn, Xi) 또는 ((XnL, XiL), (XnS, XiS))
            return _len(x[0])
        return len(x)

    def _oversample_side(side, target_len):
        if isinstance(side, tuple):  # (Xn, Xi)
            Xn, Xi = side
            cur = len(Xn)
            k, r = divmod(target_len, cur)
            Xn_new = torch.cat([Xn]*k + [Xn[:r]], dim=0)
            Xi_new = torch.cat([Xi]*k + [Xi[:r]], dim=0)
            return (Xn_new, Xi_new)
        else:
            cur = len(side)
            k, r = divmod(target_len, cur)
            return torch.cat([side]*k + [side[:r]], dim=0)

    # 단기: 오버샘플 적용하지 않음 -> 그대로 반환
    if not isinstance(targets, tuple):
        return data, targets

    # 장기: 길이 맞추기
    A, B = data            # long, short 입력
    yA, yB = targets       # long, short 타깃
    lenA, lenB = _len(A), _len(B)
    if lenA == lenB:
        return data, targets

    if lenA < lenB:
        A_new = _oversample_side(A, lenB)
        yA_new = _oversample_side(yA, lenB) if isinstance(yA, torch.Tensor) else yA
        return (A_new, B), (yA_new, yB)
    else:
        B_new = _oversample_side(B, lenA)
        yB_new = _oversample_side(yB, lenA) if isinstance(yB, torch.Tensor) else yB
        return (A, B_new), (yA, yB_new)

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path,  weights_only=True))
    return model

def smape(y_true, y_pred, eps=1e-6):
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    return np.mean(num / den) * 100

# MASE 정의
def mase(y_true, y_pred, naive_forecast=None):
    mae_pred = np.mean(np.abs(y_true - y_pred))
    if naive_forecast is None:
        naive_forecast = np.roll(y_true, shift=1)
        naive_forecast[0] = y_true[0]  # Shift로 발생하는 문제 해결
    mae_naive = np.mean(np.abs(y_true - naive_forecast))
    mase = mae_pred / mae_naive
    return mase

def train_for_long_term_forecast(
    model, model_name,
    train_data, train_targets,
    eval_data,  eval_targets,
    model_path, num_epochs=100, batch_size=64,
    learning_rate=0.001, patience=5,
    oversample_eval=False, alpha=0.3, beta=1.0
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 학습 데이터: 항상 오버샘플 적용 (기존 정책)
    train_loader, train_kind = _make_loader_longshort(
        train_data, train_targets, batch_size, shuffle=True, apply_oversample=True
    )
    # 평가 데이터: oversample_eval 플래그로 제어
    eval_loader,  eval_kind  = _make_loader_longshort(
        eval_data,  eval_targets,  batch_size, shuffle=False, apply_oversample=oversample_eval
    )
    assert train_kind == eval_kind

    criterion = SimpleMSELoss(alpha=alpha, beta=beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    best_val, bad = float('inf'), 0
    for epoch in range(num_epochs):
        # ------- train -------
        model.train(); s=0.0; n=0
        for batch in train_loader:
            optimizer.zero_grad()
            if train_kind == "umass":
                XnL, XiL, XnS, XiS, yL, yS = [b.to(device) for b in batch]
                if 'Att' in model_name:
                    outL, outS, _ = model(XnL, XnS, XiL, XiS)
                else:
                    outL, outS = model(XnL, XnS, XiL, XiS)
            else:
                XL, XS, yL, yS = [b.to(device) for b in batch]
                if 'Att' in model_name:
                    outL, outS, _ = model(XL, XS)
                else:
                    outL, outS = model(XL, XS)

            loss = criterion(outL, yL, outS, yS)
            loss.backward(); optimizer.step()
            s += loss.item() * (yL.size(0)+yS.size(0)); n += (yL.size(0)+yS.size(0))
        train_loss = s / max(n,1)

        # ------- eval -------
        model.eval(); s=0.0; n=0
        with torch.no_grad():
            for batch in eval_loader:
                if eval_kind == "umass":
                    XnL, XiL, XnS, XiS, yL, yS = [b.to(device) for b in batch]
                    if 'Att' in model_name:
                        outL, outS, _ = model(XnL, XnS, XiL, XiS)
                    else:
                        outL, outS = model(XnL, XnS, XiL, XiS)
                else:
                    XL, XS, yL, yS = [b.to(device) for b in batch]
                    if 'Att' in model_name:
                        outL, outS, _ = model(XL, XS)
                    else:
                        outL, outS = model(XL, XS)

                loss = criterion(outL, yL, outS, yS).item()
                s += loss * (yL.size(0)+yS.size(0)); n += (yL.size(0)+yS.size(0))
        val_loss = s / max(n,1)
        scheduler.step(val_loss)

        # alpha/beta 스케줄 유지
        if not (criterion.alpha == 1.0 and criterion.beta == 1.0):
            criterion.alpha = min(1.0, criterion.alpha + 0.05)
            criterion.beta  = max(0.1,  criterion.beta  - 0.05)

        if val_loss < best_val:
            best_val, bad = val_loss, 0
            torch.save(model.state_dict(), model_path)
            print(f"[Long/Short] ✓ Epoch {epoch+1} improved: val_loss={val_loss:.8f} (train={train_loss:.8f})")
        else:
            bad += 1
            print(f"[Long/Short] noe improved at {epoch+1}")
            if bad >= patience:
                print(f"[Long/Short] early stop at {epoch+1}")
                break

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

def train_for_short_term_forecast(
    model, model_name,
    train_sequences, train_targets,
    eval_sequences,  eval_targets,
    model_path,
    num_epochs=100, batch_size=64, learning_rate=1e-3, patience=5
):
    """
    model      : LSTM/GRU/CNNLSTM/LSTM-Att 등 (통합 입력 지원: (x_num, x_icon) or x)
    model_name : 문자열(예: 'LSTM', 'CNNLSTM', 'LSTM-Att' 등)
    train_sequences, eval_sequences:
        EPC   : X
        UMass : (X_num, X_icon)
    train_targets, eval_targets:
        y     : (B, pred_len)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loader, train_kind = _make_loader_short(train_sequences, train_targets, batch_size, shuffle=True)
    eval_loader,  eval_kind  = _make_loader_short(eval_sequences,  eval_targets,  batch_size, shuffle=False)
    assert train_kind == eval_kind, "Train/Eval 데이터셋 유형이 다릅니다."

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val = np.inf
    bad_epochs = 0

    for epoch in range(1, num_epochs + 1):
        # ---------------- train ----------------
        model.train()
        running, n = 0.0, 0

        for batch in tqdm(train_loader, desc=f"[Short] Epoch {epoch}/{num_epochs}", leave=False):
            optimizer.zero_grad()

            if train_kind == "umass":
                xn, xi, y = [b.to(device) for b in batch]
                if 'Att' in model_name:
                    out, _ = model(xn, xi)
                else:
                    out = model(xn, xi)
            else:
                x, y = [b.to(device) for b in batch]
                if 'Att' in model_name:
                    out, _ = model(x)
                else:
                    out = model(x)

            loss = criterion(out.squeeze(), y)
            loss.backward()
            optimizer.step()

            running += loss.item() * y.size(0)
            n += y.size(0)

        train_loss = running / max(n, 1)

        # ---------------- eval ----------------
        model.eval()
        running, n = 0.0, 0
        with torch.no_grad():
            for batch in eval_loader:
                if eval_kind == "umass":
                    xn, xi, y = [b.to(device) for b in batch]
                    if 'Att' in model_name:
                        out, _ = model(xn, xi)
                    else:
                        out = model(xn, xi)
                else:
                    x, y = [b.to(device) for b in batch]
                    if 'Att' in model_name:
                        out, _ = model(x)
                    else:
                        out = model(x)

                loss = criterion(out.squeeze(), y).item()
                running += loss * y.size(0)
                n += y.size(0)

        val_loss = running / max(n, 1)

        # ---------------- checkpoint / early stop ----------------
        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), model_path)
            print(f"[Short] ✓ Epoch {epoch} improved: val_loss={val_loss:.8f} (train={train_loss:.8f})")
        else:
            bad_epochs += 1
            print(f"[Short] noe improved at {epoch+1}")
            
            if bad_epochs >= patience:
                print(f"[Short] Early stopping at epoch {epoch} (best val={best_val:.8f})")
                break

    # best model 복원
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    

def evaluate_for_long_term_forecast(model, eval_data, eval_targets, model_name="", 
                                    batch_size=64, oversample_eval=False, target_scaler=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    loader, kind = _make_loader_long_eval(eval_data, eval_targets, batch_size, oversample_eval)

    outsL, outsS, tgsL, tgsS = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if kind == "umass":
                XnL, XiL, XnS, XiS, yL, yS = [b.to(device) for b in batch]
                outputs = model(XnL, XnS, XiL, XiS)
            else:
                XL, XS, yL, yS = [b.to(device) for b in batch]
                outputs = model(XL, XS)

            if 'Att' in model_name:
                outL, outS, _ = outputs
            else:
                outL, outS = outputs

            outsL.append(outL.detach().cpu().numpy())
            outsS.append(outS.detach().cpu().numpy())
            tgsL.append(yL.detach().cpu().numpy())
            tgsS.append(yS.detach().cpu().numpy())

    yL_pred = np.concatenate(outsL, axis=0)
    yS_pred = np.concatenate(outsS, axis=0)
    yL_true = np.concatenate(tgsL, axis=0)
    yS_true = np.concatenate(tgsS, axis=0)

    # === target_scaler 처리 (tuple이면 short만) ===
    if target_scaler is not None:
        if isinstance(target_scaler, (list, tuple)):
            short_scaler = target_scaler[1]
        else:
            short_scaler = target_scaler
        if short_scaler is not None:
            yS_pred = short_scaler.inverse_transform(yS_pred)
            yS_true = short_scaler.inverse_transform(yS_true)

    if np.isnan(yS_pred).any() or np.isnan(yS_true).any():
        print("Warning: NaN detected in evaluation data. Returning default scores.")
        return {"R2 Short": 0.0, "Adjusted R2": 0.0, "SMAPE Short": 0.0, "MASE Short": 0.0}

    # === 단기(Short) 기준으로 지표 산출 ===
    r2_short = r2_score(yS_true, yS_pred)
    n = yS_true.shape[0]
    if isinstance(eval_data[1], tuple):  # UMass
        k = eval_data[1][0].shape[-1]
    else:                                # EPC
        k = eval_data[1].shape[-1]
    adjusted_r2 = 1 - ((1 - r2_short) * (n - 1)) / max(1, (n - k - 1))
    smape_short = smape(yS_true, yS_pred)
    mase_short  = mase(yS_true, yS_pred)

    print(f"[Long/Short] R²(Short): {r2_short:.4f} | Adj.R²: {adjusted_r2:.4f} "
          f"| SMAPE(Short): {smape_short:.2f} | MASE(Short): {mase_short:.4f}")
    return {
        "R2 Short": r2_short,
        "Adjusted R2": adjusted_r2,
        "SMAPE Short": smape_short,
        "MASE Short": mase_short
    }



def evaluate_for_short_term_forecast(model, eval_sequences, eval_targets, model_name="", 
                                     batch_size=64, target_scaler=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    loader, kind = _make_loader_short_eval(eval_sequences, eval_targets, batch_size, shuffle=False)

    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            if kind == "umass":
                xn, xi, y = [b.to(device) for b in batch]
                out = model(xn, xi)[0] if ('Att' in model_name) else model(xn, xi)
            else:
                x, y = [b.to(device) for b in batch]
                out = model(x)[0] if ('Att' in model_name) else model(x)
            preds.append(out.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    if target_scaler is not None:
        y_pred = target_scaler.inverse_transform(y_pred)
        y_true = target_scaler.inverse_transform(y_true)

    if np.isnan(y_pred).any() or np.isnan(y_true).any():
        print("Warning: NaN detected in evaluation data. Returning default scores.")
        return {"R2": 0.0, "Adjusted R2": 0.0, "SMAPE": 0.0, "MASE": 0.0}

    # R² / Adj.R²
    r2 = r2_score(y_true, y_pred)
    n = y_true.shape[0]
    # feature 수(k): UMass면 수치 피처 차원, EPC면 입력 피처 차원
    if isinstance(eval_sequences, tuple):
        k = eval_sequences[0].shape[-1]  # numeric feature dim
    else:
        k = eval_sequences.shape[-1]
    adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / max(1, (n - k - 1))

    # SMAPE / MASE (유틸에 정의되어 있다고 가정)
    smape_score = smape(y_true, y_pred)
    mase_score  = mase(y_true, y_pred)

    print(f"[Short] R²: {r2:.4f} | Adj.R²: {adjusted_r2:.4f} | SMAPE: {smape_score:.2f} | MASE: {mase_score:.4f}")
    return {"R2": r2, "Adjusted R2": adjusted_r2, "SMAPE": smape_score, "MASE": mase_score}





def _make_loader_longshort(data, targets, batch_size, shuffle, apply_oversample: bool):
    """
    data:
      EPC   -> (X_long, X_short)
      UMass -> ((XnL, XiL), (XnS, XiS))
    targets: (yL, yS)

    apply_oversample: True  -> 데이터 길이 맞추도록 오버샘플
                      False -> 작은 쪽 기준으로 잘라서 맞춤
    """
    # 오버샘플 여부 결정
    if apply_oversample:
        data, targets = oversample_any(data, targets)
        (A, B), (yA, yB) = data, targets
    else:
        # 작은 쪽 길이에 맞춰 "다운샘플"(슬라이스) — 기존 코드 동작 유지
        (A, B) = data
        (yA, yB) = targets

        def _len(x): return len(x[0]) if isinstance(x, tuple) else len(x)
        min_n = min(_len(A), _len(B))

        def _slice_side(side, n):
            if isinstance(side, tuple):
                Xn, Xi = side
                return (Xn[:n], Xi[:n])
            else:
                return side[:n]

        A = _slice_side(A, min_n)
        B = _slice_side(B, min_n)
        yA = yA[:min_n]
        yB = yB[:min_n]

    # DataLoader 구성
    if isinstance(A, tuple):  # UMass
        XnL, XiL = A; XnS, XiS = B
        ds = TensorDataset(
            XnL.to(torch.float32), XiL.to(torch.long),
            XnS.to(torch.float32), XiS.to(torch.long),
            yA.to(torch.float32),  yB.to(torch.float32)
        )
        kind = "umass"
    else:                      # EPC
        ds = TensorDataset(
            A.to(torch.float32), B.to(torch.float32),
            yA.to(torch.float32), yB.to(torch.float32)
        )
        kind = "epc"

    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return loader, kind


def _make_loader_short(sequences, targets, batch_size, shuffle):
    """
    sequences:
      - EPC   : X                   (B, T, D)
      - UMass : (X_num, X_icon)     (B, T, D_num), (B, T)
    targets:
      - y     : (B, pred_len)
    return: (DataLoader, kind)  where kind in {"epc", "umass"}
    """
    if isinstance(sequences, tuple):
        # UMass: (수치, 아이콘)
        Xn, Xi = sequences
        ds = TensorDataset(
            Xn.to(torch.float32),
            Xi.to(torch.long),
            targets.to(torch.float32),
        )
        kind = "umass"
    else:
        # EPC: 수치만
        X = sequences
        ds = TensorDataset(
            X.to(torch.float32),
            targets.to(torch.float32),
        )
        kind = "epc"

    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return loader, kind

def _make_loader_short_eval(sequences, targets, batch_size, shuffle=False):
    if isinstance(sequences, tuple):  # UMass
        Xn, Xi = sequences
        ds = TensorDataset(
            Xn.to(torch.float32),
            Xi.to(torch.long),
            targets.to(torch.float32),
        )
        kind = "umass"
    else:  # EPC
        X = sequences
        ds = TensorDataset(
            X.to(torch.float32),
            targets.to(torch.float32),
        )
        kind = "epc"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False), kind

def _make_loader_long_eval(eval_data, eval_targets, batch_size, oversample_eval=False):
    eval_long, eval_short = eval_data
    yL, yS = eval_targets

    def _len(x):
        return len(x[0]) if isinstance(x, tuple) else len(x)

    if oversample_eval:
        lenL, lenS = _len(eval_long), _len(eval_short)
        if lenL != lenS:
            tgt = max(lenL, lenS)
            def _overside(side, target_len):
                if isinstance(side, tuple):
                    Xn, Xi = side
                    cur = len(Xn)
                    k, r = divmod(target_len, cur)
                    return (torch.cat([Xn]*k + [Xn[:r]], dim=0),
                            torch.cat([Xi]*k + [Xi[:r]], dim=0))
                else:
                    cur = len(side)
                    k, r = divmod(target_len, cur)
                    return torch.cat([side]*k + [side[:r]], dim=0)
            if lenL < tgt:
                eval_long = _overside(eval_long, tgt)
                yL = _overside(yL, tgt) if isinstance(yL, torch.Tensor) else yL
            if lenS < tgt:
                eval_short = _overside(eval_short, tgt)
                yS = _overside(yS, tgt) if isinstance(yS, torch.Tensor) else yS
    else:
        tgt = min(_len(eval_long), _len(eval_short))
        def _slice_side(side, n):
            if isinstance(side, tuple):
                Xn, Xi = side
                return (Xn[:n], Xi[:n])
            else:
                return side[:n]
        eval_long  = _slice_side(eval_long, tgt)
        eval_short = _slice_side(eval_short, tgt)
        yL = yL[:tgt]
        yS = yS[:tgt]

    if isinstance(eval_long, tuple):  # UMass
        XnL, XiL = eval_long; XnS, XiS = eval_short
        ds = TensorDataset(
            XnL.to(torch.float32), XiL.to(torch.long),
            XnS.to(torch.float32), XiS.to(torch.long),
            yL.to(torch.float32),  yS.to(torch.float32)
        )
        kind = "umass"
    else:  # EPC
        ds = TensorDataset(
            eval_long.to(torch.float32),  eval_short.to(torch.float32),
            yL.to(torch.float32),         yS.to(torch.float32)
        )
        kind = "epc"

    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False), kind
