import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler

from .lime_extend import LimeExplainer
from .lime_multiterm_extend import LimeExplainer_MT
from .shap import ShapExplainer
from .attention import AttentionExplainer
from .grad_cam import GradCAMExplainer
from .lrp import LRPExplainer


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x
    
def unpack_X_and_icon(data):
    """
    data: np/torch or tuple/list like (X_num, X_icon, y, ...)
    returns: (X_num_np, X_icon_np_or_None)
    """
    if isinstance(data, (list, tuple)):
        X_num = data[0]
        X_icon = data[1] if len(data) > 1 else None
    else:
        X_num, X_icon = data, None
    return _to_numpy(X_num), _to_numpy(X_icon)

def unpack_eval_for_single(eval_data):
    """
    returns: (X_num_np, X_icon_np_or_None, N)
    """
    X_num, X_icon = unpack_X_and_icon(eval_data)
    return X_num, X_icon, len(X_num)


def get_explainer(explainer_type, model, device, train_data, sequence_length, 
                  input_size, selected_features, is_long_term_forecast, scaler=None):

    if is_long_term_forecast:
        # --- NEW: tuple-safe unpack for long/short ---
        X_long, ICON_long = unpack_X_and_icon(train_data[0])   # (N, S_long, F_long), (N, S_long) or None
        X_short, ICON_short = unpack_X_and_icon(train_data[1]) # (N, S_short, F_short), (N, S_short) or None

        if explainer_type == 'LIME':
            # 각 term별로 스케일러를 (N, S*F) 기준으로 fit
            scaler_long = MinMaxScaler()
            scaler_short = MinMaxScaler()

            if X_long.ndim != 3 or X_short.ndim != 3:
                raise ValueError(f"[LIME_MT] Expect (N,S,F) for both long/short. "
                                 f"Got long={X_long.shape}, short={X_short.shape}")
            Nl, Sl, Fl = X_long.shape
            Ns, Ss, Fs = X_short.shape
            scaler_long.fit(X_long.reshape(Nl, Sl * Fl))
            scaler_short.fit(X_short.reshape(Ns, Ss * Fs))


            return LimeExplainer_MT(
                model=model,
                device=device,
                train_long=X_long,
                train_short=X_short,
                icon_long=ICON_long,            # ★ icon 전달
                icon_short=ICON_short,          # ★ icon 전달
                sequence_length_long=sequence_length['long'],
                sequence_length_short=sequence_length['short'],
                input_size_long=input_size['long'],
                input_size_short=input_size['short'],
                selected_features_long=selected_features['long'],
                selected_features_short=selected_features['short'],
                scaler_long=skaler_long if False else scaler_long,  # (오타 방지) 그냥 scaler_long
                scaler_short=scaler_short
            )

        elif explainer_type == 'SHAP':
            return ShapExplainer(model, train_data, sequence_length, input_size, selected_features, device)
        elif explainer_type == 'ATTENTION':
            return AttentionExplainer(model, device, selected_features)
        else:
            print(f"Invalid explainer type: {explainer_type}")
            return None
    else:
        # ---- SINGLE-TERM ----
        X_train, ICON_train = unpack_X_and_icon(train_data)

        if explainer_type == 'LIME':
            scaler = MinMaxScaler()
            if X_train.ndim == 3:
                N, S, F = X_train.shape
                scaler.fit(X_train.reshape(N, -1))
            elif X_train.ndim == 2:
                scaler.fit(X_train)
            else:
                raise ValueError(f"Unsupported train X shape: {X_train.shape}")

            return LimeExplainer(
                model=model,
                device=device,
                sequences=X_train,               # 수치 시퀀스
                icon_sequences=ICON_train,       # ★ 아이콘 시퀀스 추가
                sequence_length=sequence_length['single'],
                input_size=input_size['single'],
                selected_features=selected_features['single'],
                scaler=scaler
            )
            
        elif explainer_type == 'SHAP':
            return ShapExplainer(model, train_data, sequence_length, input_size, selected_features, device)
        elif explainer_type == 'ATTENTION':
            return AttentionExplainer(model, device, selected_features)
        else:
            print(f"Invalid explainer type: {explainer_type}")
            return None