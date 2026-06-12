import os
import json
import numpy as np
import torch


def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

class BaseExplainer:
    def __init__(self, model, device, sequences, sequence_length, input_size, selected_features):
        self.model = model
        self.device = device
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.selected_features = selected_features

    def explain(self, data_point):
        raise NotImplementedError("The explain method must be implemented by the subclass.")

    def inverse_transform_time_features(self, sequences, selected_features, scaler, sequence_length=None):
        """
        sequences: (N,S,F) or (N,S*F) or (S,F) or (S*F,)
        selected_features: list[str] (길이 = F)
        scaler: MinMaxScaler fitted on either F or S*F
        sequence_length: S (없으면 추론)
        """
        X = _to_numpy(sequences)
        if scaler is None:
            return X  # 스케일러 없으면 그대로 반환

        F = len(selected_features)
        n_features_fit = getattr(scaler, "n_features_in_", None)
        if n_features_fit is None:
            # 구버전 스케일러 호환
            n_features_fit = scaler.scale_.shape[0]

        # 현재 텐서 모양 파악
        if X.ndim == 3:
            N, S, F_in = X.shape
            assert F_in == F, f"[inverse] F mismatch: data F={F_in}, selected F={F}"
            # 스케일러가 S*F로 fit되었는지, F로 fit되었는지에 따라 처리
            if n_features_fit == S * F:
                X_flat = X.reshape(N, S * F)
                X_inv = scaler.inverse_transform(X_flat).reshape(N, S, F)
            elif n_features_fit == F:
                # 드물게 timestep별로 동일 스케일을 쓴 경우
                X_2d = X.reshape(N * S, F)
                X_inv = scaler.inverse_transform(X_2d).reshape(N, S, F)
            else:
                raise ValueError(f"[inverse] unexpected scaler n_features_in_={n_features_fit}, "
                                 f"but data is (N,S,F={F})")
            return X_inv

        elif X.ndim == 2:
            N, D = X.shape
            # D가 F인지 S*F인지 판단
            if D == F:
                if n_features_fit == F:
                    return scaler.inverse_transform(X)
                elif n_features_fit % F == 0:
                    # sequence_length 추론 또는 인자 사용
                    S = sequence_length if sequence_length is not None else (n_features_fit // F)
                    if S * F != n_features_fit:
                        raise ValueError("[inverse] cannot infer sequence length")
                    # 현재는 (N,F)라서 타임축 정보가 없음 → 그대로 반환(혹은 에러)
                    # 안전하게 스케일만 원복
                    return scaler.inverse_transform(X)
                else:
                    raise ValueError(f"[inverse] scaler fitted to {n_features_fit} features, but got D={D}")
            else:
                # 아마 (N, S*F)
                if n_features_fit != D:
                    raise ValueError(f"[inverse] scaler fitted to {n_features_fit} features, but got D={D}")
                X_inv = scaler.inverse_transform(X)
                # sequence_length가 주어지면 원형 복원
                if sequence_length is not None and (D % F == 0) and (D // F == sequence_length):
                    return X_inv.reshape(N, sequence_length, F)
                return X_inv

        elif X.ndim == 1:
            D = X.shape[0]
            if n_features_fit != D:
                raise ValueError(f"[inverse] scaler fitted to {n_features_fit} features, but got D={D}")
            return scaler.inverse_transform(X[None, :]).squeeze(0)

        else:
            raise ValueError(f"[inverse] unsupported ndim={X.ndim}")

    # def inverse_transform_time_features(self, sequences, selected_features, scaler):
    #     """
    #     정규화된 시퀀스 데이터를 원래 값으로 복구하고, 시간 피처(`month`, `day`, `hour`)도 복구
    
    #     Args:
    #         sequences (numpy.ndarray): 정규화된 시퀀스 데이터
    #         selected_features (list): 선택된 피처 목록
    #         scaler (object): MinMaxScaler 또는 StandardScaler 객체
        
    #     Returns:
    #         numpy.ndarray: 복구된 시퀀스 데이터 (실제 값)
    #     """
    #     # 전체 시퀀스 데이터의 정규화된 부분을 원래 값으로 복구
    #     original_sequences = scaler.inverse_transform(sequences.reshape(-1, len(selected_features))).reshape(sequences.shape)
        
    #     return original_sequences