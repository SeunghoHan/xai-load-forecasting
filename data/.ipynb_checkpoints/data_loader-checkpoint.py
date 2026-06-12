import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from typing import Tuple, Optional

class EPCDataset:
    def __init__(self, file_path, sequence_length, prediction_length, target_features):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        # Add datetime features to target features
        self.datetime_features = ['sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']
        self.selected_features = target_features + self.datetime_features

    def load_data(self):
        data = pd.read_csv(self.file_path)
        data = self._add_datetime_features(data)
        data.drop(columns=['datetime'], errors='ignore', inplace=True)
        data.fillna(data.mean(), inplace=True)

        # Select features
        data_selected = data[self.selected_features]

        # Normalize the data
        data_scaled = self.scaler.fit_transform(data_selected.values)

        # Create sequences
        sequences, targets = self._create_sequences(data_scaled)

        # Split the data into train and eval sets
        train_sequences, eval_sequences, train_targets, eval_targets = train_test_split(sequences, targets, test_size=0.2, random_state=42)

        return (
            torch.tensor(train_sequences, dtype=torch.float32),
            torch.tensor(train_targets, dtype=torch.float32).squeeze(-1),
            torch.tensor(eval_sequences, dtype=torch.float32),
            torch.tensor(eval_targets, dtype=torch.float32).squeeze(-1)
        )
    

    def _add_datetime_features(self, data):
        # Convert datetime column and add cyclic features
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['hour'] = data['datetime'].dt.hour
        data['day'] = data['datetime'].dt.day
        data['month'] = data['datetime'].dt.month

        data['sin_hour'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['sin_day'] = np.sin(2 * np.pi * data['day'] / 31)
        data['cos_day'] = np.cos(2 * np.pi * data['day'] / 31)
        data['sin_month'] = np.sin(2 * np.pi * data['month'] / 12)
        data['cos_month'] = np.cos(2 * np.pi * data['month'] / 12)

        return data

    def _create_sequences(self, data):
        # Generate sequences and targets
        sequences = []
        targets = []

        for i in range(len(data) - self.sequence_length - self.prediction_length):
            sequences.append(data[i:i + self.sequence_length])
            targets.append(data[i + self.sequence_length:i + self.sequence_length + self.prediction_length, 0])

        return np.array(sequences), np.array(targets)

# class EPCDataset:
#     def __init__(
#         self,
#         file_path: str,
#         sequence_length: int,
#         prediction_length: int,
#         feature_cols: Optional[list] = None,
#         target_col: str = 'Global_active_power',
#         eval_ratio: float = 0.2,
#         x_scaler: Optional[object] = None,
#         y_scaler: Optional[object] = None,
#         # 이전 버전과의 호환성을 위한 인자
#         target_features: Optional[list] = None
#     ):
#         self.file_path = file_path
#         self.sequence_length = sequence_length
#         self.prediction_length = prediction_length
#         self.eval_ratio = eval_ratio

#         # 외부에서 스케일러를 제공하지 않으면 새로 생성
#         self.scaler = x_scaler if x_scaler is not None else MinMaxScaler()
#         self.target_scaler = y_scaler if y_scaler is not None else MinMaxScaler()

#         # 사용할 기본 수치형 피처 목록
#         default_num_features = [
#             'Global_active_power', 'Global_intensity', 'Sub_metering_1',
#             'Sub_metering_2', 'Sub_metering_3', 'Temperature', 'Humidity'
#         ]
#         self.datetime_features = ['sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']

#         # 인자 우선순위: feature_cols > target_features > default
#         base_feats = feature_cols if feature_cols is not None else target_features
#         if base_feats is None:
#             base_feats = default_num_features

#         # 타겟 변수를 항상 맨 앞으로 배치 (시퀀스 생성 시 용이)
#         if target_col in base_feats:
#             base_feats.remove(target_col)
#         self.selected_features = [target_col] + base_feats + self.datetime_features
#         self.target_col = target_col

#     def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         """데이터를 로드, 전처리, 분할하고 텐서로 변환하여 반환합니다."""
#         df = pd.read_csv(self.file_path)
#         df = self._add_datetime_features(df)

#         # 실제 존재하는 컬럼만 선택
#         use_cols = [c for c in self.selected_features if c in df.columns]
#         if self.target_col not in use_cols:
#             raise ValueError(f"Target column '{self.target_col}' not found in the dataframe.")
#         data = df[use_cols].copy()

#         # 수치형 데이터에 대해서만 결측치를 평균으로 대체
#         num_cols = data.select_dtypes(include=np.number).columns.tolist()
#         data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
        
#         # --- 💡 핵심 수정: 시간 순서대로 데이터 분할 (Data Leakage 방지) ---
#         total_len = len(data)
#         split_idx = int(total_len * (1.0 - self.eval_ratio))

#         train_df = data.iloc[:split_idx].copy()
#         eval_df  = data.iloc[split_idx:].copy()

#         # --- 💡 핵심 수정: 스케일러를 학습 데이터에만 fit ---
#         self.scaler.fit(train_df.values)
#         train_scaled = self.scaler.transform(train_df.values)
#         eval_scaled  = self.scaler.transform(eval_df.values)

#         # 분리된 데이터셋으로 시퀀스 생성
#         X_tr, y_tr = self._create_sequences(train_scaled)
#         X_ev, y_ev = self._create_sequences(eval_scaled)
        
#         # 타겟 스케일러는 학습 데이터의 타겟(y_tr)에만 fit
#         self.target_scaler.fit(y_tr)

#         # 텐서로 변환
#         X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
#         y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
#         X_ev_t = torch.tensor(X_ev, dtype=torch.float32)
#         y_ev_t = torch.tensor(y_ev, dtype=torch.float32)

#         return X_tr_t, y_tr_t, X_ev_t, y_ev_t

#     def _add_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """데이터프레임에 주기성을 나타내는 시간 관련 피처를 추가합니다."""
#         if 'datetime' not in df.columns:
#             raise ValueError("Input CSV must contain a 'datetime' column.")
#         df = df.copy()
#         dt = pd.to_datetime(df['datetime'])
        
#         df['hour']  = dt.dt.hour
#         df['day']   = dt.dt.day
#         df['month'] = dt.dt.month

#         df['sin_hour']  = np.sin(2 * np.pi * df['hour'] / 24)
#         df['cos_hour']  = np.cos(2 * np.pi * df['hour'] / 24)
#         df['sin_day']   = np.sin(2 * np.pi * df['day'] / 31)
#         df['cos_day']   = np.cos(2 * np.pi * df['day'] / 31)
#         df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
#         df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

#         # 모델 입력에 불필요한 원본 열 제거
#         df.drop(columns=['datetime', 'hour', 'day', 'month'], errors='ignore', inplace=True)
#         return df

#     def _create_sequences(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """연속된 데이터 배열로부터 입력 시퀀스와 타겟 시퀀스를 생성합니다."""
#         seqs, ys = [], []
#         num_rows, num_features = arr.shape
#         L = self.sequence_length
#         H = self.prediction_length

#         for i in range(num_rows - L - H + 1):
#             window = arr[i : i + L]            # (sequence_length, num_features)
#             target = arr[i + L : i + L + H, 0] # (prediction_length,) - 첫 번째 열이 타겟
#             seqs.append(window)
#             ys.append(target)
            
#         return np.array(seqs), np.array(ys)