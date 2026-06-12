









# from typing import List, Optional, Tuple, Dict, Any
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# import torch
# from dataclasses import dataclass

# @dataclass
# class EmbeddingInfo:
#     num_embeddings: int
#     padding_idx: Optional[int]
#     index_to_token: List[str]
#     token_to_index: Dict[str, int]

# class UMassDataset:
#     """
#     Preprocessor & sequence maker for UMass Home + weather data (merged CSV).
#     Assumptions based on 'HomeA_with_weather.csv':
#       - time column named 'timeline' (string), convertible to datetime
#       - categorical weather 'icon' with string values (e.g., 'clear-day', 'rain')
#       - multiple numeric meter/weather columns including target
#     Key features:
#       - Adds cyclic datetime features (hour/day/month)
#       - Scales numeric columns with MinMaxScaler (excluding 'icon' string)
#       - Converts 'icon' to integer indices for use with an nn.Embedding
#       - Creates rolling sequences of length `sequence_length`
#       - Targets are the next `prediction_length` steps for `target_col`
#       - Chronological split by default to respect time series ordering
#     """

#     def __init__(
#         self,
#         file_path: str,
#         sequence_length: int,
#         prediction_length: int,
#         target_col: str,
#         include_cols: Optional[List[str]] = None,
#         test_size: float = 0.2,
#         chronological_split: bool = True,
#         add_datetime_features: bool = True,
#     ) -> None:
#         self.file_path = file_path
#         self.sequence_length = sequence_length
#         self.prediction_length = prediction_length
#         self.target_col = target_col
#         self.include_cols = include_cols  # if None: infer all numeric + weather except 'icon' & 'timeline'
#         self.test_size = test_size
#         self.chronological_split = chronological_split
#         self.add_datetime_features = add_datetime_features

#         self.scaler = MinMaxScaler()
#         self.target_scaler = MinMaxScaler()  # kept for parity; not used separately unless needed

#         # set during load
#         self.embedding_info: Optional[EmbeddingInfo] = None
#         self.numeric_feature_names: List[str] = []
#         self.icon_index_series: Optional[np.ndarray] = None

#     def _add_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         if 'timeline' not in df.columns:
#             raise ValueError("Expected a 'timeline' column in the dataset.")
#         dt = pd.to_datetime(df['timeline'], errors='coerce')
#         if dt.isna().any():
#             # Drop rows with invalid timestamps
#             mask = ~dt.isna()
#             df = df.loc[mask].copy()
#             dt = dt.loc[mask]
#         df['hour'] = dt.dt.hour
#         df['day'] = dt.dt.day
#         df['month'] = dt.dt.month

#         # cyclic encodings
#         df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
#         df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
#         df['sin_day'] = np.sin(2 * np.pi * df['day'] / 31)
#         df['cos_day'] = np.cos(2 * np.pi * df['day'] / 31)
#         df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
#         df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
#         return df

#     def _encode_icon(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, EmbeddingInfo]:
#         # safe fill and build mapping
#         if 'icon' not in df.columns:
#             # No icon present -> create a single dummy class "none"
#             tokens = ['none']
#             token_to_index = {t: i for i, t in enumerate(tokens)}
#             df['icon_idx'] = 0
#         else:
#             df['icon'] = df['icon'].fillna('unknown').astype(str)
#             tokens = sorted(df['icon'].unique().tolist())
#             token_to_index = {t: i for i, t in enumerate(tokens)}
#             df['icon_idx'] = df['icon'].map(token_to_index).astype(int)

#         emb = EmbeddingInfo(
#             num_embeddings=len(token_to_index),
#             padding_idx=None,
#             index_to_token=tokens,
#             token_to_index=token_to_index,
#         )
#         return df, emb

#     def _infer_numeric_columns(self, df: pd.DataFrame) -> List[str]:
#         # If include_cols is provided, honor it; else choose all numeric except id/time/icon fields
#         if self.include_cols is not None:
#             cols = list(self.include_cols)
#         else:
#             exclude = {'timeline', 'icon'}
#             cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
#         # ensure target is first in the numeric feature order for continuity with old pipeline
#         if self.target_col not in cols:
#             raise ValueError(f"target_col '{self.target_col}' not found among numeric columns")
#         cols = [self.target_col] + [c for c in cols if c != self.target_col]
#         return cols

#     def _create_sequences(
#         self,
#         X_num: np.ndarray,
#         icon_idx: np.ndarray,
#         target_idx: int
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """
#         Build rolling windows.
#         X_num: [N, F] numeric features (scaled), target at index target_idx
#         icon_idx: [N] integer indices per timestep
#         Returns:
#           seq_num: [M, seq_len, F]
#           seq_icon: [M, seq_len] (int)
#           y: [M, pred_len] (target values)
#         """
#         N = X_num.shape[0]
#         seq_len = self.sequence_length
#         pred_len = self.prediction_length

#         max_start = N - seq_len - pred_len
#         if max_start <= 0:
#             raise ValueError("Not enough rows to create sequences with the given lengths. "
#                              f"Got N={N}, seq_len={seq_len}, pred_len={pred_len}.")

#         seq_num_list = []
#         seq_icon_list = []
#         y_list = []

#         for i in range(max_start):
#             seq_num_list.append(X_num[i:i+seq_len, :])
#             seq_icon_list.append(icon_idx[i:i+seq_len])
#             y_list.append(X_num[i+seq_len:i+seq_len+pred_len, target_idx])

#         seq_num = np.stack(seq_num_list, axis=0)
#         seq_icon = np.stack(seq_icon_list, axis=0).astype(np.int64)
#         y = np.stack(y_list, axis=0)
#         return seq_num, seq_icon, y

#     def load_data(self) -> Tuple[
#         torch.Tensor, torch.Tensor, torch.Tensor,
#         torch.Tensor, torch.Tensor, torch.Tensor,
#         EmbeddingInfo, List[str]
#     ]:
#         """
#         Returns:
#           train_X_num: [B1, seq_len, F] float32
#           train_X_icon: [B1, seq_len] long
#           train_y: [B1, pred_len] float32
#           eval_X_num: [B2, seq_len, F] float32
#           eval_X_icon: [B2, seq_len] long
#           eval_y: [B2, pred_len] float32
#           embedding_info: EmbeddingInfo (for nn.Embedding)
#           numeric_feature_names: List[str] (order used in X_num; target_col is index 0)
#         """
#         df = pd.read_csv(self.file_path)

#         # 1) datetime features
#         if self.add_datetime_features:
#             df = self._add_datetime_features(df)

#         # 2) icon -> index
#         df, emb = self._encode_icon(df)

#         # 3) Fill missing numeric values with column means
#         num_cols_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
#         df[num_cols_all] = df[num_cols_all].fillna(df[num_cols_all].mean())

#         # 4) Select numeric features
#         self.numeric_feature_names = self._infer_numeric_columns(df)

#         # 5) Scale numeric features
#         X_num = self.scaler.fit_transform(df[self.numeric_feature_names].values)

#         # 6) Extract icon indices array
#         icon_idx = df['icon_idx'].values

#         # 7) Create sequences
#         target_idx = self.numeric_feature_names.index(self.target_col)  # should be 0
#         seq_num, seq_icon, y = self._create_sequences(X_num, icon_idx, target_idx)

#         # 8) Split train/eval (chronologically by default)
#         N = seq_num.shape[0]
#         split = int(N * (1 - self.test_size))
#         if self.chronological_split:
#             train_X_num, eval_X_num = seq_num[:split], seq_num[split:]
#             train_X_icon, eval_X_icon = seq_icon[:split], seq_icon[split:]
#             train_y, eval_y = y[:split], y[split:]
#         else:
#             # random split
#             rng = np.random.RandomState(42)
#             idx = np.arange(N)
#             rng.shuffle(idx)
#             split_idx = idx[:split]
#             eval_idx = idx[split:]
#             train_X_num, eval_X_num = seq_num[split_idx], seq_num[eval_idx]
#             train_X_icon, eval_X_icon = seq_icon[split_idx], seq_icon[eval_idx]
#             train_y, eval_y = y[split_idx], y[eval_idx]

#         # 9) Tensors
#         train_X_num = torch.tensor(train_X_num, dtype=torch.float32)
#         eval_X_num  = torch.tensor(eval_X_num,  dtype=torch.float32)
#         train_X_icon = torch.tensor(train_X_icon, dtype=torch.long)
#         eval_X_icon  = torch.tensor(eval_X_icon,  dtype=torch.long)
#         train_y = torch.tensor(train_y, dtype=torch.float32)
#         eval_y  = torch.tensor(eval_y,  dtype=torch.float32)

#         self.embedding_info = emb
#         return (train_X_num, train_X_icon, train_y,
#                 eval_X_num,  eval_X_icon,  eval_y,
#                 emb, self.numeric_feature_names)
