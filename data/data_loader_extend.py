import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import re
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class EPCDataset:
    """
    EPCDataset (auto-selected features)
    - CSV에서 숫자형 컬럼 전체를 자동 선택 + datetime 주기 피처 6개 추가.
    - target_name 미지정 시 기본 'Global_active_power', 없으면 첫 숫자형 컬럼 사용.
    """
    def __init__(self, file_path, sequence_length, prediction_length, target_name=None, filtered_features=None):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        # self.scaler = MinMaxScaler()
        # self.target_scaler = MinMaxScaler()  # 호환성 유지용 (별도 미사용)
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler(feature_range=(0,1))
        self.filtered_features = list(filtered_features) if filtered_features is not None else None

        self.datetime_features = ['sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']
        self.target_name = target_name  # None이면 기본 규칙 적용

        self.numeric_feature_names = None  # load_data에서 채움
        self.selected_features = None      # numeric + datetime

    def load_data(self, test_size=0.2, random_state=42):
        # 1) read
        data = pd.read_csv(self.file_path)

        # 2) datetime 처리 + cyclics
        data = self._add_datetime_features(data)
        data.drop(columns=['datetime'], errors='ignore', inplace=True)

        # 3) 숫자형 전체 선택
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 4) 결측치 처리
        if numeric_cols:
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean(numeric_only=True))

    
        # 5) filtered_features 처리 (dict/list/None 모두 허용)
        requested = self.filtered_features
        if isinstance(requested, dict):
            # long/short 자동 선택 규칙: prediction_length가 길면 long로 간주
            # 필요시 당신 프로젝트 규칙에 맞게 바꾸세요.
            key = 'long' if self.prediction_length >= 168 else 'short'
            requested = requested.get(key, None)
    
        # 컬럼명 매핑(대소문자 무시), datetime 제외
        if requested is not None:
            # 실제 존재하는 숫자형 컬럼을 소문자→원본명으로 매핑
            lower_map = {c.lower(): c for c in numeric_cols}
            req_list = [f for f in list(requested) if f.lower() != 'datetime']  # datetime 자동 제외
            matched, missing = [], []
            for f in req_list:
                hit = lower_map.get(f.lower(), None)
                if hit is not None:
                    matched.append(hit)
                else:
                    missing.append(f)
    
            if len(matched) == 0:
                raise ValueError(
                    "None of the filtered features exist in the dataset.\n"
                    f"Requested (after cleaning): {req_list}\n"
                    f"Available numeric columns (sample): {numeric_cols[:20]}"
                )
            if len(missing) > 0:
                print(f"[EPCDataset] Warning: filtered features not found and ignored: {missing}")
    
            self.selected_features = matched
        else:
            self.selected_features = numeric_cols
    
        self.numeric_feature_names = self.selected_features

        
        # # 5) selected_features = numeric + datetime(이미 numeric로 생성됨)
        # #    datetime cyclic들은 이미 numeric_cols에 들어가 있으므로 중복 제거
        # self.numeric_feature_names = numeric_cols
        # self.selected_features = numeric_cols

        # # 6) 스케일링
        # numeric_array = data[self.selected_features].values  # [T, F]
        # data_scaled = self.scaler.fit_transform(numeric_array)

        # 7) 타깃 결정
        target_name = self._resolve_target_name()
        if target_name not in self.selected_features:
            # fallback: 첫 숫자형 컬럼
            target_name = self.selected_features[0]
        target_col_idx = self.selected_features.index(target_name)

        # 8) 시퀀스 생성
        # sequences, targets = self._create_sequences(data_scaled, target_col_idx)
        sequences, targets = self._create_sequences(data[self.selected_features].values, target_col_idx)

        # 9) split
        X_tr, X_ev, y_tr, y_ev = train_test_split(sequences, targets, test_size=test_size, random_state=random_state)

        # 7) fit scaler on train만
        N, T, F = X_tr.shape
        self.scaler.fit(X_tr.reshape(-1, F))
        X_tr = self.scaler.transform(X_tr.reshape(-1, F)).reshape(N, T, F)

        N, T, F = X_ev.shape
        X_ev = self.scaler.transform(X_ev.reshape(-1, F)).reshape(N, T, F)
    
        self.target_scaler.fit(y_tr)
        y_tr = self.target_scaler.transform(y_tr)
        y_ev = self.target_scaler.transform(y_ev)

        # 10) 텐서화
        return (
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
            torch.tensor(X_ev, dtype=torch.float32),
            torch.tensor(y_ev, dtype=torch.float32),
        )

    def _add_datetime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data['datetime'] = pd.to_datetime(data['datetime'])

        # for LSTM/GRU/CNNLSTM models 
        # data['hour']  = data['datetime'].dt.hour
        # data['day']   = data['datetime'].dt.day
        # data['month'] = data['datetime'].dt.month
    
        # data['sin_hour']  = np.sin(2 * np.pi * data['hour']  / 24)
        # data['cos_hour']  = np.cos(2 * np.pi * data['hour']  / 24)
        # data['sin_day']   = np.sin(2 * np.pi * data['day']   / 31)
        # data['cos_day']   = np.cos(2 * np.pi * data['day']   / 31)
        # data['sin_month'] = np.sin(2 * np.pi * data['month'] / 12)
        # data['cos_month'] = np.cos(2 * np.pi * data['month'] / 12)

        # for Transformer models
        hour  = data['datetime'].dt.hour
        day   = data['datetime'].dt.day
        month = data['datetime'].dt.month
    
        data['sin_hour']  = np.sin(2 * np.pi * hour  / 24)
        data['cos_hour']  = np.cos(2 * np.pi * hour  / 24)
        data['sin_day']   = np.sin(2 * np.pi * day   / 31)
        data['cos_day']   = np.cos(2 * np.pi * day   / 31)
        data['sin_month'] = np.sin(2 * np.pi * month / 12)
        data['cos_month'] = np.cos(2 * np.pi * month / 12)
        
        return data

    def _create_sequences(self, data_scaled: np.ndarray, target_col_idx: int):
        seq_len, pred_len = self.sequence_length, self.prediction_length
        T = len(data_scaled)

        sequences, targets = [], []
        for i in range(T - seq_len - pred_len):
            sequences.append(data_scaled[i:i + seq_len, :])  # [seq_len, F]
            y = data_scaled[i + seq_len:i + seq_len + pred_len, target_col_idx]
            targets.append(y)

        return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32)

    def _resolve_target_name(self):
        # 우선순위: 사용자 지정 → 'Global_active_power' → 첫 숫자형
        if self.target_name:
            return self.target_name
        if 'Global_active_power' in (self.numeric_feature_names or []):
            return 'Global_active_power'
        return (self.numeric_feature_names or [''])[0]
        
class UMassDataset: 
    """
    UMass (use_total merged) dataset loader (auto-selected features)
    - 숫자형 컬럼 전체 자동 선택 + datetime 주기 피처 6개 추가.
    - 'icon'은 범주형 → 임베딩 인덱스(LongTensor)로 시퀀스에 포함(모델에서 nn.Embedding으로 처리).
    - target_name 미지정 시 기본 'use_total', 없으면 첫 숫자형 컬럼 사용.

    추가:
      - 타깃은 롱테일 완화(상위 퍼센타일 클리핑 or log1p) 후 target_scaler로만 스케일.
      - 입력 숫자 피처는 타깃 칼럼 제외하고 self.scaler로 스케일(이중 스케일 방지).
    """
    def __init__(self, file_path, sequence_length, prediction_length,
                 target_name=None, categorical_features=('icon',),
                 filtered_features=None,
                 # ---- 타깃 분포 안정화 옵션 ----
                 target_clip_upper_q=0.995,   # 상위 퍼센타일 클리핑 (None이면 비활성)
                 target_clip_lower_q=None,    # 하위 퍼센타일(필요시) 예: 0.005
                 target_log1p=False           # True면 log1p→표준화/MinMax 패턴 가능(여기선 MinMax 유지)
                 ):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length

        # 숫자형(타깃 제외) 스케일러
        self.scaler = MinMaxScaler()
        # 타깃 전용 스케일러 (평가 시 inverse_transform에 사용)
        self.target_scaler = MinMaxScaler()

        self.datetime_features = ['sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']
        self.target_name = target_name
        self.categorical_features = list(categorical_features) if categorical_features else []

        # set in load_data
        self.numeric_feature_names = None
        self.categorical_feature_names = None
        self.selected_features = None
        self.filtered_features = list(filtered_features) if filtered_features is not None else None

        self.icon2idx, self.idx2icon = None, None

        # 분포 안정화 옵션
        self.target_clip_upper_q = target_clip_upper_q
        self.target_clip_lower_q = target_clip_lower_q
        self.target_log1p = target_log1p

    def load_data(self, test_size=0.2, random_state=42):
        # 1) read
        data = pd.read_csv(self.file_path)

        # 2) datetime 처리 + cyclics
        data = self._add_datetime_features(data)
        data.drop(columns=['datetime'], errors='ignore', inplace=True)

        # 3) 범주형 존재 여부 확인 (현재 기대: 'icon')
        all_cols = list(data.columns)
        present_cats = [c for c in self.categorical_features if c in all_cols]
        self.categorical_feature_names = present_cats

        # 변환 대상: icon 제외 전 컬럼 (시간 sin/cos 포함)
        cols_except_icon = [c for c in data.columns if c not in present_cats]
        if cols_except_icon:
            # 문자/혼합형 → 숫자(float)로 강제 (변환 불가 값은 NaN)
            data[cols_except_icon] = data[cols_except_icon].apply(pd.to_numeric, errors='coerce')
    
            # 전체 NaN인 컬럼은 0.0으로 먼저 채우기(평균이 NaN 되는 문제 방지)
            for c in cols_except_icon:
                if data[c].isna().all():
                    data[c] = 0.0
    
            # 일반 결측치는 평균으로 채움
            data[cols_except_icon] = data[cols_except_icon].fillna(data[cols_except_icon].mean(numeric_only=True))

        
        # 4) 숫자형 전체 선택 (icon 등 범주형 제외)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        col_map = {c.lower(): c for c in numeric_cols}  
        self.numeric_feature_names = numeric_cols
        self.selected_features = numeric_cols 

        # ✅ 최종 feature 선택 로직
        time_feats = [f for f in self.datetime_features if f in numeric_cols]
        if self.filtered_features is None:
            used_features = numeric_cols  # 전체 숫자형 (시간 포함)
        else:
            base = []
            for f in self.filtered_features:
                if f.lower() in col_map:
                    base.append(col_map[f.lower()])
            used_features = list(dict.fromkeys(base + time_feats))  # 선택 피처 + 시간 피처

        # ★ 타깃 누락 방지
        if self.target_name and self.target_name in numeric_cols and self.target_name not in used_features:
            used_features = [self.target_name] + used_features
        
        # ★ 필수 가드
        if len(used_features) == 0:
            raise ValueError(
                f"No valid features after filtering. "
                f"filtered_features={self.filtered_features}, numeric_cols[:8]={numeric_cols[:8]}"
            )
        
        self.selected_features = used_features
        
        # 5) (결측 보정은 위에서 끝났으므로 그대로 진행)
        for cat in present_cats:
            data[cat] = data[cat].astype('object').fillna('unknown')

        # 6) icon 인덱싱
        icon_idx = None
        if 'icon' in present_cats:
            unique_vals = pd.Index(data['icon'].astype(str).unique())
            self.idx2icon = unique_vals.tolist()
            self.icon2idx = {v: i for i, v in enumerate(self.idx2icon)}
            icon_idx = data['icon'].astype(str).map(self.icon2idx).astype(np.int64).values

        # 7) 타깃 결정
        target_name = self._resolve_target_name()
        if target_name not in self.selected_features:
            target_name = self.selected_features[0]
        target_col_idx = self.selected_features.index(target_name)

        # 8) ---- 타깃 분포 안정화 + 타깃 스케일 ----
        y_raw = data[target_name].values.astype(np.float32).reshape(-1, 1)

        # (a) optional clip (상위 퍼센타일)
        if self.target_clip_upper_q is not None:
            hi = np.quantile(y_raw, self.target_clip_upper_q)
            lo = None
            if self.target_clip_lower_q is not None:
                lo = np.quantile(y_raw, self.target_clip_lower_q)
            y_proc = np.clip(y_raw, a_min=lo, a_max=hi)
        else:
            y_proc = y_raw

        # (b) optional log1p
        if self.target_log1p:
            y_fit = np.log1p(y_proc)
        else:
            y_fit = y_proc

        # (c) 타깃 전용 스케일러 적합 & 변환
        self.target_scaler.fit(y_fit)
        y_scaled = self.target_scaler.transform(y_fit).reshape(-1)  # shape: [T]

        # 9) ---- 입력 숫자 피처 스케일 (타깃 칼럼 제외) ----
        #    - 타깃 칼럼은 이미 y_scaled로 대체 (이중 스케일 방지)
        numeric_df = data[self.selected_features].copy()
        # 타깃 제외 컬럼 목록
        non_target_cols = [c for c in self.selected_features if c != target_name]

        if len(non_target_cols) > 0:
            non_target_arr = numeric_df[non_target_cols].values.astype(np.float32)
            non_target_scaled = self.scaler.fit_transform(non_target_arr)  # [T, F-1]
            # 스케일된 비타깃 칼럼 채워넣기
            numeric_df.loc[:, non_target_cols] = non_target_scaled
        # 타깃 칼럼은 y_scaled 그대로 덮어쓰기
        numeric_df.loc[:, target_name] = y_scaled.astype(np.float32)

        data_scaled = numeric_df.values.astype(np.float32)  # [T, F] (타깃은 y_scaled, 나머진 self.scaler)

        # 10) 시퀀스 생성 (타깃 인덱스는 target_col_idx)
        num_seq, cat_seq, targets = self._create_sequences(data_scaled, icon_idx, target_col_idx)

        # 11) split (주의: 현재는 랜덤 split. 시계열이면 시간 순서 split 고려 가능)
        if cat_seq is None:
            Xn_tr, Xn_ev, y_tr, y_ev = train_test_split(num_seq, targets, test_size=test_size, random_state=random_state)
            train_cat = eval_cat = None
        else:
            Xn_tr, Xn_ev, Xc_tr, Xc_ev, y_tr, y_ev = train_test_split(
                num_seq, cat_seq, targets, test_size=test_size, random_state=random_state
            )
            train_cat, eval_cat = Xc_tr, Xc_ev

        # 12) 텐서화
        train_num = torch.tensor(Xn_tr, dtype=torch.float32)
        eval_num  = torch.tensor(Xn_ev, dtype=torch.float32)
        train_y   = torch.tensor(y_tr,  dtype=torch.float32)
        eval_y    = torch.tensor(y_ev,  dtype=torch.float32)

        if train_cat is not None:
            train_cat = torch.tensor(train_cat, dtype=torch.long)
            eval_cat  = torch.tensor(eval_cat,  dtype=torch.long)

        return train_num, train_cat, train_y, eval_num, eval_cat, eval_y

    def _add_datetime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data['datetime'] = pd.to_datetime(data['datetime'])

        # data['hour']  = data['datetime'].dt.hour
        # data['day']   = data['datetime'].dt.day
        # data['month'] = data['datetime'].dt.month
    
        # data['sin_hour']  = np.sin(2 * np.pi * data['hour']  / 24)
        # data['cos_hour']  = np.cos(2 * np.pi * data['hour']  / 24)
        # data['sin_day']   = np.sin(2 * np.pi * data['day']   / 31)
        # data['cos_day']   = np.cos(2 * np.pi * data['day']   / 31)
        # data['sin_month'] = np.sin(2 * np.pi * data['month'] / 12)
        # data['cos_month'] = np.cos(2 * np.pi * data['month'] / 12)
        
        hour  = data['datetime'].dt.hour
        day   = data['datetime'].dt.day
        month = data['datetime'].dt.month
    
        data['sin_hour']  = np.sin(2 * np.pi * hour  / 24)
        data['cos_hour']  = np.cos(2 * np.pi * hour  / 24)
        data['sin_day']   = np.sin(2 * np.pi * day   / 31)
        data['cos_day']   = np.cos(2 * np.pi * day   / 31)
        data['sin_month'] = np.sin(2 * np.pi * month / 12)
        data['cos_month'] = np.cos(2 * np.pi * month / 12)
        return data

    def _create_sequences(self, data_scaled: np.ndarray, icon_idx: np.ndarray, target_col_idx: int):
        seq_len, pred_len = self.sequence_length, self.prediction_length
        T = len(data_scaled)

        num_sequences = []
        cat_sequences = [] if icon_idx is not None else None
        targets = []

        for i in range(T - seq_len - pred_len):
            num_sequences.append(data_scaled[i:i + seq_len, :])  # [seq_len, F]
            if icon_idx is not None:
                cat_sequences.append(icon_idx[i:i + seq_len])     # [seq_len]
            y = data_scaled[i + seq_len:i + seq_len + pred_len, target_col_idx]
            targets.append(y)

        num_sequences = np.asarray(num_sequences, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        if icon_idx is not None:
            cat_sequences = np.asarray(cat_sequences, dtype=np.int64)
        return num_sequences, cat_sequences, targets

    def _resolve_target_name(self):
        # 우선순위: 사용자 지정 → 'use_total' → 첫 숫자형
        if self.target_name:
            return self.target_name
        if 'use_total' in (self.numeric_feature_names or []):
            return 'use_total'
        return (self.numeric_feature_names or [''])[0]

    # Helpers
    def get_numeric_feature_names(self):
        return list(self.selected_features) if self.selected_features else []
        
    def get_categorical_feature_names(self):
        return list(self.categorical_feature_names) if self.categorical_feature_names else []
        
    def get_icon_vocab(self):
        return self.icon2idx, self.idx2icon




        
        

# class UMassDataset:
#     def __init__(
#         self,
#         file_path: str,
#         sequence_length: int,
#         prediction_length: int,
#         # ▼▼ 기존 코드 호환용: target_features 받아서 target_col로 매핑
#         target_col: str | None = None,
#         target_features: list | None = None,
#         # ▼▼ 나머지는 그대로
#         cat_col: str | None = "icon",
#         datetime_col: str = "timeline",
#         drop_cols: list[str] | None = None,
#         test_size: float = 0.2,
#         include_target_in_x: bool = False,
#         random_state: int = 42,
#         scaler: MinMaxScaler | None = None,
#         target_scaler: MinMaxScaler | None = None,
#         # ▼▼ 혹시 모를 추가 키워드 흡수 (호환성)
#         **kwargs
#     ):
#         # target_features가 들어오면 첫 항목을 target_col로 사용
#         if target_col is None:
#             target_col = "use_total"

#         self.file_path = file_path
#         self.sequence_length = sequence_length
#         self.prediction_length = prediction_length
#         self.target_col = target_col
#         self.cat_col = cat_col
#         self.datetime_col = datetime_col
#         self.drop_cols = drop_cols or []
#         self.test_size = test_size
#         self.include_target_in_x = include_target_in_x
#         self.random_state = random_state

#         self.scaler = scaler if scaler is not None else MinMaxScaler()
#         self.target_scaler = target_scaler if target_scaler is not None else MinMaxScaler()

#         self.icon_vocab: dict[str, int] | None = None
#         self.feature_names_: list[str] | None = None
#         self.icon_unknown_id: int = 0

#         # 선택: 기존 코드와 인터페이스 맞추기 위해 selected_features 제공
#         self.selected_features: list[str] | None = None

#     # ====== Public ======
#     def load_data(self):
#         """
#         반환(UMass 형식, 호출부와 호환):
#           Xn_tr, Xi_tr, y_tr, Xn_ev, Xi_ev, y_ev
#           - Xn_* : (B,L,D) float32
#           - Xi_* : (B,L)   int64
#           - y_*  : (B,P)   float32
#         """
#         # 1) 로드 + 시간 정리
#         df = pd.read_csv(self.file_path)
#         df = self._read_and_prepare() if hasattr(self, "_read_and_prepare") else df
    
#         # _read_and_prepare가 없다면 최소한 아래 3줄은 수행되어야 합니다:
#         # df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
#         # df = df.sort_values(self.datetime_col).set_index(self.datetime_col)
#         # (필요시 reindex/보간/드롭 등)
    
#         # 2) 타깃 결정
#         if getattr(self, "target_col", None):
#             target_col = self.target_col
#         else:
#             target_col = self._infer_target(df)
    
#         # 3) 수치/아이콘 분리
#         num_df, icon_ids = self._select_numeric_and_icon(df, target_col)  # (T,D_num), (T,)
#         X_num_all = num_df.values.astype(np.float32)
#         X_icon_all = icon_ids.astype(np.int64)
#         y_all = pd.to_numeric(df[target_col], errors="coerce").values
    
#         # 4) 시퀀스(스케일 전)
#         Xn_raw, Xi_raw, Y_raw = self._make_windows(X_num_all, X_icon_all, y_all)
#         if len(Xn_raw) == 0:
#             raise ValueError("시퀀스를 만들 수 없습니다. sequence/prediction 길이를 확인하세요.")
    
#         # 5) 시간 순서 split (누출 방지)
#         N = Xn_raw.shape[0]
#         n_eval = max(1, int(N * self.test_size))
#         n_tr = N - n_eval
#         Xn_tr_raw, Xi_tr, y_tr_raw = Xn_raw[:n_tr], Xi_raw[:n_tr], Y_raw[:n_tr]
#         Xn_ev_raw, Xi_ev, y_ev_raw = Xn_raw[n_tr:], Xi_raw[n_tr:], Y_raw[n_tr:]
    
#         # 6) 스케일링 (train에만 fit → eval transform)
#         Btr, L, D = Xn_tr_raw.shape
#         self.scaler.fit(Xn_tr_raw.reshape(Btr * L, D))
#         Xn_tr = self.scaler.transform(Xn_tr_raw.reshape(Btr * L, D)).reshape(Btr, L, D)
#         Xn_ev = self.scaler.transform(Xn_ev_raw.reshape(-1, D)).reshape(Xn_ev_raw.shape)
    
#         self.target_scaler.fit(y_tr_raw.reshape(-1, 1))
#         y_tr = self.target_scaler.transform(y_tr_raw.reshape(-1, 1)).reshape(y_tr_raw.shape)
#         y_ev = self.target_scaler.transform(y_ev_raw.reshape(-1, 1)).reshape(y_ev_raw.shape)
    
#         # 7) 텐서 변환
#         Xn_tr_t = torch.tensor(Xn_tr, dtype=torch.float32)
#         Xi_tr_t = torch.tensor(Xi_tr, dtype=torch.long)
#         y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32)
    
#         Xn_ev_t = torch.tensor(Xn_ev, dtype=torch.float32)
#         Xi_ev_t = torch.tensor(Xi_ev, dtype=torch.long)
#         y_ev_t  = torch.tensor(y_ev,  dtype=torch.float32)
    
#         # 8) (호환성) selected_features 채워주기
#         if hasattr(self, "feature_names_"):
#             self.selected_features = list(self.feature_names_)
#         else:
#             self.selected_features = list(num_df.columns)
    
#         # 9) 호출부가 기대하는 형식(6개)으로 반환
#         return Xn_tr_t, Xi_tr_t, y_tr_t, Xn_ev_t, Xi_ev_t, y_ev_t

#     def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
#         """평가 시 타깃 역변환(help)"""
#         y_scaled = np.asarray(y_scaled)
#         if y_scaled.ndim == 1:
#             y_scaled = y_scaled.reshape(-1, 1)
#         return self.target_scaler.inverse_transform(y_scaled)

#     # ====== Internal (간결 버전) ======
#     def _read_and_prepare(self) -> pd.DataFrame:
#         df = pd.read_csv(self.file_path)

#         # 시간컬럼 정규화
#         if self.datetime_col not in df.columns:
#             # 대체 후보 자동 탐색
#             for c in ["timeline", "datetime", "Date & Time", "date_time", "date_&_time"]:
#                 if c in df.columns:
#                     self.datetime_col = c
#                     break
#         if self.datetime_col not in df.columns:
#             raise ValueError(f"'{self.datetime_col}' column not found.")
#         df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
#         df = df.sort_values(self.datetime_col).set_index(self.datetime_col)

#         # 드롭(요청 컬럼 + gen류)
#         drop_list = [c for c in (self.drop_cols or []) if c in df.columns]
#         if drop_list:
#             df = df.drop(columns=drop_list)
#         gen_like = [c for c in df.columns if c.lower().startswith("gen_") or c.lower() == "gen"]
#         if gen_like:
#             df = df.drop(columns=gen_like)

#         # 시간 인덱스 연속화(시간 누락 보간 대비)
#         full_idx = pd.date_range(df.index.min(), df.index.max(), freq="1h")
#         df = df.reindex(full_idx)

#         # 범주형(icon) 안전 매핑
#         if self.cat_col and self.cat_col in df.columns:
#             icons = df[self.cat_col]
#         else:
#             # 없으면 새로 만들기(전부 unknown)
#             icons = pd.Series(index=df.index, data="unknown", name=self.cat_col or "icon")

#         # NaN → "unknown" → str
#         icons = icons.where(icons.notna(), "unknown").astype(str)
#         if not isinstance(self.icon_vocab, dict) or len(self.icon_vocab) == 0:
#             uniq = sorted(icons.unique().tolist())
#             if "unknown" not in uniq:
#                 uniq.insert(0, "unknown")
#             self.icon_vocab = {s: i for i, s in enumerate(uniq)}
#             self.icon_unknown_id = self.icon_vocab.get("unknown", 0)
#         icon_ids = icons.map(self.icon_vocab).fillna(self.icon_unknown_id).astype(int)
#         df["icon_id"] = icon_ids.values

#         # 날짜 주기 특성 추가
#         hour = df.index.hour
#         day = df.index.day
#         month = df.index.month
#         df["sin_hour"]  = np.sin(2 * np.pi * hour / 24)
#         df["cos_hour"]  = np.cos(2 * np.pi * hour / 24)
#         df["sin_day"]   = np.sin(2 * np.pi * day  / 31)
#         df["cos_day"]   = np.cos(2 * np.pi * day  / 31)
#         df["sin_month"] = np.sin(2 * np.pi * month / 12)
#         df["cos_month"] = np.cos(2 * np.pi * month / 12)

#         # 수치형 결측 보간: 시간 보간 → 시각별 중앙값 → 전체 중앙값
#         num_cols = df.select_dtypes(include="number").columns.tolist()
#         df[num_cols] = df[num_cols].interpolate(method="time", limit_direction="both")
#         if df[num_cols].isna().any().any():
#             by_hour = df.groupby(df.index.hour)[num_cols].transform("median")
#             df[num_cols] = df[num_cols].fillna(by_hour)
#             df[num_cols] = df[num_cols].fillna(df[num_cols].median())

#         return df

#     def _infer_target(self, df: pd.DataFrame) -> str:
#         # 우선순위: use_total > use_m* > use
#         lower = {c.lower(): c for c in df.columns}
#         for cand in ["use_total", "use_m3", "use_m2", "use_m4", "use"]:
#             if cand in lower:
#                 return lower[cand]
#         # 백업: 첫 수치형
#         num_cols = df.select_dtypes(include="number").columns.tolist()
#         if not num_cols:
#             raise ValueError("No numeric columns to choose as target.")
#         return num_cols[0]

#     def _select_numeric_and_icon(self, df: pd.DataFrame, target_col: str):
#         """
#         입력 수치 피처와 icon_id 분리.
#         - 기본: 타깃은 입력에서 제외(include_target_in_x=False)
#         - 날짜 주기 특성 이미 포함되어 있음
#         """
#         icon_ids = df["icon_id"].astype(int).values if "icon_id" in df.columns else np.zeros(len(df), dtype=int)
#         num_cols = df.select_dtypes(include="number").columns.tolist()
#         if "icon_id" in num_cols:
#             num_cols.remove("icon_id")
#         if not self.include_target_in_x and target_col in num_cols:
#             num_cols.remove(target_col)
#         self.feature_names_ = num_cols
#         return df[num_cols].copy(), icon_ids

#     def _make_windows(self, X_num: np.ndarray, X_icon: np.ndarray, y: np.ndarray):
#         """(T,D), (T,), (T,) → (N,L,D), (N,L), (N,P)"""
#         L, P = self.sequence_length, self.prediction_length
#         T = len(X_num)
#         N = T - L - P
#         if N <= 0:
#             return np.empty((0, L, X_num.shape[1])), np.empty((0, L), dtype=np.int64), np.empty((0, P))
#         Xn = np.stack([X_num[i:i+L]          for i in range(N)], axis=0)
#         Xi = np.stack([X_icon[i:i+L]         for i in range(N)], axis=0)
#         Y  = np.stack([y[i+L:i+L+P]          for i in range(N)], axis=0)
#         return Xn, Xi, Y