import os
import re
import json
import numpy as np
import torch
import lime
from tqdm import tqdm
from lime import lime_tabular
import matplotlib.pyplot as plt 
from collections import defaultdict 

from .base_explainer import BaseExplainer


def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

class LimeExplainer_MT(BaseExplainer):
    def __init__(
        self,
        model, device,
        train_long, train_short,
        icon_long=None, icon_short=None,                 # ★ NEW: icon
        sequence_length_long=None, sequence_length_short=None,
        input_size_long=None, input_size_short=None,
        selected_features_long=None, selected_features_short=None,
        scaler_long=None, scaler_short=None              # ★ NEW: term별 scaler
    ):
        super().__init__(model, device, train_long, sequence_length_long, input_size_long, selected_features_long)

        # 저장 (numpy로)
        self.train_long  = _to_numpy(train_long)
        self.train_short = _to_numpy(train_short)
        self.icon_long   = _to_numpy(icon_long)  if icon_long  is not None else None
        self.icon_short  = _to_numpy(icon_short) if icon_short is not None else None

        self.sequence_length_long  = sequence_length_long
        self.sequence_length_short = sequence_length_short
        self.input_size_long  = input_size_long
        self.input_size_short = input_size_short
        self.selected_features_long  = selected_features_long
        self.selected_features_short = selected_features_short

        self.scaler_long  = scaler_long
        self.scaler_short = scaler_short

        # icon 배치 컨텍스트 (predict 시 사용)
        self._icon_ctx_long  = None
        self._icon_ctx_short = None
        self._icon_point_long_single  = None  # explain()에서 1개 보관 후 predict에서 tile
        self._icon_point_short_single = None

        # LIME explainer (term별)
        self.lime_tab_long  = self._create_explainer(self.train_long,  self.sequence_length_long,  self.selected_features_long)
        self.lime_tab_short = self._create_explainer(self.train_short, self.sequence_length_short, self.selected_features_short)
    
    def _create_explainer(self, train_data, sequence_length, selected_features, max_samples=2500):
        # max_samples: 학습 데이터 몇개 이용해서 local explainer 만들지
        
        reshaped_data = train_data.reshape(len(train_data), -1)
        
        if reshaped_data.shape[0] > max_samples:
            idx = np.random.choice(reshaped_data.shape[0], max_samples, replace=False)
            reshaped_data = reshaped_data[idx]
    
        return lime_tabular.LimeTabularExplainer(
            training_data=reshaped_data,
            feature_names=[f"{feature}_{i}" for i in range(sequence_length) for feature in selected_features],
            class_names=['Prediction'],
            mode='regression'
        )
        
    def _predict_fn_common(self, long_num, short_num):
            """
            long_num:  (B, Sl, Fl) torch.float32
            short_num: (B, Ss, Fs) torch.float32
            icon ctx:  self._icon_ctx_long (B, Sl) / self._icon_ctx_short (B, Ss) or None
            """
            self.model.eval()
            with torch.no_grad():
                if self._icon_ctx_long is not None or self._icon_ctx_short is not None:
                    # 모델이 (long_num, short_num, long_icon, short_icon) 시그니처일 것으로 가정
                    long_icon  = torch.tensor(self._icon_ctx_long,  dtype=torch.long, device=self.device)  if self._icon_ctx_long  is not None else None
                    short_icon = torch.tensor(self._icon_ctx_short, dtype=torch.long, device=self.device) if self._icon_ctx_short is not None else None
                    outputs = self.model(long_num, short_num, long_icon, short_icon)
                else:
                    outputs = self.model(long_num, short_num)
    
                return outputs

    def _predict_fn_long(self, x_flat_long, eval_short_sampled):
        """
        LIME-long용: long만 섭동, short는 고정 샘플 배치
        x_flat_long: (B, Sl*Fl) numpy
        eval_short_sampled: (B, Ss, Fs) numpy
        """
        B = x_flat_long.shape[0]
        long_num  = torch.tensor(x_flat_long.reshape(B, self.sequence_length_long,  self.input_size_long),
                                 dtype=torch.float32, device=self.device)
        short_num = torch.tensor(eval_short_sampled, dtype=torch.float32, device=self.device)

        outputs = self._predict_fn_common(long_num, short_num)
        # 모델이 (long, short) 순서대로 tuple 출력한다고 가정
        if isinstance(outputs, tuple):
            out = outputs[0]
        else:
            out = outputs
        return out.detach().cpu().numpy()

    def _predict_fn_short(self, x_flat_short, eval_long_sampled):
        """
        LIME-short용: short만 섭동, long은 고정 샘플 배치
        """
        B = x_flat_short.shape[0]
        short_num = torch.tensor(x_flat_short.reshape(B, self.sequence_length_short, self.input_size_short),
                                 dtype=torch.float32, device=self.device)
        long_num  = torch.tensor(eval_long_sampled, dtype=torch.float32, device=self.device)

        outputs = self._predict_fn_common(long_num, short_num)
        if isinstance(outputs, tuple):
            out = outputs[1]
        else:
            out = outputs
        return out.detach().cpu().numpy()
    
    def _tile_icon_ctx(self, B):
        """
        explain()에서 보관한 단일 icon 시퀀스를 배치 크기 B로 타일링해 predict에서 사용
        """
        if self.icon_long is not None and self._icon_point_long_single is not None:
            ip = self._icon_point_long_single
            self._icon_ctx_long = np.repeat(ip[None, ...], B, axis=0)
        else:
            self._icon_ctx_long = None

        if self.icon_short is not None and self._icon_point_short_single is not None:
            ip = self._icon_point_short_single
            self._icon_ctx_short = np.repeat(ip[None, ...], B, axis=0)
        else:
            self._icon_ctx_short = None

    def explain(
        self,
        data_point_long, data_point_short,
        eval_long, eval_short,
        num_samples=1000, is_visual=False,
        icon_point_long=None, icon_point_short=None   # ★ NEW: 현재 샘플의 icon 고정
    ):
        # numpy로
        dpl = _to_numpy(data_point_long)
        dps = _to_numpy(data_point_short)
        dpl = dpl.reshape(1, -1)
        dps = dps.reshape(1, -1)

        # 아이콘 단일 샘플 보관
        self._icon_point_long_single  = _to_numpy(icon_point_long)  if icon_point_long  is not None else None
        self._icon_point_short_single = _to_numpy(icon_point_short) if icon_point_short is not None else None

        # 보조 배치(상대 term 고정용) 샘플링
        # LIME 내부에서 predict_fn 호출 때마다 B가 달라서, wrapper에서 B 감지 후 icon 타일링
        eval_long_np  = _to_numpy(eval_long)
        eval_short_np = _to_numpy(eval_short)

        # 샘플링 (num_samples개) — 필요 시 크기 제한
        num_samples = int(num_samples)
        idx_l = np.random.choice(len(eval_long_np),  size=num_samples, replace=(len(eval_long_np)  < num_samples))
        idx_s = np.random.choice(len(eval_short_np), size=num_samples, replace=(len(eval_short_np) < num_samples))
        eval_long_sampled  = eval_long_np[idx_l]
        eval_short_sampled = eval_short_np[idx_s]

        # predict wrapper: 호출 시점의 B를 감지해 icon 타일링
        def predict_long(x_flat):
            B = x_flat.shape[0]
            self._tile_icon_ctx(B)
            return self._predict_fn_long(x_flat, eval_short_sampled)

        def predict_short(x_flat):
            B = x_flat.shape[0]
            self._tile_icon_ctx(B)
            return self._predict_fn_short(x_flat, eval_long_sampled)

        # LIME 실행 (num_features는 전체 고려)
        long_exp = self.lime_tab_long.explain_instance(
            dpl.flatten(), predict_long, num_samples=num_samples,
            num_features=len(self.selected_features_long)
        )
        short_exp = self.lime_tab_short.explain_instance(
            dps.flatten(), predict_short, num_samples=num_samples,
            num_features=len(self.selected_features_short)
        )

        if is_visual:
            # 역변환은 base_explainer의 안전한 버전 사용 (S*F / F 모두 커버)
            self.visualize_lime_explanation(long_exp.as_list(),  eval_long_np,  self.sequence_length_long,  self.selected_features_long,  "Long")
            self.visualize_lime_explanation(short_exp.as_list(), eval_short_np, self.sequence_length_short, self.selected_features_short, "Short")

        return {
            "long": long_exp.as_list(),
            "short": short_exp.as_list()
        }

    def visualize_lime_explanation(self, important_features, sequences, sequence_length, selected_features, term):
        excluded_features = ['sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']
        filtered_features = [(f,i) for (f,i) in important_features if not any(e in f for e in excluded_features)]

        feature_dict = {}
        comp_op = r'(<=|>=|<|>)'
        alph_op = r'[a-zA-Z_]+\d*(?:_\d+)*'

        # 역변환 (term별 scaler 사용)
        scaler = self.scaler_long if term=="Long" else self.scaler_short
        sequences = self.inverse_transform_time_features(sequences, selected_features, scaler, sequence_length=sequence_length)

        for feature_description, importance in filtered_features:
            feature_name = feature_description
            while re.search(comp_op, feature_name):
                feature_name = re.sub(comp_op, '', feature_name)
            feature_name = re.search(alph_op, feature_name).group(0)

            actual_feature, time_step = feature_name.rsplit('_', 1)
            time_step = int(time_step)

            if actual_feature not in feature_dict:
                feature_dict[actual_feature] = []

            feature_value = sequences[0][time_step, selected_features.index(actual_feature)]
            feature_dict[actual_feature].append((time_step, feature_value, importance))

        for actual_feature, values in feature_dict.items():
            plt.figure(figsize=(10,6))
            time_steps   = [v[0] for v in values]
            actual_vals  = [v[1] for v in values]
            importances  = [v[2] for v in values]

            plt.plot(range(sequence_length), sequences[0][:, selected_features.index(actual_feature)], label=f'{actual_feature}')
            plt.scatter(time_steps, actual_vals, color='red', zorder=5, label='LIME Important Points')
            for i, (x, y) in enumerate(zip(time_steps, actual_vals)):
                plt.annotate(f'Value: {y:.3f}\nImp: {importances[i]:.3f}', (x, y),
                             textcoords="offset points", xytext=(0,10), ha='center', color='blue')
            plt.xlabel('Time Step'); plt.ylabel('Feature Value')
            plt.title(f'{term} Term Feature: {actual_feature}')
            plt.legend(loc='upper right'); plt.grid(True); plt.show()

    def extract_important_features(
        self,
        eval_long, eval_short,
        num_datapoints=256,
        top_n_for_long=10,
        top_n_for_short=10,
        num_samples=1000,
        random_state=42,
        early_stop=True,
        check_interval=64,
        stability_k=20,
        rho_thresh=0.90
    ):
        # --- tuple-safe unpack ---
        if isinstance(eval_long, tuple):
            eval_long_np, eval_ICON_long = eval_long
        else:
            eval_long_np, eval_ICON_long = eval_long, None
    
        if isinstance(eval_short, tuple):
            eval_short_np, eval_ICON_short = eval_short
        else:
            eval_short_np, eval_ICON_short = eval_short, None
    
        N = min(len(eval_long_np), len(eval_short_np))
        n = min(num_datapoints, N)
        rng = np.random.default_rng(random_state)
        idx = rng.choice(N, size=n, replace=False)
    
        sel_long       = eval_long_np[idx]
        sel_short      = eval_short_np[idx]
        sel_ICON_long  = eval_ICON_long[idx] if eval_ICON_long is not None else None
        sel_ICON_short = eval_ICON_short[idx] if eval_ICON_short is not None else None
    
        # 중요도 누적 dict (안전 갱신 대비)
        score_long, score_short = defaultdict(float), defaultdict(float)
    
        prev_rank_long, prev_rank_short = None, None
        stable_hits = 0
    
        for i in tqdm(range(n), desc="Processing samples"):
            long_point  = sel_long[i]
            short_point = sel_short[i]
            icon_long_point  = sel_ICON_long[i] if sel_ICON_long is not None else None
            icon_short_point = sel_ICON_short[i] if sel_ICON_short is not None else None
    
            exps = self.explain(
                data_point_long=long_point,
                data_point_short=short_point,
                eval_long=eval_long_np,
                eval_short=eval_short_np,
                num_samples=num_samples,
                is_visual=False,
                icon_point_long=icon_long_point,
                icon_point_short=icon_short_point
            )
    
            # 누적 시 안전 갱신 사용
            for feat_desc, imp in exps["long"]:
                name, _ = self._extract_feature_name_and_timestep(feat_desc)
                score_long[name] = score_long.get(name, 0.0) + abs(imp)
    
            for feat_desc, imp in exps["short"]:
                name, _ = self._extract_feature_name_and_timestep(feat_desc)
                score_short[name] = score_short.get(name, 0.0) + abs(imp)
    
            # --- 안정성 기반 조기 종료 ---
            if early_stop and ((i + 1) % check_interval == 0 or (i + 1) == n):
                cur_long  = self._topk_ranking(score_long,  top_n_for_long)
                cur_short = self._topk_ranking(score_short, top_n_for_short)
    
                if prev_rank_long is not None:
                    rhoL = self._spearman_rho_safe(prev_rank_long, cur_long)
                    rhoS = self._spearman_rho_safe(prev_rank_short, cur_short)
                    print(f"[stability] i={i+1}/{n} | rho_long={rhoL:.3f}, rho_short={rhoS:.3f}")
    
                    if rhoL >= rho_thresh and rhoS >= rho_thresh:
                        stable_hits += 1
                    else:
                        stable_hits = 0
    
                    if stable_hits >= 1:  # 연속 기준 원하면 숫자 올리기
                        print(f"[Early Stop] stabilized at {i+1}")
                        break
    
                prev_rank_long, prev_rank_short = cur_long, cur_short
    
        topL = sorted(score_long.items(),  key=lambda x: x[1], reverse=True)[:top_n_for_long]
        topS = sorted(score_short.items(), key=lambda x: x[1], reverse=True)[:top_n_for_short]
        return {"long": topL, "short": topS}


    def _extract_feature_name_and_timestep(self, feature_description: str):
        comp_op = r'(<=|>=|<|>)'
        alph_op = r'[a-zA-Z_]+\d*(?:_\d+)*'
    
        # 비교 연산자 제거
        cleaned = re.sub(comp_op, '', feature_description).strip()
        m = re.search(alph_op, cleaned)
        if not m:
            raise ValueError(f"Invalid feature description: {feature_description}")
    
        feature = m.group(0)
        actual, t = feature.rsplit('_', 1)
    
        # 정규화 적용
        actual = self._normalize_feature_name(actual)
    
        return actual, int(t)

    def _topk_ranking(self, d, k):
        return [k_ for k_, _ in sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]]

    def _spearman_rho_safe(self, rank_a, rank_b):
        """
        Spearman rho를 안전하게 계산 (합집합 기준).
        서로 다른 top-k 집합이여도 KeyError 없이 계산.
        """
        feats = set(rank_a) | set(rank_b)
        if len(feats) < 2:
            return 1.0
    
        pos_a = {feat: i for i, feat in enumerate(rank_a)}
        pos_b = {feat: i for i, feat in enumerate(rank_b)}
        fill_a, fill_b = len(rank_a), len(rank_b)
    
        r1 = np.array([pos_a.get(f, fill_a) for f in feats], dtype=float)
        r2 = np.array([pos_b.get(f, fill_b) for f in feats], dtype=float)
    
        s1, s2 = r1.std(), r2.std()
        if s1 == 0 or s2 == 0:
            return 1.0
        r1 = (r1 - r1.mean()) / s1
        r2 = (r2 - r2.mean()) / s2
        return float(np.clip(np.corrcoef(r1, r2)[0, 1], -1.0, 1.0))
        
    def _normalize_feature_name(self, name: str) -> str:
        """
        Feature 이름 정규화:
        - 소문자로 변환
        - 공백 제거
        - 하이픈(-)은 언더스코어(_)로 통일
        """
        return name.lower().replace(" ", "").replace("-", "_")