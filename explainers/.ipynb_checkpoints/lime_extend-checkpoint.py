import os
import re
import json
import numpy as np
import torch
import lime
import torch
from tqdm import tqdm
from lime import lime_tabular
import matplotlib.pyplot as plt 

from .base_explainer import BaseExplainer

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

class LimeExplainer(BaseExplainer):
    def __init__(self, model, device, sequences, sequence_length, input_size, selected_features, scaler,
                 icon_sequences=None):
        super().__init__(model, device, sequences, sequence_length, input_size, selected_features)
        self.sequences = self.sequences.cpu().numpy() if isinstance(self.sequences, torch.Tensor) else self.sequences
        if self.sequences is None or len(self.sequences) == 0:
            raise ValueError("train_sequences is None or empty.")

        # ★ 아이콘 시퀀스 보관 (없을 수도 있음)
        self.icon_sequences = icon_sequences
        if isinstance(self.icon_sequences, torch.Tensor):
            self.icon_sequences = self.icon_sequences.detach().cpu().numpy()

        self.sequence_length = sequence_length
        self.input_size = input_size
        self.selected_features = selected_features
        self.scaler = scaler

        self._icon_context = None  # ★ predict_fn에서 사용할 현재 배치용 icon 보관
        self.lime_tab = self.create_explainer()
        
    def create_explainer(self, max_samples=2500):
        reshaped = self.sequences.reshape(len(self.sequences), -1)
    
        # 학습 데이터가 많으면 일부만 샘플링
        if reshaped.shape[0] > max_samples:
            idx = np.random.choice(reshaped.shape[0], max_samples, replace=False)
            reshaped = reshaped[idx]
    
        return lime_tabular.LimeTabularExplainer(
            training_data=reshaped,
            feature_names=[f"{feature}_{i}" for i in range(self.sequence_length) for feature in self.selected_features],
            class_names=['Prediction'],
            mode='regression'
        )
        
    def predict_fn(self, input_data):
        """
        input_data: (B, S*F) 평탄화된 수치 특성들
        self._icon_context: (B, S) 또는 (B, S, ?) 아이콘 배치 (explain()에서 세팅)
        """
        # 수치 입력 복원
        X_num = input_data.reshape(-1, self.sequence_length, self.input_size)
        X_num = torch.tensor(X_num, dtype=torch.float32, device=self.device)

        # ★ 아이콘 입력 준비
        if self.icon_sequences is not None:
            if self._icon_context is None:
                raise ValueError("Model expects icon input, but _icon_context is not set.")
            X_icon = torch.tensor(self._icon_context, dtype=torch.long, device=self.device)
            outputs = self.model(X_num, X_icon)
        else:
            outputs = self.model(X_num)

        if isinstance(outputs, tuple):
            outputs = outputs[1]
        return outputs.detach().cpu().numpy()

    def explain(self, data_point, eval_sequences, num_features=10, num_samples=1000, is_visual=False,
                icon_point=None):
        """
        data_point: (S, F) 수치 시퀀스 1개
        eval_sequences: 평가용 수치 배열 또는 (X_num, X_icon)
        icon_point: (S,) 또는 (S, something) 현재 인스턴스의 아이콘 시퀀스. 모델이 icon을 쓰면 필수.
        """
        # eval unpack
        if isinstance(eval_sequences, (list, tuple)):
            eval_X = eval_sequences[0]
        else:
            eval_X = eval_sequences

        if isinstance(data_point, torch.Tensor):
            data_point = data_point.cpu().numpy()

        # ★ 아이콘 고정 컨텍스트 설정 (LIME가 생성하는 샘플 배치 크기에 맞춰 predict_fn에서 반복 사용)
        if self.icon_sequences is not None:
            if icon_point is None:
                raise ValueError("icon_vocab_size/use_icon=True 모델인데 icon_point가 없습니다.")
            icon_point = _to_numpy(icon_point)
            # lime 내부에서 predict_fn 호출 시 배치 크기 B가 달라짐 -> predict_fn 시작 시점에 맞춰
            # B개로 반복한 아이콘 배치를 만들 수 있도록 "원본 1개"를 보관
            # 여기서는 원본 1개만 보관하고, predict_fn 내부에서 B에 맞춰 tile 하는 대신,
            # simple하게 여기서 B를 알 수 없으므로 predict_fn에서 B 감지 후 tile 하도록 변경
            self._icon_point_single = icon_point  # (S,) or (S, ?)
        else:
            self._icon_point_single = None

        # predict_fn에서 B를 알아 tile 하기 위해 래퍼 구성
        def predict_with_icon(inverse_perturbed):
            # inverse_perturbed: (B, S*F)
            B = inverse_perturbed.shape[0]
            if self.icon_sequences is not None:
                # (S,) → (B, S)
                if self._icon_point_single.ndim == 1:
                    icon_batch = np.repeat(self._icon_point_single[None, :], B, axis=0)
                else:
                    # (S, D) 같은 형태면 앞에 B 반복
                    icon_batch = np.repeat(self._icon_point_single[None, ...], B, axis=0)
                self._icon_context = icon_batch
            return self.predict_fn(inverse_perturbed)

        specific_data_point = data_point.reshape(1, -1)
        specific_explanation = self.lime_tab.explain_instance(
            specific_data_point.flatten(),
            predict_with_icon,   # ★ 여기서 래퍼 사용
            num_samples=num_samples,
            num_features=250
        )

        important = specific_explanation.as_list()
        filtered = self._filter_time_features(important)

        if is_visual:
            self._draw_lime_bar_chart(filtered, top_k=10)
            self.visualize_lime_explanation(filtered, eval_X, self.sequence_length, self.selected_features)

        return important

    
    def _draw_lime_bar_chart(self, important_features, top_k=10):
        import matplotlib.pyplot as plt
    
        features, scores = zip(*important_features[:top_k])
        colors = ['green' if score > 0 else 'red' for score in scores]
    
        fig, ax = plt.subplots(figsize=(10, 3.5))  # 가로 길고 세로 얇게
        y_pos = range(len(features))
    
        ax.barh(y_pos, scores, align='center', color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=10)
        ax.invert_yaxis()  # 중요도 높은 것이 위로
        ax.set_title(f'Top {top_k} LIME Feature Importances', fontsize=12)
        ax.set_xlim(min(scores) - 0.01, max(scores) + 0.01)
        ax.grid(False)  # 격자 제거
    
        plt.tight_layout()
        plt.show()

    
    def _filter_time_features(self, important_features, exclude_keywords=['hour', 'day', 'month']):
        filtered = []
        for feature, score in important_features:
            if not any(key in feature for key in exclude_keywords):
                filtered.append((feature, score))
        return filtered


    def visualize_lime_explanation(self, important_features, sequences, sequence_length, selected_features):

        sequences = _to_numpy(sequences)
        
        feature_dict = {}
        comp_op = r'(<=|>=|<|>)'
        alph_op = r'[a-zA-Z_]+\d*(?:_\d+)*'

        # 정규화된 데이터를 원래 값으로 복구 (시간 피처 포함)
        # original_sequences = self.inverse_transform_time_features(sequences, selected_features, self.scaler)

        original_sequences = self.inverse_transform_time_features(
            sequences, selected_features, self.scaler, sequence_length=sequence_length
        )
        
        for feature_description, importance in important_features:
            feature_name = re.sub(comp_op, '', feature_description).strip()
            feature_name = re.search(alph_op, feature_name).group(0)
            actual_feature, time_step = feature_name.rsplit('_', 1)
            time_step = int(time_step)

            if actual_feature not in feature_dict:
                feature_dict[actual_feature] = []

            feature_value = original_sequences[0][time_step, selected_features.index(actual_feature)]
            feature_dict[actual_feature].append((time_step, feature_value, importance))

        # 시각화 수행
        for actual_feature, values in feature_dict.items():
            plt.figure(figsize=(10, 6))
            time_steps = [v[0] for v in values]
            actual_values = [v[1] for v in values]
            importances = [v[2] for v in values]

            plt.plot(range(sequence_length), original_sequences[0][:, selected_features.index(actual_feature)], label=f'{actual_feature}')
            plt.scatter(time_steps, actual_values, color='red', zorder=5, label='LIME Important Points')

            # for i, (x, y) in enumerate(zip(time_steps, actual_values)):
            #     plt.annotate(f'Value: {y:.3f}\nImp: {importances[i]:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', color='blue')

            plt.xlabel('Time Step')
            plt.ylabel('Feature Value')
            plt.title(f'Feature: {actual_feature}')
            plt.legend(loc='upper right')
            plt.grid(False)
            plt.show()
    
            # Print the detailed information about the important points
            print(f'\nFeature: {actual_feature} - Detailed Information')
            for time_step, value, importance in values:
                print(f'Time Step: {time_step}, Feature Value: {value:.3f}, Importance: {importance:.3f}')

    def extract_important_features(
        self, eval_data, top_n=5, num_samples=1000, num_datapoints=500,
        sample_fraction=None, max_samples=None,
        random_state=42, early_stop=False,
        check_interval=100, stability_k=20, rho_thresh=0.9
    ):
        """
        데이터 샘플링 후 LIME 기반 feature importance 추출
        """
        if isinstance(eval_data, (list, tuple)):
            eval_X, eval_ICON = eval_data[0], (eval_data[1] if len(eval_data) > 1 else None)
        else:
            eval_X, eval_ICON = eval_data, None

        eval_X = _to_numpy(eval_X)
        if eval_ICON is not None:
            eval_ICON = _to_numpy(eval_ICON)

        excluded = ['sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']
        # --- 초기화: 정규화된 selected_features 사용 ---
        score = {self._normalize_feature_name(f): 0.0 for f in self.selected_features}

        N = len(eval_X)
        if sample_fraction is not None:
            num_datapoints = int(N * sample_fraction)
        num_datapoints = min(num_datapoints, N)

        rng = np.random.default_rng(random_state)
        idx = rng.choice(N, size=num_datapoints, replace=False)

        selected_X = eval_X[idx]
        selected_ICON = eval_ICON[idx] if eval_ICON is not None else None

        # 안정성 체크용
        last_rank = None

        for i, data_point in enumerate(tqdm(selected_X, desc="Processing samples", total=num_datapoints)):
            icon_point = selected_ICON[i] if selected_ICON is not None else None

            explanation = self.explain(
                data_point=data_point,
                eval_sequences=(eval_X, eval_ICON),
                num_samples=num_samples,
                is_visual=False,
                icon_point=icon_point
            )

            for feat_desc, imp in explanation:
                name, _ = self._extract_feature_name_and_timestep(feat_desc)
                if name not in excluded:
                    # 안전 누적 (없는 키는 자동 생성)
                    score[name] = score.get(name, 0.0) + abs(imp)

            # 안정성 조기 종료 체크
            if (i + 1) % check_interval == 0 or (i + 1) == num_datapoints:
                top_now = self._topk_ranking(score, stability_k)
                if last_rank is not None and early_stop:
                    rho = self._spearman_rho_safe(last_rank, top_now)
                    print(f"[stability] i={i+1}/{num_datapoints}, Spearman rho(top{stability_k})={rho:.3f}")
                    if rho >= rho_thresh:
                        print("Early stopping triggered.")
                        break
                last_rank = top_now

        # 최종 top-N 반환
        top = sorted(score.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return {"single": top}
    
    def _extract_feature_name_and_timestep(self, feature_description: str):
        """
        LIME 피처 설명 문자열에서 feature 이름과 timestep을 추출
        """
        comp_op = r'(<=|>=|<|>)'
        alph_op = r'[a-zA-Z_]+\d*(?:_\d+)*'

        # 비교 연산자 제거
        cleaned_description = re.sub(comp_op, '', feature_description).strip()

        # feature 이름 추출
        match = re.search(alph_op, cleaned_description)
        if match:
            feature_name = match.group(0)
            # 시간 스텝 분리
            if "_" in feature_name:
                actual_feature, time_step = feature_name.rsplit("_", 1)
                actual_feature = self._normalize_feature_name(actual_feature)
                return actual_feature, int(time_step)
            else:
                actual_feature = self._normalize_feature_name(feature_name)
                return actual_feature, 0
        else:
            raise ValueError(f"Invalid feature description: {feature_description}")

    def _topk_ranking(self, d, k):
        return [k_ for k_, _ in sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]]

    def _spearman_rho_safe(self, rank_a, rank_b):
        """
        Spearman rho를 안전하게 계산 (합집합 기준)
        """
        feats = set(rank_a) | set(rank_b)
        pos_a = {feat: i for i, feat in enumerate(rank_a)}
        pos_b = {feat: i for i, feat in enumerate(rank_b)}

        r1, r2 = [], []
        for f in feats:
            r1.append(pos_a.get(f, len(rank_a)))
            r2.append(pos_b.get(f, len(rank_b)))

        r1, r2 = np.array(r1, dtype=float), np.array(r2, dtype=float)
        if r1.size < 2:
            return 1.0
        return np.corrcoef(r1, r2)[0, 1]
    
    def _normalize_feature_name(self, name: str) -> str:
        """
        피처 이름 정규화: 대소문자, 공백, 하이픈/언더스코어 차이 방지
        """
        return name.lower().replace(" ", "").replace("-", "_")
    