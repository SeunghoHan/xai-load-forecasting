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

class LimeExplainer(BaseExplainer):
    def __init__(self, model, device, sequences, sequence_length, input_size, selected_features, scaler):
        super().__init__(model, device, sequences, sequence_length, input_size, selected_features)
        
        self.sequences = self.sequences.cpu().numpy() if isinstance(self.sequences, torch.Tensor) else self.sequences
        
        if self.sequences is None or len(self.sequences) == 0:
            raise ValueError("train_sequences is None or empty. Please check your data preprocessing.")


        self.sequence_length = sequence_length
        self.input_size = input_size
        self.selected_features = selected_features
        self.scaler = scaler

        self.lime_tab = self.create_explainer()
        
    def create_explainer(self):
        reshaped_data = self.sequences.reshape(len(self.sequences), -1)
        return lime_tabular.LimeTabularExplainer(
            training_data=reshaped_data,
            feature_names=[f"{feature}_{i}" for i in range(self.sequence_length) for feature in self.selected_features],
            class_names=['Prediction'],
            mode='regression'
        )
        
    def predict_fn(self, input_data):
        input_data = input_data.reshape(-1, self.sequence_length, self.input_size)
        self.model.eval()
        input_data = torch.tensor(input_data, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_data)
            if isinstance(outputs, tuple):
                outputs = outputs[1]  # Short-term prediction
        
        return outputs.cpu().numpy()

    def explain(self, data_point, eval_sequences, num_features=10, num_samples=1000, is_visual=False):
        if isinstance(data_point, torch.Tensor):
            data_point = data_point.cpu().numpy()

        specific_data_point = data_point.reshape(1, -1)

        specific_explanation = self.lime_tab.explain_instance(
            specific_data_point.flatten(),
            self.predict_fn,
            num_samples=num_samples,
            num_features=250  # 상위 N개 추출
            # num_features=len(self.selected_features)  # 모든 피처 고려
        )
        
        important_features = specific_explanation.as_list()
        
        filtered_important_features = self._filter_time_features(important_features)

        if is_visual:
            self._draw_lime_bar_chart(filtered_important_features, top_k=10)

            self.visualize_lime_explanation(filtered_important_features, eval_sequences, 
                                            self.sequence_length, self.selected_features)
        
            
            # specific_explanation.show_in_notebook(show_table=True)
            # specific_explanation.as_pyplot_figure()
            
            # self.visualize_lime_explanation(important_features, eval_sequences, 
            #                                 self.sequence_length, self.selected_features)
        
        return important_features

    
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
        
        feature_dict = {}
        comp_op = r'(<=|>=|<|>)'
        alph_op = r'[a-zA-Z_]+\d*(?:_\d+)*'

        # 정규화된 데이터를 원래 값으로 복구 (시간 피처 포함)
        original_sequences = self.inverse_transform_time_features(sequences, selected_features, self.scaler)

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

    def extract_important_features(self, eval_data, num_datapoints=50, top_n=5, num_samples=1000):
        """
        여러 샘플에 대해 LIME을 수행하고, 중요도가 큰 상위 N개의 feature를 추출합니다.
    
        Args:
            eval_data: 평가 데이터셋 (단일 term)
            num_datapoints: LIME을 적용할 샘플 수
            top_n: 중요도가 높은 상위 feature 수
            num_samples: LIME에서 샘플링할 데이터 개수num_datapoints
    
        Returns:
            모델 예측에 많은 영향을 미치는 상위 feature 리스트
        """
        excluded_features = ['sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']
        feature_importance = {feature: 0 for feature in self.selected_features}
    
        # 평가 데이터에서 무작위 샘플 선택
        indices = np.random.choice(len(eval_data), num_datapoints, replace=False)
        selected_data = eval_data[indices]
    
        for i, data_point in enumerate(tqdm(selected_data, desc="Processing samples", total=num_datapoints)):
            explanation = self.explain(
                data_point=data_point,
                eval_sequences=eval_data,
                num_samples=num_samples,
                is_visual=False
            )
    
            for feature_description, importance in explanation:
                actual_feature_name, _ = self._extract_feature_name_and_timestep(feature_description)
                if actual_feature_name not in excluded_features:
                    feature_importance[actual_feature_name] += abs(importance)
    
            if i % 30 == 0:
                print(f"Processing {i}/{num_datapoints} samples...")
                print("Important Features: ", feature_importance)
    
        # 상위 N개의 feature 선택
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
        # 부족한 경우 추가 feature 채우기
        additional_features = [
            (feature, importance) for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            if feature not in dict(top_features)
        ][:max(0, top_n - len(top_features))]
    
        top_features.extend(additional_features)
    
        return {"single": top_features}

    def _extract_feature_name_and_timestep(self, feature_description):
        comp_op = r'(<=|>=|<|>)'
        alph_op = r'[a-zA-Z_]+\d*(?:_\d+)*'
    
        # 비교 연산자와 숫자 범위를 제거
        cleaned_description = re.sub(comp_op, '', feature_description).strip()
    
        # feature 이름 추출
        match = re.search(alph_op, cleaned_description)
        if match:
            feature_name = match.group(0)
            # 시간 스텝 분리
            actual_feature, time_step = feature_name.rsplit('_', 1)
            return actual_feature, int(time_step)
        else:
            raise ValueError(f"Invalid feature description: {feature_description}")