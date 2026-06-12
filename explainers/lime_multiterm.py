import os
import re
import json
import numpy as np
import torch
import lime
from tqdm import tqdm
from lime import lime_tabular
import matplotlib.pyplot as plt 

from .base_explainer import BaseExplainer

class LimeExplainer_MT(BaseExplainer):
    def __init__(self, model, device, train_long, train_short, sequence_length_long, sequence_length_short, 
                 input_size_long, input_size_short, selected_features_long, selected_features_short, scaler):
        """
        Args:
            model: PyTorch 모델.
            device: 모델이 동작하는 디바이스 (CPU 또는 GPU).
            train_long: Long-term 데이터 시퀀스.
            train_short: Short-term 데이터 시퀀스.
            sequence_length_long: Long-term 시퀀스 길이.
            sequence_length_short: Short-term 시퀀스 길이.
            input_size_long: Long-term 입력 피처의 크기.
            input_size_short: Short-term 입력 피처의 크기.
            selected_features_long: Long-term 선택된 피처 목록.
            selected_features_short: Short-term 선택된 피처 목록.
            scaler: 데이터 정규화를 위한 스케일러.
        """
        super().__init__(model, device, train_long, sequence_length_long, input_size_long, selected_features_long)
        self.train_long = train_long.cpu().numpy() if isinstance(train_long, torch.Tensor) else train_long
        self.train_short = train_short.cpu().numpy() if isinstance(train_short, torch.Tensor) else train_short
        self.sequence_length_long = sequence_length_long
        self.sequence_length_short = sequence_length_short
        self.input_size_long = input_size_long
        self.input_size_short = input_size_short
        self.selected_features_long = selected_features_long
        self.selected_features_short = selected_features_short
        self.scaler = scaler

        self.lime_tab_long = self.create_explainer(self.train_long, self.sequence_length_long, self.selected_features_long)
        self.lime_tab_short = self.create_explainer(self.train_short, self.sequence_length_short, self.selected_features_short)

    def create_explainer(self, train_data, sequence_length, selected_features):
        reshaped_data = train_data.reshape(len(train_data), -1)
        return lime_tabular.LimeTabularExplainer(
            training_data=reshaped_data,
            feature_names=[f"{feature}_{i}" for i in range(sequence_length) for feature in selected_features],
            class_names=['Prediction'],
            mode='regression'
        )

    def predict_fn(self, input_data, output_index):
        """
        LIME에서 사용하는 예측 함수.
        """
        long_input = input_data["long"].reshape(-1, self.sequence_length_long, self.input_size_long)
        short_input = input_data["short"].reshape(-1, self.sequence_length_short, self.input_size_short)

        self.model.eval()
        # long_input = torch.tensor(long_input, dtype=torch.float32).to(self.device)
        # short_input = torch.tensor(short_input, dtype=torch.float32).to(self.device)

        long_input = (
            torch.tensor(long_input, dtype=torch.float32).to(self.device)
            if not isinstance(long_input, torch.Tensor)
            else long_input.clone().detach().to(self.device)
        )
        short_input = (
            torch.tensor(short_input, dtype=torch.float32).to(self.device)
            if not isinstance(short_input, torch.Tensor)
            else short_input.clone().detach().to(self.device)
        )

        with torch.no_grad():
            outputs = self.model(long_input, short_input)
            if isinstance(outputs, tuple):
                selected_output = outputs[output_index]
            else:
                selected_output = outputs

            return selected_output.cpu().numpy()


    def explain(self, data_point_long, data_point_short, eval_long, eval_short, 
                num_samples=1000, is_visual=False):
        """
        Long/Short-term 데이터에 대한 설명 생성.
        Feature 중요도는 모든 feature에 대해서 뽑고, 시각화할때만 n개 추출
        """
        if isinstance(data_point_long, torch.Tensor):
            data_point_long = data_point_long.cpu().numpy()
        if isinstance(data_point_short, torch.Tensor):
            data_point_short = data_point_short.cpu().numpy()

        data_point_long = data_point_long.reshape(1, -1)
        data_point_short = data_point_short.reshape(1, -1)

        eval_long_sampled = eval_long[np.random.choice(len(eval_long), num_samples, replace=False)]
        eval_short_sampled = eval_short[np.random.choice(len(eval_short), num_samples, replace=False)]

        long_explanation = self.lime_tab_long.explain_instance(
            data_point_long.flatten(),
            lambda x: self.predict_fn({
                "long": x.reshape(-1, self.sequence_length_long, self.input_size_long),
                "short": eval_short_sampled
            }, output_index=0),  # Use outputs[0] for long-term explanation
            num_samples=num_samples,
            num_features=len(self.selected_features_long)
        )

        short_explanation = self.lime_tab_short.explain_instance(
            data_point_short.flatten(),
            lambda x: self.predict_fn({
                "long": eval_long_sampled,
                "short": x.reshape(-1, self.sequence_length_short, self.input_size_short)
            }, output_index=1),  # Use outputs[1] for short-term explanation
            num_samples=num_samples,
            num_features=len(self.selected_features_short)
        )

        if is_visual:
            long_explanation.show_in_notebook(show_table=True)
            self.visualize_lime_explanation(long_explanation.as_list(), 
                                            eval_long, 
                                            self.sequence_length_long, 
                                            self.selected_features_long, "Long")
    
            short_explanation.show_in_notebook(show_table=True)
            self.visualize_lime_explanation(short_explanation.as_list(), 
                                            eval_short, 
                                            self.sequence_length_short, 
                                            self.selected_features_short, "Short")
        return {
            "long": long_explanation.as_list(),
            "short": short_explanation.as_list()
        }

    def visualize_lime_explanation(self, important_features, sequences, sequence_length, selected_features, term):
        excluded_features = ['sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']
        
        # 제외할 feature 필터링
        filtered_features = [
            (feature, importance) 
            for feature, importance in important_features 
            if not any(excluded in feature for excluded in excluded_features)
        ]

        ## Dictionary to group features by their base name
        feature_dict = {}
        comp_op = r'(<=|>=|<|>)'
        alph_op = r'[a-zA-Z_]+\d*(?:_\d+)*'
    
        for feature_description, importance in filtered_features:
            feature_name = feature_description
            while re.search(comp_op, feature_name):
                feature_name = re.sub(comp_op, '', feature_name)
    
            feature_name = re.search(alph_op, feature_name)
            feature_name = feature_name.group(0)
    
            actual_feature, time_step = feature_name.rsplit('_', 1)
            time_step = int(time_step)
    
            # Add the feature to the dictionary
            if actual_feature not in feature_dict:
                feature_dict[actual_feature] = []
            
            feature_value = sequences[0][time_step, selected_features.index(actual_feature)]
            feature_dict[actual_feature].append((time_step, feature_value, importance))
    
        # Plot each feature with its important time steps
        for actual_feature, values in feature_dict.items():
            plt.figure(figsize=(10, 6))
            time_steps = [v[0] for v in values]
            actual_values = [v[1] for v in values]
            importances = [v[2] for v in values]
    
            plt.plot(range(sequence_length), sequences[0][:, selected_features.index(actual_feature)], label=f'{actual_feature}')
            plt.scatter(time_steps, actual_values, color='red', zorder=5, label='LIME Important Points')
    
            # Annotate the points with their importance and actual value
            for i, (x, y) in enumerate(zip(time_steps, actual_values)):
                plt.annotate(f'Value: {y:.3f}\nImp: {importances[i]:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', color='blue')
    
            plt.xlabel('Time Step')
            plt.ylabel('Feature Value')
            plt.title(f'{term} Term Feature: {actual_feature}')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.show()
    
            print(f'\nFeature: {actual_feature} ({term} Term) - Detailed Information')
            for time_step, value, importance in values:
                print(f'Time Step: {time_step}, Feature Value: {value:.3f}, Importance: {importance:.3f}')

    def extract_important_features(self, eval_long, eval_short, num_datapoints=50, top_n_for_long=5, top_n_for_short=5, num_samples=1000):
        excluded_features = ['sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']

        feature_importance_long = {feature: 0 for feature in self.selected_features_long}
        feature_importance_short = {feature: 0 for feature in self.selected_features_short}

        indices = np.random.choice(len(eval_long), num_datapoints, replace=False)
        selected_long = eval_long[indices]
        selected_short = eval_short[indices]

        for i, (long_point, short_point) in enumerate(tqdm(zip(selected_long, selected_short), desc="Processing samples", total=num_datapoints)):
            explanations = self.explain(
                data_point_long=long_point,
                data_point_short=short_point,
                eval_long=eval_long,
                eval_short=eval_short,
                num_samples=num_samples,
                is_visual=False
            )

            for feature_description, importance in explanations["long"]:
                actual_feature_name, _ = self._extract_feature_name_and_timestep(feature_description)
                if actual_feature_name not in excluded_features:
                    feature_importance_long[actual_feature_name] += abs(importance)

            for feature_description, importance in explanations["short"]:
                actual_feature_name, _ = self._extract_feature_name_and_timestep(feature_description)
                if actual_feature_name not in excluded_features:
                    feature_importance_short[actual_feature_name] += abs(importance)

            if i % 30 == 0:
                print("Important Long-term Features: ", feature_importance_long)
                print("Important Short-term Features: ", feature_importance_short)
                
        
        top_features_long = sorted(feature_importance_long.items(), key=lambda x: x[1], reverse=True)[:top_n_for_long]
        top_features_short = sorted(feature_importance_short.items(), key=lambda x: x[1], reverse=True)[:top_n_for_short]

        # top_n 개수 유지
        additional_long_features = [
            (feature, importance) for feature, importance in sorted(feature_importance_long.items(), key=lambda x: x[1], reverse=True)
            if feature not in dict(top_features_long)
        ][:max(0, top_n_for_long - len(top_features_long))]

        additional_short_features = [
            (feature, importance) for feature, importance in sorted(feature_importance_short.items(), key=lambda x: x[1], reverse=True)
            if feature not in dict(top_features_short)
        ][:max(0, top_n_for_short - len(top_features_short))]

        top_features_long.extend(additional_long_features)
        top_features_short.extend(additional_short_features)
        
        return {
            "long": top_features_long,
            "short": top_features_short
        }

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