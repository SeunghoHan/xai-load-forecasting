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

        self.lime_tab = self.create_explainer()
        self.scaler = scaler 
    

    # def create_explainer(self):
    #     return lime_tabular.LimeTabularExplainer(
    #         training_data=self.sequences.reshape(-1, self.sequence_length * self.input_size),
    #         feature_names=[f"{feature}_{i}" for i in range(self.sequence_length) for feature in self.selected_features],
    #         class_names=['Global_active_power'],
    #         mode='regression'
    #     )


    def create_explainer(self):
        reshaped_data = self.sequences.reshape(len(self.sequences), -1)
        return lime_tabular.LimeTabularExplainer(
            training_data=reshaped_data,
            feature_names=[f"{feature}_{i}" for i in range(self.sequence_length) for feature in self.selected_features],
            class_names=['Global_active_power'],
            mode='regression'
        )
        
    # def predict_fn(self, input_data):
    #     input_data = input_data.reshape(-1, self.sequence_length, self.input_size)
    #     self.model.eval()
    #     input_data = torch.tensor(input_data, dtype=torch.float32).to(self.device)
    #     with torch.no_grad():
    #         outputs = self.model(input_data)
    #         if type(outputs) == tuple:
    #             outputs = outputs[0]
    #     return outputs.cpu().numpy()

    def predict_fn(self, input_data):
        input_data = input_data.reshape(-1, self.sequence_length, self.input_size)
        self.model.eval()
        input_data = torch.tensor(input_data, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_data)
            if isinstance(outputs, tuple):
                outputs = outputs[1]  # Short-term prediction
        return outputs.cpu().numpy()
        

    def explain(self, data_point, eval_sequences, data_index=0, num_features=10, num_samples=1000, is_visual=False):
        specific_data_point = data_point.reshape(-1)
        specific_explanation = self.lime_tab.explain_instance(
            specific_data_point, self.predict_fn, num_samples=num_samples, num_features=num_features
        )
        important_features = specific_explanation.as_list()
        
        if is_visual:
            specific_explanation.show_in_notebook(show_table=True)
            self.visualize_lime_explanation(important_features, eval_sequences, 
                                            self.sequence_length, self.selected_features, data_index)
        
        return important_features

    def visualize_lime_explanation(self, important_features, sequences, sequence_length, selected_features, data_index=0):
        ## Dictionary to group features by their base name
        feature_dict = {}
        comp_op = r'(<=|>=|<|>)'
        alph_op = r'[a-zA-Z_]+\d*(?:_\d+)*'
    
        # 정규화된 데이터를 원래 값으로 복구 (시간 피처 포함)
        original_sequences = self.inverse_transform_time_features(sequences, selected_features, self.scaler)
    
        for feature_description, importance in important_features:
            feature_name = feature_description
            while re.search(comp_op, feature_name):
                feature_name = re.sub(comp_op, '', feature_name)
    
            feature_name = re.search(alph_op, feature_name)
            feature_name = feature_name.group(0)
    
            feature_index = self.lime_tab.feature_names.index(feature_name)
            actual_feature, time_step = feature_name.rsplit('_', 1)
            time_step = int(time_step)
    
            # Add the feature to the dictionary
            if actual_feature not in feature_dict:
                feature_dict[actual_feature] = []
            
            # Use the original (denormalized) sequence data for plotting
            feature_value = original_sequences[0][time_step, selected_features.index(actual_feature)]
            feature_dict[actual_feature].append((time_step, feature_value, importance))
    
        # Plot each feature with its important time steps
        for actual_feature, values in feature_dict.items():
            plt.figure(figsize=(10, 6))
            time_steps = [v[0] for v in values]
            actual_values = [v[1] for v in values]
            importances = [v[2] for v in values]
    
            plt.plot(range(sequence_length), original_sequences[0][:, selected_features.index(actual_feature)], label=f'{actual_feature}')
            plt.scatter(time_steps, actual_values, color='red', zorder=5, label='LIME Important Points')
    
            # Annotate the points with their importance and actual value
            for i, (x, y) in enumerate(zip(time_steps, actual_values)):
                plt.annotate(f'Value: {y:.3f}\nImp: {importances[i]:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', color='blue')
    
            plt.xlabel('Time Step')
            plt.ylabel('Feature Value')
            plt.title(f'Feature: {actual_feature}')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.show()
    
            # Print the detailed information about the important points
            print(f'\nFeature: {actual_feature} - Detailed Information')
            for time_step, value, importance in values:
                print(f'Time Step: {time_step}, Feature Value: {value:.3f}, Importance: {importance:.3f}')

    def extract_important_features(self, eval_sequences, num_samples=50, top_n=3):
        """
        여러 샘플에 대해 LIME을 수행하고, 중요도가 큰 상위 N개의 feature를 추출합니다.
        
        Args:
            eval_sequences: 평가 데이터셋
            num_samples: LIME을 적용할 샘플 수
            top_n: 중요도가 높은 상위 feature 수
        
        Returns:
            모델 예측에 많은 영향을 미치는 상위 feature 리스트
        """
        # 초기 feature 중요도를 담을 딕셔너리 생성
        feature_importance = {feature: 0 for feature in self.selected_features}
        print("Feature keys in importance dictionary:", feature_importance.keys())
        
        # 평가 데이터셋에서 랜덤으로 샘플 선택
        indices = np.random.choice(len(eval_sequences), num_samples, replace=False)
        selected_sequences = eval_sequences[indices]
        
        for i, data_point in enumerate(tqdm(selected_sequences, desc="Processing samples"), start=1):
            # 각 샘플에 대해 LIME 설명 생성
            explanation = self.explain(data_point, eval_sequences, num_features=len(self.selected_features))
            
            # 진행도 표시
            print(f"Processing sample {i}/{num_samples} ({(i / num_samples) * 100:.2f}%)")
            
            for feature_description, importance in explanation:
                # 뒤쪽의 `_숫자` 패턴만 제거
                actual_feature = [feature for feature in self.selected_features if feature in feature_description][0]
                
                # feature가 feature_importance에 존재하는 경우에만 추가
                if actual_feature in feature_importance:
                    feature_importance[actual_feature] += abs(importance)
                else:
                    print(f"Warning: {actual_feature} not found in selected features")
                    
            # 현재 상위 top_n 개 feature와 중요도 점수 출력
            current_top_features = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[:top_n]
            formatted_top_features = [f"Top{i+1} {feature} ({score:.4f})" for i, (feature, score) in enumerate(current_top_features)]
            print(f"\rCurrent Top {top_n} features: {', '.join(formatted_top_features)}", end="", flush=True)
    
        # 최종 중요도가 큰 상위 N개 feature와 해당 중요도 점수 추출
        final_top_features = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[:top_n]
        
        # 최종 결과 출력
        print("\nFinal Top Important Features with Scores:")
        for i, (feature, score) in enumerate(final_top_features):
            print(f"Top{i+1}: {feature} ({score:.4f})")
        
        return {feature: score for feature, score in final_top_features}
