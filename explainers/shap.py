import shap
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pdf
import seaborn as sns
import pandas as pd

class ModelOutputWrapper(torch.nn.Module):
    """Wrapper to get only the main output from a model with multiple outputs (e.g., predictions and attention weights)."""
    def __init__(self, model):
        super(ModelOutputWrapper, self).__init__()
        self.model = model
    
    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        # Assuming model output is a tuple (main_output, attention_weights)
        if isinstance(output, tuple):
            main_output = output[0]  # Get only the main prediction output
        else:
            main_output = output
        return main_output

class ShapExplainer:
    def __init__(self, model, train_data, sequence_length, input_size, selected_features, device, num_train_sample=100): 
        wrapped_model = ModelOutputWrapper(model).to(device)
        self.model = wrapped_model
        # self.model = model
        self.train_sequences = train_data
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.selected_features = selected_features
        self.device = device
        # self.explainer = shap.KernelExplainer(PyTorchLSTMModelWrapper(model, sequence_length, input_size, device), train_sequences[:num_train_sample].reshape(num_train_sample, -1), noise=0.1)

        # OOM 문제로 KernelExplainer -> DeepExplainer 로 변경
        background_data = torch.tensor(self.train_sequences[:num_train_sample], 
                                       dtype=torch.float32).to(self.device)
        self.explainer = shap.DeepExplainer(self.model, background_data)

    
    def explain_DeepExplainer(self, eval_sequences, num_samples=1):
        self.model.train()
        
        random_samples = eval_sequences[np.random.choice(len(eval_sequences), num_samples, replace=False)]
        random_samples_tensor = torch.tensor(random_samples, dtype=torch.float32).to(self.device)
        shap_values = self.explainer.shap_values(random_samples_tensor, check_additivity=False)
        
        return shap_values, random_samples

    def explain(self, eval_sequences, num_samples=1, nsamples=500, batch_size=1):
        random_samples = eval_sequences[np.random.choice(len(eval_sequences), num_samples, replace=False)].reshape(num_samples, -1)
        shap_values = self.explainer.shap_values(random_samples, nsamples=nsamples)
        return shap_values, random_samples
    
    def sequence_to_dataframe(self, sequence):
        # reshaped_sequence = sequence.reshape(self.sequence_length, self.input_size)
        # return pd.DataFrame(reshaped_sequence, columns=self.selected_features[:self.input_size])
        reshaped_sequence = sequence.reshape(self.sequence_length, -1)
        return pd.DataFrame(reshaped_sequence,
                            columns=self.selected_features[:reshaped_sequence.shape[1]])


    def plot_summary(self, shap_values, sample_df):
        print("Summary plot")

        feature_names = self.selected_features
        shap_values = np.mean(shap_values, axis=-1) 
        shap.summary_plot(shap_values, 
                          sample_df, 
                          feature_names=feature_names,
                          show=True)
    
    # def plot_force(self, shap_values, sample_df, target_idx):
    #     print("Force plot (target output index: {})".format(target_idx))

    #     feature_names = [f"{feature}_{i}" for i in range(self.sequence_length) for feature in self.selected_features]
    #     selected_shap_values = shap_values[ :, :, target_idx]  
    #     base_value = self.explainer.expected_value[target_idx] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value
    
    #     shap_values_explanation = shap.Explanation(
    #         values=selected_shap_values.flatten(),  
    #         base_values=base_value,  
    #         data=sample_df.values.flatten(),  
    #         feature_names=feature_names[:self.input_size * self.sequence_length]
    #     )
    
    #     # Force Plot 생성
    #     shap.force_plot(
    #         base_value=base_value,  # 모델의 기본 예측값
    #         shap_values=shap_values_explanation.values,  # 각 feature의 SHAP 값
    #         features=shap_values_explanation.data,  # 각 feature의 실제 값
    #         feature_names=shap_values_explanation.feature_names,  # feature 이름
    #         matplotlib=True, 
    #         contribution_threshold=0.02
    #     )
    def plot_force(self, shap_values, sample_df):
        print("Force plot")
        
        feature_names = [
            f"{feature}_{i}"
            for i in range(self.sequence_length)
            for feature in self.selected_features  # <- 이 부분이 filtered_feature_names 이어야 함
        ]
        
        # shap_values의 shape 확인 후 matching 되는 길이까지만 자르기
        expected_len = sample_df.values.flatten().shape[0]
        shap_values_flat = shap_values.flatten()[:expected_len]
        feature_names_flat = feature_names[:expected_len]
        
        shap_values_explanation = shap.Explanation(
            values=shap_values_flat,
            base_values=base_value,
            data=sample_df.values.flatten(),
            feature_names=feature_names_flat
        )
        
        shap.force_plot(
            base_value=base_value,
            shap_values=shap_values_explanation.values,
            features=shap_values_explanation.data,
            feature_names=shap_values_explanation.feature_names,
            matplotlib=True,
            contribution_threshold=0.02
        )
        

    # def plot_waterfall(self, shap_values, sample_df, target_idx):
    #     print("Waterfall plot (target output index: {})".format(target_idx))
    #     feature_names = [f"{feature}_{i}" for i in range(self.sequence_length) for feature in self.selected_features]
    #     selected_shap_values = shap_values[ :, :, target_idx]  
    #     base_value = self.explainer.expected_value[target_idx] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value
    
    #     shap_values_explanation = shap.Explanation(
    #         values=selected_shap_values.flatten(),  
    #         base_values=base_value,  
    #         data=sample_df.values.flatten(),  
    #         feature_names=feature_names[:self.input_size * self.sequence_length]
    #     )
    
    #     # Waterfall Plot 생성
    #     shap.waterfall_plot(shap_values_explanation, max_display=25)
    
    def plot_waterfall(self, shap_values, sample_df, target_idx):
        print("Waterfall plot (target output index: {})".format(target_idx))
    
        # feature 이름: 시간 순서로 평탄화
        feature_names = [
            f"{feature}_{i}" 
            for i in range(self.sequence_length) 
            for feature in self.selected_features
        ]
    
        # 해당 target_idx에 대한 shap 값만 추출 → shape: (sequence_len, input_dim)
        selected_shap_values = shap_values[:, :, target_idx]  # shape: (720, 7)
        shap_flat = selected_shap_values.flatten()            # shape: (5040,)
        data_flat = sample_df.values.flatten()                # shape: (5040,)
    
        assert shap_flat.shape == data_flat.shape, \
            f"SHAP ({shap_flat.shape}) != Data ({data_flat.shape})"
    
        base_value = (
            self.explainer.expected_value[target_idx]
            if isinstance(self.explainer.expected_value, (list, np.ndarray))
            else self.explainer.expected_value
        )
    
        explanation = shap.Explanation(
            values=shap_flat,
            base_values=base_value,
            data=data_flat,
            feature_names=feature_names
        )

        shap.waterfall_plot(explanation, max_display=25)
    
        
    
    # def plot_dependence(self, shap_values, sample_df, target_idx):
        # print("Dependence plot (target output index: {})".format(target_idx))

        # reshaped_shap_values = shap_values[:, :, target_idx]
        # for feature in self.selected_features:
        #     shap.dependence_plot(feature, reshaped_shap_values, sample_df)

    def plot_dependence(self, shap_values, sample_df, target_idx, target_f_idx, interaction_feature=None):
        feature_name = self.selected_features[target_f_idx]
        print(f"Dependence plot: {feature_name} (target output index: {target_idx})")
    
        try:
            reshaped_shap_values = shap_values[:, :, target_idx]
            if interaction_feature is not None:
                shap.dependence_plot(
                    ind=feature_name,
                    shap_values=reshaped_shap_values,
                    features=sample_df,
                    interaction_index=interaction_feature,
                    show=True
                )
            else:
                shap.dependence_plot(
                    ind=feature_name,
                    shap_values=reshaped_shap_values,
                    features=sample_df,
                    show=True
                )
        except Exception as e:
            print(f"[Error in dependence_plot for feature '{feature_name}']: {e}")
        
    
    # def plot_decision(self, shap_values, sample_df, target_idx):
    #     print("Decision plot (target output index: {})".format(target_idx))
    #     feature_names = [f"{feature}_{i}" for i in range(self.sequence_length) for feature in self.selected_features]
    #     selected_shap_values = shap_values[:, :, target_idx].flatten()  
    #     base_value = self.explainer.expected_value[target_idx] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value
        
    #     shap_values_explanation = shap.Explanation(
    #         values=selected_shap_values, 
    #         base_values=np.repeat(base_value, selected_shap_values.shape),  
    #         data=sample_df.values.flatten(), 
    #         feature_names=feature_names[:self.input_size * self.sequence_length]
    #     )
        
    #     shap.decision_plot(
    #         base_value=base_value, 
    #         shap_values=shap_values_explanation.values,  
    #         feature_names=feature_names[:self.input_size * self.sequence_length], 
    #         feature_display_range=slice(None, 25)
    #     )


    def plot_decision(self, shap_values, sample_df):
        print("Decision plot")
    
        feature_names = [f"{feature}_{i}" for i in range(self.sequence_length) for feature in sample_df.columns]
        base_value = self.explainer.expected_value if not isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value[0]
    
        shap_values_explanation = shap.Explanation(
            values=shap_values.flatten(),
            base_values=np.repeat(base_value, shap_values.size),
            data=sample_df.values.flatten(),
            feature_names=feature_names[:sample_df.shape[1] * self.sequence_length]
        )
    
        shap.decision_plot(
            base_value=base_value,
            shap_values=shap_values_explanation.values,
            feature_names=shap_values_explanation.feature_names,
            feature_display_range=slice(None, 25)
        )

    # def plot_scatter(self, shap_values, sample_df, target_idx):
    #     print(f"Scatter plot (target output index: {target_idx})")
        
    #     feature_names = [f"{feature}_{i}" for i in range(self.sequence_length) for feature in self.selected_features]
    #     selected_shap_values = shap_values[:, :, target_idx]

    #     shap_values_explanation = shap.Explanation(
    #         values=selected_shap_values, 
    #         base_values=np.repeat(self.explainer.expected_value, self.sequence_length),  
    #         data=sample_df.values, 
    #         feature_names=feature_names[:self.input_size * self.sequence_length]  
    #     )
    
    #     for feature in self.selected_features:
    #         feature_name = f"{feature}_0"
    #         feature_index = feature_names.index(feature_name)
    #         shap.plots.scatter(shap_values_explanation[feature_index])


    def plot_scatter(self, shap_values, sample_df):
        print("Scatter plot")
    
        feature_names = [f"{feature}_{i}" for i in range(self.sequence_length) for feature in sample_df.columns]
        
        shap_values_explanation = shap.Explanation(
            values=shap_values,
            base_values=np.repeat(self.explainer.expected_value[0], shap_values.shape[0]),
            data=sample_df.values,
            feature_names=sample_df.columns
        )
    
        for feature in sample_df.columns:
            shap.plots.scatter(shap_values_explanation[:, feature])

    
    def plot_bar(self, shap_values, sample_df):
        print("Bar plot")
        
        feature_names = self.selected_features
        selected_shap_values = np.mean(shap_values, axis=-1) 
        
        shap_values_explanation = shap.Explanation(
            values=selected_shap_values, 
            base_values=np.mean(self.explainer.expected_value),  
            data=sample_df.values, 
            feature_names=feature_names  
        )
        
        shap.plots.bar(shap_values_explanation)
    
    def plot_beeswarm(self, shap_values, sample_df):
        print("Beeswarm plot")
        
        feature_names = self.selected_features
        selected_shap_values = np.mean(shap_values, axis=-1) 
        
        shap_values_explanation = shap.Explanation(
            values=selected_shap_values, 
            base_values=np.mean(self.explainer.expected_value),  
            data=sample_df.values, 
            feature_names=feature_names  
        )
        
        shap.plots.beeswarm(shap_values_explanation)
    
    def plot_heatmap(self, shap_values, sample_df):
        print("Heatmap plot")
        
        feature_names = self.selected_features
        selected_shap_values = np.mean(shap_values, axis=-1) 
        
        shap_values_explanation = shap.Explanation(
            values=selected_shap_values, 
            base_values=np.mean(self.explainer.expected_value),  
            data=sample_df.values, 
            feature_names=feature_names  
        )

        shap.plots.heatmap(shap_values_explanation)

    # def plot_partial_dependence(self, shap_values, sample_df, target_feature_idx, target_idx):
    #     print("Partial Dependence Plot (feature: {}, target output index)".format(self.selected_features[target_feature_idx], target_idx))
    
    #     feature_names = [f"{feature}_{i}" for i in range(self.sequence_length) for feature in self.selected_features]
    #     selected_shap_values = shap_values[:, :, target_idx]
        
    #     if self.selected_features[target_feature_idx] in sample_df.columns:
    #         # Partial Dependence Plot 생성
    #         shap.dependence_plot(target_feature_idx, selected_shap_values, sample_df)
    #     else:
    #         print(f"Feature {target_feature_name} not found in sample_df columns: {sample_df.columns}")


    def plot_partial_dependence(self, shap_values, sample_df, target_feature_idx):
        feature_name = sample_df.columns[target_feature_idx]
        print(f"Partial Dependence Plot: {feature_name}")
    
        try:
            shap.dependence_plot(
                feature_name,
                shap_values,
                sample_df,
                interaction_index=None,
                show=True
            )
        except Exception as e:
            print(f"[Error in PDP]: {e}")

    def visualize_all_one_sample(self, shap_values, eval_sequences, target_f_idx1, target_f_idx2, target_idx=0):
        # target_f_idx1: for PDP
        # target_f_idx1, target_f_idx2: interaction plot
        # target_idx: target output index
        shap.initjs()

        
        ## 시간 관련 feature 제거
        filtered_shap_values, filtered_eval_sequence, filtered_feature_names = self.filter_features(shap_values, eval_sequences)
    
        sample_df = self.sequence_to_dataframe(filtered_eval_sequence)
        self.selected_features = filtered_feature_names
            
        self.plot_summary(filtered_shap_values, sample_df)    
        # self.plot_force(filtered_shap_values, sample_df)
        # self.plot_waterfall(filtered_shap_values, sample_df, target_idx=0)
        # self.plot_decision(filtered_shap_values, sample_df)
        # self.plot_scatter(filtered_shap_values, sample_df)
        # self.plot_bar(filtered_shap_values, sample_df)
        # self.plot_beeswarm(filtered_shap_values, sample_df)
        # self.plot_heatmap(filtered_shap_values, sample_df)
        # self.plot_partial_dependence(filtered_shap_values, sample_df, target_f_idx1)

        # dependence 따로 출력
        dependence_plot_configs = [
            {"target_feature": "Global_active_power", "interaction_feature": "Global_intensity"},
            {"target_feature": "Sub_metering_3", "interaction_feature": "Global_active_power"},
            {"target_feature": "Humidity", "interaction_feature": "Temperature"},
            {"target_feature": "Temperature", "interaction_feature": "Global_active_power"},
            # {"target_feature": "Humidity", "interaction_feature": "Global_active_power"},
            # {"target_feature": "Temperature", "interaction_feature": "Global_intensity"},
        ]
        # 전체 소비량이 커질 때 전류 변화에 따라 모델 출력 영향이 달라지는가
        # 세 번째 부하 영역의 소비가 전체 소비량에 따라 얼마나 중요한지
        # 외부 환경 변수 간의 상호작용이 모델 예측에 어떻게 작용하는지
        # 서로 다른 영역의 소비가 상호작용하며 영향을 주는지
        # 전류량이 특정 부하에서의 소비에 따라 다른 영향을 주는지

        # target_idx는 대부분 0 (단일 출력 모델 기준)
        target_idx = 0
        
        # 각 조합에 대해 plot_dependence 호출
        for config in dependence_plot_configs:
            target_f_idx = self.selected_features.index(config["target_feature"])
            self.plot_dependence(
                shap_values=filtered_shap_values,
                sample_df=sample_df,
                target_idx=target_idx,
                target_f_idx=target_f_idx,
                interaction_feature=config["interaction_feature"]
            )
    

    
        
        # self.plot_summary(shap_values, sample_df)
        # self.plot_force(shap_values, sample_df, target_idx)
        # self.plot_waterfall(shap_values, sample_df, target_idx)
        # self.plot_dependence(shap_values, sample_df, target_idx)
        # self.plot_decision(shap_values, sample_df, target_idx)
        # self.plot_scatter(shap_values, sample_df, target_idx)
        # self.plot_bar(shap_values, sample_df)
        # self.plot_beeswarm(shap_values, sample_df)
        # self.plot_heatmap(shap_values, sample_df)
        # self.plot_partial_dependence(shap_values, sample_df, target_f_idx1, target_idx)


    def get_non_time_feature_indices(self):
        return [i for i, f in enumerate(self.selected_features) if all(tf not in f for tf in ['sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month'])]

    def filter_features(self, shap_values, eval_sequence):
        indices = self.get_non_time_feature_indices()
        filtered_shap_values = shap_values[:, indices]
        filtered_eval_sequence = eval_sequence[:, indices]
        filtered_feature_names = [self.selected_features[i] for i in indices]
        return filtered_shap_values, filtered_eval_sequence, filtered_feature_names
    
    
    def visualize_all_multiple_samples(self, shap_values, eval_sequences, target_f_idx1, target_f_idx2, target_idx=0):
        """
            여러 샘플에 대한 시각화를 수행하는 함수.
        """
        shap.initjs()
        # feature_names = [f"{feature}_{i}" for i in range(self.sequence_length) for feature in self.selected_features]

        # 여러 샘플에 대해 SHAP 값을 평균 계산
        shap_values_mean = np.mean(np.abs(shap_values), axis=0)
        eval_sequences_mean = np.mean(eval_sequences, axis=0)
        sample_df = self.sequence_to_dataframe(eval_sequences_mean)

        self.plot_summary(shap_values_mean, sample_df)
        self.plot_force(shap_values_mean, sample_df, target_idx)
        self.plot_waterfall(shap_values_mean, sample_df, target_idx)
        self.plot_dependence(shap_values_mean, sample_df, target_idx)
        self.plot_decision(shap_values_mean, sample_df, target_idx)
        self.plot_scatter(shap_values_mean, sample_df, target_idx)
        self.plot_bar(shap_values_mean, sample_df)
        self.plot_beeswarm(shap_values_mean, sample_df)
        self.plot_heatmap(shap_values_mean, sample_df)
        self.plot_partial_dependence(shap_values_mean, sample_df, target_f_idx1, target_idx)

    def explain_correlation(self, eval_sequences, num_samples=1):
        random_samples = eval_sequences[np.random.choice(len(eval_sequences), num_samples, replace=False)]
        random_samples_tensor = torch.tensor(random_samples, dtype=torch.float32).to(self.device)
        shap_values = self.explainer.shap_values(random_samples.reshape(num_samples, -1))
        return shap_values, random_samples

    def plot_dependence_correlation(self, shap_values, sample_df, target_feature, interaction_feature=None):
        """
        SHAP Dependence Plot: 특정 feature와 선택한 feature의 상호작용을 보여주는 플롯.
        target_feature: 분석할 feature
        interaction_feature: 상호작용할 feature (선택적)
        """
        print(f"Dependence plot for {target_feature}")
    
        # Feature 이름 확인 및 변환
        feature_names = self.selected_features # 샘플 데이터프레임의 컬럼명을 feature 이름으로 사용
    
        if target_feature not in feature_names:
            raise ValueError(f"Could not find feature named: {target_feature}")
    
        if interaction_feature  not in feature_names:
            raise ValueError(f"Could not find interaction feature named: {interaction_feature}")

        if shap_values.shape[0] != sample_df.shape[0]:
            shap_values = shap_values.reshape(sample_df.shape[0], -1)
    
        # SHAP dependence plot 호출
        shap.dependence_plot(target_feature, shap_values, sample_df, interaction_index=interaction_feature)

    def plot_shap_value_correlation(self, shap_values):
        """
        SHAP Value Correlation Plot: SHAP 값 간의 상관관계를 보여주는 플롯.
        """
        print("SHAP Value Correlation Plot")
        shap_values_reshaped = shap_values.reshape(self.sequence_length, self.input_size)
        shap_corr = np.corrcoef(shap_values_reshaped, rowvar=False)

        plt.figure(figsize=(10, 8))
        sns.heatmap(shap_corr, annot=True, cmap='coolwarm', xticklabels=self.selected_features, yticklabels=self.selected_features, vmin=-1, vmax=1)
        plt.title("Correlation of SHAP Values (Feature Contribution)")
        plt.show()

    def visualize_correlation(self, shap_values, eval_sequences, target_feature_1, target_feature_2):
        # feature_names = [f"{feature}_{i}" for i in range(self.sequence_length) for feature in self.selected_features]

        sample_df = self.sequence_to_dataframe(eval_sequences)
        
        # 24개 아웃풋에 대해서 평균으로 해서 출력에 대한 feature 상관관계 분석
        shap_values = np.mean(shap_values, axis=-1) 
        
        # self.plot_dependence_correlation(shap_values, sample_df, 
        #                                  target_feature=target_feature_1, 
        #                                  interaction_feature=target_feature_2)  
        
        self.plot_shap_value_correlation(shap_values)  


    def plot_shap_value_correlation_multiple_samples(self, shap_values):
        """
        여러 샘플에 대한 SHAP 값들의 상관관계를 분석하는 함수.
        """
        feature_names = self.selected_features
        
        # 샘플 전체의 SHAP 값을 평균으로 변환
        shap_values_reshaped = np.mean(np.abs(shap_values), axis=0).reshape(len(feature_names), -1)
    
        # SHAP 값 간 상관 행렬 계산
        shap_corr = np.corrcoef(shap_values_reshaped)
    
        # 상관 행렬 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(shap_corr, annot=True, cmap='coolwarm', xticklabels=feature_names, yticklabels=feature_names, vmin=-1, vmax=1)
        plt.title("SHAP Value Correlation Matrix (Feature Contribution)")
        plt.show()

    def _visualize_correlation_multiple_samples(self, shap_values):

        # shap_values의 크기를 확인하고 필요 시 reshape
        shap_values_mean = np.mean(np.abs(shap_values), axis=0)
        shap_values_mean = np.mean(shap_values_mean, axis=-1) 
        
        # 확인: shap_values_mean의 크기가 올바른지 출력
        print(f"shap_values_mean shape: {shap_values_mean.shape}")
        
        shap_values_reshaped = shap_values_mean.reshape(self.sequence_length, self.input_size)

        # feature_names에 맞게 sample_df 생성
        sample_df = pd.DataFrame(shap_values_reshaped, columns=self.selected_features)
        
        # SHAP 값들의 상관관계 행렬 계산 및 시각화
        self.plot_shap_value_correlation_multiple_samples(shap_values)

    def visualize_correlation_multiple_samples(self, shap_values):
        """
        여러 샘플에 대한 SHAP 값들의 상관관계를 분석하는 함수.
        개별 샘플에 대해 상관 행렬을 구하고, 그 행렬들의 평균을 사용합니다.
        """
        # 개별 샘플 상관 행렬을 저장할 리스트
        individual_corrs = []
        valid_samples = 0  # 유효한 샘플 개수
    
        for sample_shap_values in shap_values:
            # 개별 샘플을 (sequence_length, input_size)로 reshape
            # 24개 아웃풋에 대해서 평균으로 해서 출력에 대한 feature 상관관계 분석
            sample_shap_values = np.mean(sample_shap_values, axis=-1) 
            sample_shap_values_reshaped = sample_shap_values.reshape(self.sequence_length, 
                                                                     self.input_size)
            
            # 상관 행렬 계산
            sample_corr = np.corrcoef(sample_shap_values_reshaped, rowvar=False)
            individual_corrs.append(sample_corr)
            valid_samples += 1  # 유효 샘플 카운트 증가
        
    
        # 유효한 상관 행렬의 평균 계산
        if valid_samples > 0:
            avg_corr = np.mean(individual_corrs, axis=0)
    
            # 상관 행렬 시각화
            plt.figure(figsize=(10, 8))
            sns.heatmap(avg_corr, annot=True, cmap='coolwarm', xticklabels=self.selected_features, yticklabels=self.selected_features, vmin=-1, vmax=1)
            plt.title("Average SHAP Value Correlation Matrix (Feature Contribution)")
            plt.show()
        else:
            print("No valid samples for correlation calculation.")


    def extract_important_features(self, shap_values, top_n=5):
        """
        여러 샘플에 대해 SHAP을 수행한 후 중요도가 높은 상위 N개의 feature를 추출합니다.
        """
        # 모든 샘플에 대해 SHAP 값의 절대값 평균 계산
        shap_values_mean = np.mean(np.abs(shap_values[0]), axis=0)
        feature_importance = {feature: 0 for feature in self.selected_features}
        
        # 각 feature의 중요도를 누적
        for i, feature in enumerate(self.selected_features):
            feature_importance[feature] = np.mean(shap_values_mean[:, i])

        # 상위 top_n feature 추출
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_features_dict = {feature: importance for feature, importance in top_features}
        
        print(f"Top {top_n} important features based on SHAP values:")
        for rank, (feature, importance) in enumerate(top_features, start=1):
            print(f"Top {rank}: {feature} (Importance: {importance})")
        
        return top_features_dict
        
