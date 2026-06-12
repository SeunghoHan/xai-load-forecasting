import shap
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class PyTorchLSTMModelWrapper:
    def __init__(self, model, sequence_length, input_size, device):
        self.model = model
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.device = device

    def __call__(self, x):
        x = x.reshape((x.shape[0], self.sequence_length, self.input_size))
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(x).cpu().numpy()
            
    def predict(self, x):
        x = x.reshape((x.shape[0], self.sequence_length, self.input_size))
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(x).cpu().numpy().flatten()

class ShapExplainer:
    def __init__(self, model, train_sequences, sequence_length, input_size, selected_features, device, num_train_sample=100):
        self.model = model
        self.train_sequences = train_sequences
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.selected_features = selected_features
        self.device = device
        self.explainer = shap.KernelExplainer(PyTorchLSTMModelWrapper(model, sequence_length, input_size, device), train_sequences[:num_train_sample].reshape(num_train_sample, -1))

    
    # def explain(self, eval_sequences, num_samples=1, nsamples=10):
    #     random_samples = eval_sequences[np.random.choice(len(eval_sequences), num_samples, replace=False)].reshape(num_samples, -1)
    #     shap_values = self.explainer.shap_values(random_samples, nsamples=nsamples)
    #     return shap_values, random_samples

    def explain(self, eval_sequences, num_samples=1, nsamples=500, batch_size=1):
        random_samples = eval_sequences[np.random.choice(len(eval_sequences), num_samples, replace=False)].reshape(num_samples, -1)
            
        shap_values = self.explainer.shap_values(random_samples, nsamples=nsamples)

        # shap_values_list = []
    
        # for i in range(0, num_samples, batch_size):
        #     batch_sequences = random_samples[i:i+batch_size]
        #     shap_values_batch = self.explainer.shap_values(batch_sequences, nsamples=nsamples)
        #     shap_values_list.append(shap_values_batch)
    
        # shap_values = np.concatenate(shap_values_list, axis=0)
    
        return shap_values, random_samples
        

    def sequence_to_dataframe(self, sequence, feature_names):
        reshaped_sequence = sequence.reshape(self.sequence_length, self.input_size)
        return pd.DataFrame(reshaped_sequence, columns=feature_names[:self.input_size])

    def plot_summary(self, shap_values, sample_df, feature_names):
        print("Summary plot")
        shap.summary_plot(shap_values.reshape(self.sequence_length, -1), 
                          sample_df, 
                          feature_names=feature_names[:self.input_size * self.sequence_length], 
                          show=True)

    def plot_force(self, shap_values, sample_df, feature_names, sample_index):
        print("Force plot")
        shap_values_explanation = shap.Explanation(
            values=shap_values.flatten(), 
            base_values=np.repeat(self.explainer.expected_value, self.sequence_length * self.input_size), 
            data=sample_df.values.flatten(), 
            feature_names=feature_names[:self.sequence_length * self.input_size]
        )
        shap.force_plot(
            base_value=self.explainer.expected_value, 
            shap_values=shap_values_explanation.values, 
            features=shap_values_explanation.data, 
            feature_names=shap_values_explanation.feature_names,
            matplotlib=True
        )

    def plot_waterfall(self, shap_values, sample_df, feature_names, sample_index):
        print("Waterfall plot")
        shap_values_explanation = shap.Explanation(
            values=shap_values.flatten(), 
            base_values=self.explainer.expected_value, 
            data=sample_df.values.flatten(), 
            feature_names=feature_names[:self.input_size * self.sequence_length]
        )
        shap.waterfall_plot(shap_values_explanation)

    def plot_dependence(self, shap_values, sample_df, sample_index):
        print("Dependence plot")
        for feature in self.selected_features:
            feature_name = f"{feature}_0"
            shap.dependence_plot(feature_name, shap_values.reshape(self.sequence_length, -1), sample_df)

    def plot_decision(self, shap_values, sample_df, feature_names, sample_index):
        print("Decision plot")
        shap_values_explanation = shap.Explanation(
            values=shap_values.flatten(), 
            base_values=self.explainer.expected_value, 
            data=sample_df.values.flatten(), 
            feature_names=feature_names[:self.input_size * self.sequence_length]
        )
        shap.decision_plot(self.explainer.expected_value, shap_values_explanation.values, feature_names=feature_names[:self.input_size * self.sequence_length])

    def plot_scatter(self, shap_values, sample_df, feature_names, sample_index):
        print("Scatter plot")
        shap_values_explanation = shap.Explanation(
            values=shap_values.reshape(self.sequence_length, -1), 
            base_values=np.repeat(self.explainer.expected_value, self.sequence_length), 
            data=sample_df.values, 
            feature_names=feature_names[:self.input_size]
        )
        for feature in self.selected_features:
            feature_name = f"{feature}_0"
            feature_index = feature_names.index(feature_name)
            shap.plots.scatter(shap_values_explanation[:, feature_index])
            

    def plot_bar(self, shap_values, sample_df, feature_names, sample_index):
        print("Bar plot")
        shap_values_explanation = shap.Explanation(
            values=shap_values.reshape(self.sequence_length, -1), 
            base_values=self.explainer.expected_value, 
            data=sample_df.values, 
            feature_names=feature_names[:self.input_size * self.sequence_length]
        )
        shap.plots.bar(shap_values_explanation)

    def plot_beeswarm(self, shap_values, sample_df, feature_names, sample_index):
        print("Beaswarm plot")
        shap_values_explanation = shap.Explanation(
            values=shap_values.reshape(self.sequence_length, -1), 
            base_values=self.explainer.expected_value, 
            data=sample_df.values, 
            feature_names=feature_names[:self.input_size * self.sequence_length]
        )
        shap.plots.beeswarm(shap_values_explanation)

    def plot_heatmap(self, shap_values, sample_df, feature_names, sample_index):
        print("Heatamp plot")
        shap_values_explanation = shap.Explanation(
            values=shap_values.reshape(self.sequence_length, -1), 
            base_values=self.explainer.expected_value, 
            data=sample_df.values, 
            feature_names=feature_names[:self.input_size * self.sequence_length]
        )
        shap.plots.heatmap(shap_values_explanation)

    def plot_partial_dependence(self, shap_values, sample_df, feature_name, sample_index):
        print("Partial dependence plot")
        shap_values_explanation = shap.Explanation(
            values=shap_values.flatten(), 
            base_values=self.explainer.expected_value, 
            data=sample_df.values.flatten(), 
            feature_names=[feature_name]
        )
        if feature_name in sample_df.columns:
            shap.dependence_plot(feature_name, shap_values.reshape(self.sequence_length, self.input_size), sample_df)
        else:
            print(f"Feature {feature_name} not found in sample_df columns: {sample_df.columns}")

    def plot_interaction(self, shap_values, sample_df, sample_index):
        print("Interaction plot")
        """
        Interaction plot to show how two features interact and affect the predictions.
        """
        # Calculate interaction values using SHAP
        interaction_values = self.explainer.shap_interaction_values(sample_df)
        
        # Select two features to plot their interaction
        feature_1 = self.selected_features[0]
        feature_2 = self.selected_features[1]

        # Plot the interaction between the two features
        shap.dependence_plot(
            (self.selected_features.index(feature_1), self.selected_features.index(feature_2)), 
            interaction_values[sample_index], 
            sample_df
        )

    def visualize_all_one_sample(self, shap_values, eval_sequences):
        shap.initjs()
        feature_names = [f"{feature}_{i}" for i in range(self.sequence_length) for feature in self.selected_features]

        for i in range(len(eval_sequences)):
            sample_df = self.sequence_to_dataframe(eval_sequences[i], feature_names)
            self.plot_summary(shap_values[i], sample_df, feature_names)
            self.plot_force(shap_values[i], sample_df, feature_names, i)
            self.plot_waterfall(shap_values[i], sample_df, feature_names, i)
            self.plot_dependence(shap_values[i], sample_df, i)
            self.plot_decision(shap_values[i], sample_df, feature_names, i)
            self.plot_scatter(shap_values[i], sample_df, feature_names, i)
            self.plot_bar(shap_values[i], sample_df, feature_names, i)
            self.plot_beeswarm(shap_values[i], sample_df, feature_names, i)
            self.plot_heatmap(shap_values[i], sample_df, feature_names, i)
            self.plot_partial_dependence(shap_values[i], sample_df, self.selected_features[0], i)
        
    def visualize_all_multiple_samples(self, shap_values, eval_sequences):
        """
        여러 샘플에 대한 시각화를 수행하는 함수.
        """
        shap.initjs()
        feature_names = [f"{feature}_{i}" for i in range(self.sequence_length) for feature in self.selected_features]

        # 여러 샘플에 대해 SHAP 값을 평균 계산
        shap_values_mean = np.mean(np.abs(shap_values), axis=0)
        eval_sequences_mean = np.mean(eval_sequences, axis=0)
        sample_df = self.sequence_to_dataframe(eval_sequences_mean, feature_names)
        
        # 여러 샘플에 대한 시각화
        self.plot_summary(shap_values_mean, sample_df, feature_names)
        self.plot_force(shap_values_mean, sample_df, feature_names, 0)
        self.plot_waterfall(shap_values_mean, sample_df, feature_names, 0)
        self.plot_dependence(shap_values_mean, sample_df, 0)
        self.plot_decision(shap_values_mean, sample_df, feature_names, 0)
        self.plot_scatter(shap_values_mean, sample_df, feature_names, 0)
        self.plot_bar(shap_values_mean, sample_df, feature_names, 0)
        self.plot_beeswarm(shap_values_mean, sample_df, feature_names, 0)
        self.plot_heatmap(shap_values_mean, sample_df, feature_names, 0)
        self.plot_partial_dependence(shap_values_mean, sample_df, self.selected_features[0], 0)


    def explain_correlation(self, eval_sequences, num_samples=1, nsamples=30):
        random_samples = eval_sequences[np.random.choice(len(eval_sequences), num_samples, replace=False)]
        shap_values = self.explainer.shap_values(random_samples.reshape(num_samples, -1), nsamples=nsamples)
        return shap_values, random_samples

    def plot_dependence_correlation(self, shap_values, sample_df, target_feature, interaction_feature=None):
        """
        SHAP Dependence Plot: 특정 feature와 선택한 feature의 상호작용을 보여주는 플롯.
        target_feature: 분석할 feature
        interaction_feature: 상호작용할 feature (선택적)
        """
        print(f"Dependence plot for {target_feature}")
    
        # Feature 이름 확인 및 변환
        feature_names = sample_df.columns.tolist()  # 샘플 데이터프레임의 컬럼명을 feature 이름으로 사용
    
        if target_feature not in feature_names:
            raise ValueError(f"Could not find feature named: {target_feature}")
    
        if interaction_feature and interaction_feature not in feature_names:
            raise ValueError(f"Could not find interaction feature named: {interaction_feature}")
    
        # reshape shap_values to match the shape of sample_df
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

    def plot_shap_value_correlation_multiple_samples(self, shap_values, feature_names):
        """
        여러 샘플에 대한 SHAP 값들의 상관관계를 분석하는 함수.
        """
        # 샘플 전체의 SHAP 값을 평균으로 변환
        shap_values_reshaped = np.mean(np.abs(shap_values), axis=0).reshape(len(feature_names), -1)
    
        # SHAP 값 간 상관 행렬 계산
        shap_corr = np.corrcoef(shap_values_reshaped)
    
        # 상관 행렬 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(shap_corr, annot=True, cmap='coolwarm', xticklabels=feature_names, yticklabels=feature_names, vmin=-1, vmax=1)
        plt.title("SHAP Value Correlation Matrix (Feature Contribution)")
        plt.show()

    def visualize_correlation(self, shap_values, eval_sequences, target_feature_1, target_feature_2):
        feature_names = [f"{feature}_{i}" for i in range(self.sequence_length) for feature in self.selected_features]
    
        # eval_sequences의 크기가 올바르므로, 추가적인 reshape는 필요하지 않습니다.
        for i in range(len(eval_sequences)):
            print(f"Processing sequence {i} of eval_sequences...")
    
            # 데이터 확인 및 시각화
            try:
                sample_df = self.sequence_to_dataframe(eval_sequences[i], feature_names)
            except ValueError as e:
                print(f"Error with sequence {i}: {e}")
                continue  # 오류가 있는 시퀀스는 스킵
    
            self.plot_summary(shap_values[i], sample_df, feature_names)
            self.plot_dependence_correlation(shap_values[i], sample_df, 
                                             target_feature=target_feature_1, 
                                             interaction_feature=target_feature_2)  
            self.plot_shap_value_correlation(shap_values[i])  


    def plot_shap_value_correlation_multiple_samples(self, shap_values, feature_names):
        """
        여러 샘플에 대한 SHAP 값들의 상관관계를 분석하는 함수.
        """
        # 샘플 전체의 SHAP 값을 평균으로 변환
        shap_values_reshaped = np.mean(np.abs(shap_values), axis=0).reshape(len(feature_names), -1)
    
        # SHAP 값 간 상관 행렬 계산
        shap_corr = np.corrcoef(shap_values_reshaped)
    
        # 상관 행렬 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(shap_corr, annot=True, cmap='coolwarm', xticklabels=feature_names, yticklabels=feature_names, vmin=-1, vmax=1)
        plt.title("SHAP Value Correlation Matrix (Feature Contribution)")
        plt.show()

    def plot_summary2(self, shap_values, sample_df, feature_names):
        print("Summary plot")
        shap.summary_plot(shap_values, sample_df, feature_names=feature_names, show=True)

    def visualize_correlation_multiple_samples(self, shap_values, eval_sequences):
        feature_names = [f"{feature}_{i}" for i in range(self.sequence_length) for feature in self.selected_features]

        # shap_values의 크기를 확인하고 필요 시 reshape
        shap_values_mean = np.mean(np.abs(shap_values), axis=0)
        
        # 확인: shap_values_mean의 크기가 올바른지 출력
        print(f"shap_values_mean shape: {shap_values_mean.shape}")
        
        # 샘플 수에 맞게 reshape: eval_sequences.shape[1]은 feature의 수를 나타냄
        shap_values_reshaped = shap_values_mean.reshape(self.sequence_length, self.input_size)

        # feature_names에 맞게 sample_df 생성
        sample_df = pd.DataFrame(shap_values_reshaped, columns=self.selected_features)
        
        # 여러 샘플에 대한 summary plot 호출
        self.plot_summary2(shap_values_reshaped, sample_df, self.selected_features)

        # SHAP 값들의 상관관계 행렬 계산 및 시각화
        self.plot_shap_value_correlation_multiple_samples(shap_values, self.selected_features)

