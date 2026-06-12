import os
import re
import json
import numpy as np
import torch
import lime
from lime import lime_tabular
import matplotlib.pyplot as plt

from .base_explainer import BaseExplainer

class LimeExplainer(BaseExplainer):
    def __init__(self, model, device, sequences, sequence_length, input_size, selected_features):
        # explainer_path = f'./trained_explainer/lime_{len(selected_features)}.pkl'
        super().__init__(model, device, sequences, sequence_length, input_size, selected_features)
        self.lime_tab = self.create_explainer()
        self.scaler = scaler  
    

    def create_explainer(self):
        return lime_tabular.LimeTabularExplainer(
            training_data=self.sequences.reshape(-1, self.sequence_length * self.input_size),
            feature_names=[f"{feature}_{i}" for i in range(self.sequence_length) for feature in self.selected_features],
            class_names=['Global_active_power'],
            mode='regression'
        )

    def predict_fn(self, input_data):
        input_data = input_data.reshape(-1, self.sequence_length, self.input_size)
        self.model.eval()
        input_data = torch.tensor(input_data, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_data)
        return outputs.cpu().numpy()

    def explain(self, data_point, data_index=0, num_features=10):
        specific_data_point = data_point.reshape(-1)
        specific_explanation = self.lime_tab.explain_instance(specific_data_point, self.predict_fn, num_samples=1000, num_features=num_features)
        specific_explanation.show_in_notebook(show_table=True)
        important_features = specific_explanation.as_list()
        self.visualize_lime_explanation(important_features, self.sequences, self.sequence_length, self.selected_features, data_index)
        return specific_explanation.as_list()

    def visualize_lime_explanation(self, important_features, sequences, sequence_length, selected_features, data_index=0):
        ## Dictionary to group features by their base name
        feature_dict = {}
        comp_op = r'(<=|>=|<|>)'
        alph_op = r'[a-zA-Z_]+\d*(?:_\d+)*'
        
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
            
            feature_dict[actual_feature].append((time_step, sequences[0][time_step, selected_features.index(actual_feature)], importance))
    
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
            plt.title(f'Feature: {actual_feature}')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.show()
        
            # Print the detailed information about the important points
            print(f'\nFeature: {actual_feature} - Detailed Information')
            for time_step, value, importance in values:
                print(f'Time Step: {time_step}, Feature Value: {value:.3f}, Importance: {importance:.3f}')