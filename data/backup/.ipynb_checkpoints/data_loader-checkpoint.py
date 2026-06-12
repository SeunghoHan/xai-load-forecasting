import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split


class EPCDataset:
    def __init__(self, file_path, sequence_length, prediction_length, target_features):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        # Add datetime features to target features
        self.datetime_features = ['sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']
        self.selected_features = target_features + self.datetime_features

    def load_data(self):
        data = pd.read_csv(self.file_path)
        data = self._add_datetime_features(data)
        data.drop(columns=['datetime'], errors='ignore', inplace=True)
        data.fillna(data.mean(), inplace=True)

        # Select features
        data_selected = data[self.selected_features]

        # Normalize the data
        data_scaled = self.scaler.fit_transform(data_selected.values)

        # Create sequences
        sequences, targets = self._create_sequences(data_scaled)

        # Split the data into train and eval sets
        train_sequences, eval_sequences, train_targets, eval_targets = train_test_split(sequences, targets, test_size=0.2, random_state=42)

        return (
            torch.tensor(train_sequences, dtype=torch.float32),
            torch.tensor(train_targets, dtype=torch.float32).squeeze(-1),
            torch.tensor(eval_sequences, dtype=torch.float32),
            torch.tensor(eval_targets, dtype=torch.float32).squeeze(-1)
        )
    

    def _add_datetime_features(self, data):
        # Convert datetime column and add cyclic features
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['hour'] = data['datetime'].dt.hour
        data['day'] = data['datetime'].dt.day
        data['month'] = data['datetime'].dt.month

        data['sin_hour'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['sin_day'] = np.sin(2 * np.pi * data['day'] / 31)
        data['cos_day'] = np.cos(2 * np.pi * data['day'] / 31)
        data['sin_month'] = np.sin(2 * np.pi * data['month'] / 12)
        data['cos_month'] = np.cos(2 * np.pi * data['month'] / 12)

        return data

    def _create_sequences(self, data):
        # Generate sequences and targets
        sequences = []
        targets = []

        for i in range(len(data) - self.sequence_length - self.prediction_length):
            sequences.append(data[i:i + self.sequence_length])
            targets.append(data[i + self.sequence_length:i + self.sequence_length + self.prediction_length, 0])

        return np.array(sequences), np.array(targets)

