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


class MultiTermDataset:
    def __init__(self, file_path, long_term_length, short_term_length, prediction_length, target_features=None):
        self.file_path = file_path
        self.long_term_length = long_term_length
        self.short_term_length = short_term_length
        self.prediction_length = prediction_length
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()  # Target 전용 스케일러 추가

        self.datetime_features = ['sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']
        self.selected_features = target_features if target_features else [
            'Global_active_power', 'Global_intensity', 
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
            'Temp_Min', 'Temp_Max', 'Humidity_Min', 'Humidity_Max'
        ] + self.datetime_features

    def load_data(self):
        data = pd.read_csv(self.file_path)
        data = self._add_datetime_features(data)
        data.drop(columns=['datetime'], errors='ignore', inplace=True)
        data.fillna(data.mean(), inplace=True)
        data_selected = data[self.selected_features]
        data_scaled = self.scaler.fit_transform(data_selected.values)

        long_sequences, short_sequences, targets = self.create_multi_term_sequences(data_scaled)

        # 타겟 스케일링
        targets = self.target_scaler.fit_transform(targets)

        train_long, eval_long, train_short, eval_short, train_targets, eval_targets = train_test_split(
            long_sequences, short_sequences, targets, test_size=0.2, random_state=42, shuffle=False
        )

        return (
            torch.tensor(train_long, dtype=torch.float32),
            torch.tensor(train_short, dtype=torch.float32),
            torch.tensor(train_targets, dtype=torch.float32).squeeze(-1),
            torch.tensor(eval_long, dtype=torch.float32),
            torch.tensor(eval_short, dtype=torch.float32),
            torch.tensor(eval_targets, dtype=torch.float32).squeeze(-1)
        )

    def _add_datetime_features(self, data):
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

    def create_multi_term_sequences(self, data):
        long_sequences = []
        short_sequences = []
        targets = []

        for i in range(len(data) - self.long_term_length - self.short_term_length - self.prediction_length):
            long_sequences.append(data[i:i + self.long_term_length])
            short_sequences.append(data[i + self.long_term_length:i + self.long_term_length + self.short_term_length])
            targets.append(data[i + self.long_term_length + self.short_term_length:
                                i + self.long_term_length + self.short_term_length + self.prediction_length, 0].flatten())

        return np.array(long_sequences), np.array(short_sequences), np.array(targets)

    
class SingleTermDataset:
    def __init__(self, file_path, sequences_length, prediction_length, target_features):
        self.file_path = file_path
        self.sequences_length = sequences_length
        self.prediction_length = prediction_length
        self.selected_features = target_features
        self.scaler = MinMaxScaler()
    
    def load_data(self, single_term=False):
        """
        Load data and return long/short-term sequences for multi-term models
        or single sequences for single-term models.
        """
        # 데이터 로드 및 전처리
        data = pd.read_csv(self.file_path)
        data.drop(columns=['datetime'], errors='ignore', inplace=True)

        # 결측값 처리 (PowerWeatherDataset 방식)
        data.fillna(data.mean(), inplace=True)

        # 피처 선택 및 스케일링
        data_selected = data[self.selected_features]
        data_scaled = self.scaler.fit_transform(data_selected.values)

        # 시퀀스 생성
        sequences, targets = self.create_single_term_sequences(data_scaled)
        train_sequences, eval_sequences, train_targets, eval_targets = train_test_split(
            sequences, targets, test_size=0.2, random_state=42, shuffle=False
        )
        return (
            torch.tensor(train_sequences, dtype=torch.float32),
            torch.tensor(train_targets, dtype=torch.float32),
            torch.tensor(eval_sequences, dtype=torch.float32),
            torch.tensor(eval_targets, dtype=torch.float32)
        )

    def create_single_term_sequences(self, data):
        sequences = []
        targets = []
        for i in range(len(data) - self.sequences_length - self.prediction_length):
            # 단일 시퀀스 생성
            sequences.append(data[i:i + self.sequences_length])
            targets.append(data[i + self.sequences_length:i + self.sequences_length + self.prediction_length, 0])
        return np.array(sequences), np.array(targets)


class PowerWeatherDataset:
    def __init__(self, file_path, sequence_length=24*30, prediction_length=24, target_features=[]):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.scaler = MinMaxScaler()
        
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
        sequences, targets = self.create_sequences(data_scaled)

        # Split the data into train and eval sets
        train_sequences, eval_sequences, train_targets, eval_targets = train_test_split(sequences, targets, test_size=0.2, random_state=42)

        return (
            torch.tensor(train_sequences, dtype=torch.float32),
            torch.tensor(train_targets, dtype=torch.float32).squeeze(-1),
            torch.tensor(eval_sequences, dtype=torch.float32),
            torch.tensor(eval_targets, dtype=torch.float32).squeeze(-1)
        )

    def create_sequences(self, data):
        sequences = []
        targets = []
        for i in range(len(data) - self.sequence_length - self.prediction_length):
            sequences.append(data[i:i + self.sequence_length])
            # Next 'prediction_length' values for 'Global_active_power' (first column)
            targets.append(data[i + self.sequence_length : i + self.sequence_length + self.prediction_length, 0])
        return np.array(sequences), np.array(targets)

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


class PowerWeatherDatasetWithSeason:
    def __init__(self, file_path, sequence_length=24*30, prediction_length=24, target_features=[]):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.scaler = MinMaxScaler()

        # 기본 feature 설정
        if len(target_features) == 0:
            self.selected_features = ['Global_active_power', 'Global_intensity', 
                                      'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 
                                      'Temp_Avg', 'Humidity_Avg', 'sin_hour', 'cos_hour', 
                                      'sin_day', 'cos_day', 'sin_month', 'cos_month']
        else:
            self.selected_features = target_features

    def load_data_by_season(self):
        # Load data
        data = pd.read_csv(self.file_path)

        # Drop non-numeric columns (like 'datetime')
        if 'datetime' in data.columns:
            data.drop(columns=['datetime'], inplace=True)

        # Handle missing values for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

        # Ensure 'season' column exists
        if 'season' not in data.columns:
            raise ValueError("The dataset must contain a 'season' column for this function.")

        # Normalize data
        data_scaled = self.scaler.fit_transform(data[self.selected_features])

        # Split by season
        season_data = {}
        for season in ['spring', 'summer', 'autumn', 'winter']:
            season_df = data[data['season'] == season]
            season_scaled = self.scaler.transform(season_df[self.selected_features])

            # Create sequences for this season
            sequences, targets = self.create_sequences(season_scaled)
            season_data[season] = train_test_split(sequences, targets, test_size=0.2, random_state=42)

        return season_data

    def create_sequences(self, data):
        sequences = []
        targets = []
        for i in range(len(data) - self.sequence_length - self.prediction_length):
            # Input sequence
            sequences.append(data[i:i + self.sequence_length])
            # Prediction target (next 'prediction_length' values for the first feature)
            targets.append(data[i + self.sequence_length:i + self.sequence_length + self.prediction_length, 0])
        return np.array(sequences), np.array(targets)
        

class PowerConsumptionDataset:
    def __init__(self, file_path, feature_idx=[], num_of_features=7, sequence_length=60):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()

        all_features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        
        if len(feature_idx)>0:
            self.selected_features = [all_features[i] for i in feature_idx if i < len(all_features)]
        else:
            self.selected_features = all_features
    
        # self.selected_features = all_features[0:num_of_features]
        
    def load_data(self):
        # Load and preprocess data
        data = pd.read_csv(self.file_path, sep=';', parse_dates={'datetime': ['Date', 'Time']}, 
                           infer_datetime_format=True, low_memory=False, na_values=['?'])

        # Fill missing values
        data.fillna(data.mean(), inplace=True)

        # Select all features
        data_selected = data[self.selected_features]

        # Normalize the data
        data_scaled = self.scaler.fit_transform(data_selected.values)

        # Create sequences
        sequences, targets = self.create_sequences(data_scaled)

        # Split the data into train and eval sets
        train_sequences, eval_sequences, train_targets, eval_targets = train_test_split(sequences, targets, test_size=0.2, random_state=42)

        return train_sequences, eval_sequences, train_targets, eval_targets

    def create_sequences(self, data):
        sequences = []
        targets = []
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:i + self.sequence_length])
            targets.append(data[i + self.sequence_length, 0])  # Predicting 'Global_active_power'
        return np.array(sequences), np.array(targets)