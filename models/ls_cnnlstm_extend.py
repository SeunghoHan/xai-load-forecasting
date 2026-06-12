import torch
import torch.nn as nn

class LongShortCNNLSTM(nn.Module):
    def __init__(self, 
                 long_input_size_num, 
                 short_input_size_num, 
                 hidden_size,  
                 num_layers, 
                 long_output_size, 
                 short_output_size, 
                 long_term_length, 
                 short_term_length, 
                 dropout=0.3,
                 icon_vocab_size=None,
                 icon_embed_dim=8):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # icon embedding (공용)
        self.use_icon = bool(icon_vocab_size) and icon_vocab_size > 0
        if self.use_icon:
            self.icon_emb = nn.Embedding(icon_vocab_size, icon_embed_dim)
            long_input_size = long_input_size_num + icon_embed_dim
            short_input_size = short_input_size_num + icon_embed_dim
        else:
            self.icon_emb = None
            long_input_size = long_input_size_num
            short_input_size = short_input_size_num

        # --- Long-term CNN ---
        self.long_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, long_input_size), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        )
        self.long_cnn_output_size = self._calculate_cnn_output_size(
            self.long_cnn, (1, 1, long_term_length, long_input_size)
        )
        self.long_lstm = nn.LSTM(
            input_size=self.long_cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout_rate,
            bidirectional=True
        )

        # --- Short-term CNN ---
        self.short_cnn = nn.Sequential(
            nn.Conv1d(short_input_size, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.short_cnn_output_size = self._calculate_cnn_output_size(
            self.short_cnn, (1, short_input_size, short_term_length)
        )
        self.short_lstm = nn.LSTM(
            input_size=self.short_cnn_output_size + hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout_rate,
            bidirectional=True
        )

        # Output heads
        self.long_fc = nn.Linear(hidden_size * 2, long_output_size)
        self.short_fc = nn.Linear(hidden_size * 2, short_output_size)

    def forward(self, long_num, short_num, long_icon=None, short_icon=None):
        # --- icon concat ---
        if self.use_icon:
            if long_icon is None or short_icon is None:
                raise ValueError("icon_vocab_size > 0 인데 long_icon/short_icon 입력이 없습니다.")
            long_emb = self.icon_emb(long_icon)     # (B, T, E)
            short_emb = self.icon_emb(short_icon)   # (B, T, E)
            long_in = torch.cat([long_num, long_emb], dim=-1)
            short_in = torch.cat([short_num, short_emb], dim=-1)
        else:
            long_in, short_in = long_num, short_num

        # --- Long-term path ---
        long_features = self.long_cnn(long_in.unsqueeze(1))        # (B, C, T', F')
        long_features = long_features.view(long_features.size(0), long_features.size(2), -1)
        long_output, _ = self.long_lstm(long_features)
        long_output_last = long_output[:, -1, :]                   # (B, 2H)
        long_final = self.long_fc(long_output_last)

        # --- Short-term path ---
        short_features = self.short_cnn(short_in.permute(0, 2, 1)) # (B, C, T')
        short_features = short_features.view(short_features.size(0), short_features.size(2), -1)
        combined = torch.cat(
            (short_features, long_output_last.unsqueeze(1).repeat(1, short_features.size(1), 1)), dim=2
        )
        short_output, _ = self.short_lstm(combined)
        short_final = self.short_fc(short_output[:, -1, :])

        return long_final, short_final

    def _calculate_cnn_output_size(self, module, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(*input_shape)
            output = module(dummy_input)
            if len(output.shape) == 4:   # Conv2d
                return output.size(1) * output.size(3)
            elif len(output.shape) == 3: # Conv1d
                return output.size(1)