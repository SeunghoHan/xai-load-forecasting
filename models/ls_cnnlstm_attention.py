import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
        return attn_output


class LongShortCNNLSTMWithAttention(nn.Module):
    def __init__(self, 
                 long_input_size, 
                 short_input_size, 
                 hidden_size,  
                 num_layers, 
                 long_output_size, 
                 short_output_size, 
                 long_term_length, 
                 short_term_length, 
                 num_heads=2,
                 dropout=0.3):
        super(LongShortCNNLSTMWithAttention, self).__init__()

        self.dropout_rate = dropout
        self.hidden_size = hidden_size

        # Long-term CNN (1D)
        self.long_cnn = nn.Sequential(
            nn.Conv1d(long_input_size, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.long_seq_len = self._calculate_seq_length_after_cnn(long_input_size, long_term_length, cnn_type="long")

        self.long_lstm = nn.LSTM(
            input_size=self.long_seq_len['channels'],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout_rate,
            bidirectional=True
        )

        # Short-term CNN (1D)
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

        self.short_seq_len = self._calculate_seq_length_after_cnn(short_input_size, short_term_length, cnn_type="short")

        self.short_lstm = nn.LSTM(
            input_size=self.short_seq_len['channels'],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout_rate,
            bidirectional=True
        )

        # Cross Attention Layer
        self.cross_attention = CrossAttention(embed_dim=hidden_size * 2, num_heads=num_heads)

        self.long_fc = nn.Linear(hidden_size * 2 * self.long_seq_len['length'], long_output_size)
        self.short_fc = nn.Linear(hidden_size * 2, short_output_size)

    def forward(self, long_input, short_input):
        # Long-term processing
        x_long = self.long_cnn(long_input.permute(0, 2, 1))  # (B, C, T)
        x_long = x_long.permute(0, 2, 1)  # (B, T, C)
        long_output, _ = self.long_lstm(x_long)  # (B, T, D)

        # Short-term processing
        x_short = self.short_cnn(short_input.permute(0, 2, 1))  # (B, C, T)
        x_short = x_short.permute(0, 2, 1)  # (B, T, C)
        short_output, _ = self.short_lstm(x_short)  # (B, T, D)

        # Cross Attention: short attends to long
        attended_short = self.cross_attention(short_output, long_output, long_output)

        # Long-term prediction: flatten entire sequence
        long_flat = long_output.contiguous().view(long_output.size(0), -1)
        long_final = self.long_fc(long_flat)

        # Short-term prediction: use last timestep of attended output
        short_final = self.short_fc(attended_short[:, -1, :])

        return long_final, short_final, attended_short

    def _calculate_seq_length_after_cnn(self, input_channels, seq_len, cnn_type="long"):
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, seq_len)
            if cnn_type == "long":
                output = self.long_cnn(dummy_input)
            else:
                output = self.short_cnn(dummy_input)
            return {"length": output.size(2), "channels": output.size(1)}

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CrossAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super(CrossAttention, self).__init__()
#         self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

#     def forward(self, query, key, value):
#         # query: (B, T_q, D), key/value: (B, T_k, D)
#         attn_output, _ = self.attn(query, key, value)
#         return attn_output


# class LongShortCNNLSTMWithAttention(nn.Module):
#     def __init__(self, 
#                  long_input_size, 
#                  short_input_size, 
#                  hidden_size,  
#                  num_layers, 
#                  long_output_size, 
#                  short_output_size, 
#                  long_term_length, 
#                  short_term_length, 
#                  num_heads=2,
#                  dropout=0.3):
#         super(LongShortCNNLSTMWithAttention, self).__init__()

#         self.dropout_rate = dropout

#         # Long-term CNN (1D)
#         self.long_cnn = nn.Sequential(
#             nn.Conv1d(long_input_size, 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.Dropout(self.dropout_rate),
#             nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2)
#         )

#         self.long_cnn_output_size = self._calculate_cnn_output_size(
#             self.long_cnn, (1, long_input_size, long_term_length)
#         )

#         self.long_lstm = nn.LSTM(
#             input_size=self.long_cnn_output_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=self.dropout_rate,
#             bidirectional=True
#         )

#         # Short-term CNN (1D)
#         self.short_cnn = nn.Sequential(
#             nn.Conv1d(short_input_size, 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.Dropout(self.dropout_rate),
#             nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2)
#         )

#         self.short_cnn_output_size = self._calculate_cnn_output_size(
#             self.short_cnn, (1, short_input_size, short_term_length)
#         )

#         self.short_lstm = nn.LSTM(
#             input_size=self.short_cnn_output_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=self.dropout_rate,
#             bidirectional=True
#         )

#         # Cross Attention Layer
#         self.cross_attention = CrossAttention(embed_dim=hidden_size * 2, num_heads=num_heads)

#         self.long_fc = nn.Linear(hidden_size * 2 * long_term_length, long_output_size)
#         self.short_fc = nn.Linear(hidden_size * 2, short_output_size)

#     def forward(self, long_input, short_input):
#         # Long-term processing
#         x_long = self.long_cnn(long_input.permute(0, 2, 1))  # (B, C, T) <- input (B, T, C)
#         x_long = x_long.permute(0, 2, 1)  # (B, T, C)
#         long_output, _ = self.long_lstm(x_long)  # (B, T, D)

#         # Short-term processing
#         x_short = self.short_cnn(short_input.permute(0, 2, 1))  # (B, C, T)
#         x_short = x_short.permute(0, 2, 1)  # (B, T, C)
#         short_output, _ = self.short_lstm(x_short)  # (B, T, D)

#         # Cross Attention: short attends to long
#         attended_short = self.cross_attention(short_output, long_output, long_output)

#         # Long-term prediction: flatten entire sequence
#         long_flat = long_output.contiguous().view(long_output.size(0), -1)
#         long_final = self.long_fc(long_flat)

#         # Short-term prediction: use last timestep of attended output
#         short_final = self.short_fc(attended_short[:, -1, :])

#         return long_final, short_final

#     def _calculate_cnn_output_size(self, module, input_shape):
#         with torch.no_grad():
#             dummy_input = torch.zeros(*input_shape)
#             output = module(dummy_input)
#             return output.size(1)
