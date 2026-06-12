import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionLayer, self).__init__()
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
 
    def forward(self, query, key, value):
        Q = self.query_layer(query)
        K = self.key_layer(key)
        V = self.value_layer(value)

        attention_weights = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        return attention_output, attention_weights

class LongShortCNNLSTMWithAttention(nn.Module):
    def __init__(self, input_dim_long, input_dim_short, hidden_dim, long_output_dim, output_dim, seq_len_long, seq_len_short):
        super(LongShortCNNLSTMWithAttention, self).__init__()

        # Long-term CNN-LSTM
        self.conv_long = nn.Conv1d(input_dim_long, hidden_dim, kernel_size=3, padding=1)
        self.lstm_long = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Short-term CNN-LSTM
        self.conv_short = nn.Conv1d(input_dim_short, hidden_dim, kernel_size=3, padding=1)
        self.lstm_short = nn.LSTM(hidden_dim + hidden_dim, hidden_dim, batch_first=True)  # Long-term + Short-term

        # Attention Layer
        self.attention = AttentionLayer(embed_dim=hidden_dim)

        # Fully Connected Layers
        self.fc_long = nn.Linear(seq_len_long * hidden_dim, long_output_dim)  # Long-term output compression
        self.fc_short = nn.Linear(hidden_dim, output_dim)  # Final prediction

    def forward(self, x_long, x_short):
        # Long-term branch
        x_long = self.conv_long(x_long.transpose(1, 2)).transpose(1, 2)
        x_long, (h_long, c_long) = self.lstm_long(x_long)

        # Compress long-term output
        x_long_flat = x_long.reshape(x_long.size(0), -1)
        long_representation = self.fc_long(x_long_flat)

        # Short-term branch
        x_short = self.conv_short(x_short.transpose(1, 2)).transpose(1, 2)
        attention_output, attention_weights = self.attention(x_short, x_long, x_long)  # Cross-Attention
        x_short_combined = torch.cat([x_short, attention_output], dim=-1)

        
        x_short, (h_short, c_short) = self.lstm_short(x_short_combined)

        # Final prediction
        output_short = self.fc_short(x_short[:, -1, :])

        return long_representation, output_short, attention_weights
