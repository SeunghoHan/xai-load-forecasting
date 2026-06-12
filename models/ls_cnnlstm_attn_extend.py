import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
        return attn_output


class LongShortCNNLSTMWithAttention(nn.Module):
    def __init__(self, 
                 long_input_size_num, 
                 short_input_size_num, 
                 hidden_size,  
                 num_layers, 
                 long_output_size, 
                 short_output_size, 
                 long_term_length, 
                 short_term_length, 
                 num_heads=2,
                 dropout=0.3,
                 icon_vocab_size=None,
                 icon_embed_dim=8):
        super().__init__()
        self.dropout_rate = dropout
        self.hidden_size = hidden_size

        # icon embedding
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
            nn.Conv1d(long_input_size, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.long_seq_len = self._calculate_seq_length_after_cnn(long_input_size, long_term_length, "long")
        self.long_lstm = nn.LSTM(
            input_size=self.long_seq_len['channels'],
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
        self.short_seq_len = self._calculate_seq_length_after_cnn(short_input_size, short_term_length, "short")
        self.short_lstm = nn.LSTM(
            input_size=self.short_seq_len['channels'],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout_rate,
            bidirectional=True
        )

        # Cross Attention
        self.cross_attention = CrossAttention(embed_dim=hidden_size * 2, num_heads=num_heads)

        # Output layers
        self.long_fc = nn.Linear(hidden_size * 2 * self.long_seq_len['length'], long_output_size)
        self.short_fc = nn.Linear(hidden_size * 2, short_output_size)

    def forward(self, long_num, short_num, long_icon=None, short_icon=None):
        # icon concat
        if self.use_icon:
            if long_icon is None or short_icon is None:
                raise ValueError("icon_vocab_size>0 인데 long_icon/short_icon이 None 입니다.")
            long_emb = self.icon_emb(long_icon)     # (B,T,E)
            short_emb = self.icon_emb(short_icon)
            long_in = torch.cat([long_num, long_emb], dim=-1)
            short_in = torch.cat([short_num, short_emb], dim=-1)
        else:
            long_in, short_in = long_num, short_num

        # Long-term
        x_long = self.long_cnn(long_in.permute(0, 2, 1))   # (B,C,T)
        x_long = x_long.permute(0, 2, 1)                   # (B,T,C)
        long_output, _ = self.long_lstm(x_long)            # (B,T,D)

        # Short-term
        x_short = self.short_cnn(short_in.permute(0, 2, 1))
        x_short = x_short.permute(0, 2, 1)
        short_output, _ = self.short_lstm(x_short)         # (B,T,D)

        # Cross Attention
        attended_short = self.cross_attention(short_output, long_output, long_output)

        # Long prediction
        long_flat = long_output.contiguous().view(long_output.size(0), -1)
        long_final = self.long_fc(long_flat)

        # Short prediction
        short_final = self.short_fc(attended_short[:, -1, :])

        return long_final, short_final, attended_short

    def _calculate_seq_length_after_cnn(self, input_channels, seq_len, cnn_type="long"):
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, seq_len)
            if cnn_type == "long":
                out = self.long_cnn(dummy)
            else:
                out = self.short_cnn(dummy)
            return {"length": out.size(2), "channels": out.size(1)}