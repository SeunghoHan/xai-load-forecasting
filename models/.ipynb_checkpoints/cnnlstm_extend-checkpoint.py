import torch
import torch.nn as nn
from .base import BaseModel

class CNNLSTMModel(BaseModel):
    def __init__(
        self,
        input_size_num,
        output_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        icon_vocab_size=None,
        icon_embed_dim=8
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # icon embedding 여부
        self.use_icon = bool(icon_vocab_size) and icon_vocab_size > 0
        if self.use_icon:
            self.icon_emb = nn.Embedding(icon_vocab_size, icon_embed_dim)
            conv_in_channels = input_size_num + icon_embed_dim
        else:
            self.icon_emb = None
            conv_in_channels = input_size_num

        # CNN + LSTM 구조
        self.conv1 = nn.Conv1d(conv_in_channels, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_num, x_icon=None):
        """
        x_num : (B, T, D_num)
        x_icon: (B, T) long or None
        """
        if self.use_icon:
            if x_icon is None:
                raise ValueError("icon_vocab_size가 설정되었는데 x_icon이 None 입니다.")
            emb = self.icon_emb(x_icon)           # (B, T, E)
            x = torch.cat([x_num, emb], dim=-1)   # (B, T, D_num+E)
        else:
            x = x_num                             # (B, T, D_num)

        # Conv1D 입력 형식으로 변환: (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv1(x)))  # (B, 64, T/2)

        # 다시 LSTM 입력 형식: (B, T/2, 64)
        x = x.permute(0, 2, 1)

        # 초기 hidden state/cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))           # (B, T/2, H)
        out = self.dropout(out[:, -1, :])         # (B, H)
        out = self.fc(out)                        # (B, output_size)
        return out

    def predict(self, x_num, x_icon=None):
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            x_num = torch.as_tensor(x_num, dtype=torch.float32, device=device)
            if self.use_icon:
                x_icon = torch.as_tensor(x_icon, dtype=torch.long, device=device)
            return self.forward(x_num, x_icon).cpu().numpy()