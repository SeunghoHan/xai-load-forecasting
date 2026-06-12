import torch
import torch.nn as nn

from .base import BaseModel

class LSTMModel(BaseModel):
    """
    x_num: (B, T, D_num)
    x_icon: (B, T)  [optional, int64]  -> Embedding -> (B, T, E)
    output: (B, prediction_length)
    """
    def __init__(
        self,
        input_size_num,          # 수치형 특징 개수 (UMassDataset.num_feature_names 길이)
        output_size,             # prediction_length
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        icon_vocab_size=None,    # len(ds.icon_vocab) 또는 None
        icon_embed_dim=8,        # 아이콘 임베딩 차원
        bidirectional=False
    ):
        super().__init__()
        self.use_icon = (icon_vocab_size is not None) and (icon_vocab_size > 0)
        self.icon_embed_dim = icon_embed_dim if self.use_icon else 0

        if self.use_icon:
            self.icon_emb = nn.Embedding(icon_vocab_size, icon_embed_dim)

        lstm_input_size = input_size_num + self.icon_embed_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        fc_in = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_in, output_size)

    def forward(self, x_num, x_icon=None):
        """
        x_num: (B, T, D_num)  float
        x_icon: (B, T) long or None
        """
        if self.use_icon:
            if x_icon is None:
                raise ValueError("icon_vocab_size was set, but x_icon is None.")
            emb = self.icon_emb(x_icon)          # (B, T, E)
            x = torch.cat([x_num, emb], dim=-1)  # (B, T, D_num+E)
        else:
            x = x_num

        B = x.size(0)
        # h0, c0 자동 0 초기화하면 됨(None 전달), 명시 생성도 가능
        out, _ = self.lstm(x)                    # (B, T, H*(2 if bi else 1))
        out_last = out[:, -1, :]                 # (B, H*)
        out_last = self.dropout(out_last)
        out = self.fc(out_last)                  # (B, prediction_length)
        return out

    def predict(self, x_num, x_icon=None):
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            x_num = torch.as_tensor(x_num, dtype=torch.float32, device=device)
            if self.use_icon:
                x_icon = torch.as_tensor(x_icon, dtype=torch.long, device=device)
            return self.forward(x_num, x_icon).cpu().numpy()