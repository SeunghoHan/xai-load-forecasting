import torch
import torch.nn as nn

from .base import BaseModel

class GRUModel(BaseModel):
    def __init__(
        self,
        input_size_num,
        output_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        icon_vocab_size=None,
        icon_embed_dim=8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.use_icon = bool(icon_vocab_size) and icon_vocab_size > 0
        if self.use_icon:
            self.icon_emb = nn.Embedding(icon_vocab_size, icon_embed_dim)
            gru_input_size = input_size_num + icon_embed_dim
        else:
            self.icon_emb = None
            gru_input_size = input_size_num

        # 원본과 동일: GRU 모듈에 dropout 미적용, FC 앞에서만 dropout 적용
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_num, x_icon=None):
        """
        x_num : (B, T, D_num) float
        x_icon: (B, T) long or None
        """
        if self.use_icon:
            if x_icon is None:
                raise ValueError("icon_vocab_size가 설정되었는데 x_icon이 None 입니다.")
            emb = self.icon_emb(x_icon)          # (B, T, E)
            x = torch.cat([x_num, emb], dim=-1)  # (B, T, D_num+E)
        else:
            x = x_num

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)                 # (B, T, H)
        out_last = out[:, -1, :]                 # (B, H)
        out_last = self.dropout(out_last)
        out = self.fc(out_last)                  # (B, prediction_length)
        return out

    def predict(self, x_num, x_icon=None):
        """
        예측 (EPC: x_icon 생략 / UMass: x_icon 제공)
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            x_num = torch.as_tensor(x_num, dtype=torch.float32, device=device)
            if self.use_icon:
                x_icon = torch.as_tensor(x_icon, dtype=torch.long, device=device)
            return self.forward(x_num, x_icon).cpu().numpy()