import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # no grad

    def forward(self, x):
        """
        x: [T, B, d_model] 또는 [B, T, d_model]
        반환 shape은 입력과 동일 (앞쪽 T 차원에 pos 더함)
        """
        if x.dim() == 3 and x.shape[0] < x.shape[1]:  # [T,B,D]로 가정
            T, B, D = x.shape
            return x + self.pe[:T, :D].unsqueeze(1)
        elif x.dim() == 3:  # [B,T,D]
            B, T, D = x.shape
            return x + self.pe[:T, :D].unsqueeze(0)
        else:
            raise ValueError("positional encoding expects 3D tensor")
        
class PowerTransformerForecaster(nn.Module):
    def __init__(
        self,
        num_features: int,           # numeric feature F
        pred_len: int,               # 예측길이
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        # categorical(optional)
        icon_vocab_size: int | None = None,
        icon_emb_dim: int = 16,
        # misc
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model

        # numeric -> d_model
        self.num_proj = nn.Linear(num_features, d_model)

        # optional categorical embedding (e.g., icon sequence)
        self.use_icon = icon_vocab_size is not None and icon_vocab_size > 0
        if self.use_icon:
            self.icon_emb = nn.Embedding(icon_vocab_size, icon_emb_dim)
            self.icon_proj = nn.Linear(icon_emb_dim, d_model)

        self.pos_enc_src = SinusoidalPositionalEncoding(d_model)
        self.pos_enc_tgt = SinusoidalPositionalEncoding(d_model)

        # learnable decoder queries: [pred_len, d_model]
        self.query = nn.Parameter(torch.randn(pred_len, d_model))

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # we will use [T,B,D]
            norm_first=True
        )

        # output head: predict scalar per future step
        self.head = nn.Linear(d_model, 1)

        self.enc_norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.dec_norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

        # init
        nn.init.xavier_uniform_(self.num_proj.weight)
        if self.use_icon:
            nn.init.xavier_uniform_(self.icon_proj.weight)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor | None = None):
        """
        x_num: [B, T, F]
        x_cat: [B, T] (optional, categorical ids)
        return: [B, pred_len]
        """
        B, T, F = x_num.shape

        # numeric projection
        h = self.num_proj(x_num)  # [B,T,D]

        # add optional categorical embedding
        if self.use_icon and x_cat is not None:
            icon_e = self.icon_emb(x_cat)       # [B,T,icon_emb_dim]
            icon_d = self.icon_proj(icon_e)     # [B,T,D]
            h = h + icon_d

        # positional encoding & permute to [T,B,D] for nn.Transformer
        h = self.pos_enc_src(h)                 # [B,T,D]
        src = h.permute(1, 0, 2)                # [T,B,D]
        src = self.enc_norm(src)

        # target queries: same for batch, add positional enc
        tgt = self.query.unsqueeze(1).repeat(1, B, 1)  # [pred_len,B,D]
        tgt = self.pos_enc_tgt(tgt)
        tgt = self.dec_norm(tgt)

        # no masks: we only predict future given past (decoder sees only queries, attends to src)
        out = self.transformer(src=src, tgt=tgt)       # [pred_len,B,D]

        # project each step to scalar
        out = self.head(out)           # [pred_len,B,1]
        out = out.squeeze(-1).permute(1, 0)  # [B,pred_len]
        return out


class LongShortTransformer(nn.Module):
    """
    장·단기 동시 예측용 래퍼
    - EPC: forward(XL, XS) → outL, outS
    - UMass: forward(XnL, XnS, XiL, XiS) → outL, outS
    """
    def __init__(self,
                 long_num_features: int,
                 short_num_features: int,
                 long_pred_len: int,
                 short_pred_len: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 1,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 icon_vocab_size: int | None = None,
                 icon_emb_dim: int = 16):
        super().__init__()
        self.icon_on = icon_vocab_size is not None and icon_vocab_size > 0
        self.f_long = PowerTransformerForecaster(
            num_features=long_num_features, pred_len=long_pred_len,
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            icon_vocab_size=icon_vocab_size if self.icon_on else None,
            icon_emb_dim=icon_emb_dim
        )
        self.f_short = PowerTransformerForecaster(
            num_features=short_num_features, pred_len=short_pred_len,
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            icon_vocab_size=icon_vocab_size if self.icon_on else None,
            icon_emb_dim=icon_emb_dim
        )

    # EPC: (XL, XS),  UMass: (XnL, XnS, XiL, XiS)
    def forward(self, XL, XS, XiL: torch.Tensor | None = None, XiS: torch.Tensor | None = None):
        if self.icon_on:
            yL = self.f_long(XL, XiL)
            yS = self.f_short(XS, XiS)
        else:
            yL = self.f_long(XL, None)
            yS = self.f_short(XS, None)
        return yL, yS