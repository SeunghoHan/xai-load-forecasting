# file: models/longshort_dual_tst_hier.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    def forward(self, x):  # [B,N,D]
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class PatchEmbedding1D(nn.Module):
    def __init__(self, d_model: int, patch_len: int = 24, stride: int | None = None):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride if stride is not None else patch_len
        self.conv = nn.Conv1d(1, d_model, kernel_size=self.patch_len, stride=self.stride)
        self.act = nn.GELU()
    def forward(self, x):  # x: [B,T,1]
        x = x.transpose(1, 2)      # [B,1,T]
        x = self.act(self.conv(x)) # [B,D,N]
        return x.transpose(1, 2)   # [B,N,D]

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
    def forward(self, x):
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.drop1(h)
        x = x + self.ff(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, depth: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(d_model, nhead, d_ff, dropout) for _ in range(depth)])
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class SinglePatchTST(nn.Module):
    """
    단기/단일 태스크 예측 (예: 30일 입력 → 24시간 출력)
    - forward(x_num, x_cat=None) 유지 → 기존 학습/평가 루틴과 호환
    - num_features: 입력 피처 수, pred_len: 예측 수평선
    - use_ci=True: 타깃 채널만 사용(안정적), False: 멀티변수 선형 혼합
    """
    def __init__(
        self,
        num_features: int,
        pred_len: int,
        target_idx: int = 0,
        d_model: int = 128,
        nhead: int = 8,
        d_ff: int = 512,
        depth: int = 3,
        patch_len: int = 24,
        stride: int | None = None,
        dropout: float = 0.1,
        use_ci: bool = True,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.target_idx = target_idx
        self.use_ci = use_ci

        self.input_norm = nn.LayerNorm(num_features)
        self.mix = None if use_ci else nn.Linear(num_features, 1)

        self.patch = PatchEmbedding1D(d_model=d_model, patch_len=patch_len, stride=stride)
        self.pos   = SinusoidalPositionalEncoding(d_model)
        self.encoder = TransformerEncoder(d_model, nhead, d_ff, depth, dropout)

        self.query_vec = nn.Parameter(torch.randn(1, 1, d_model))
        self.head = nn.Linear(d_model, pred_len)
        self.drop = nn.Dropout(dropout)

    def _to_target(self, x):  # [B,T,F] → [B,T,1]
        if self.use_ci:
            return x[:, :, self.target_idx:self.target_idx+1]
        return self.mix(x)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor | None = None):
        x = self.input_norm(x_num)      # [B,T,F]
        x = self._to_target(x)          # [B,T,1]
        tok = self.patch(x)             # [B,N,D]
        tok = self.pos(tok)
        enc = self.encoder(tok)         # [B,N,D]

        q = self.query_vec.expand(enc.size(0), -1, -1)           # [B,1,D]
        attn = torch.softmax(torch.matmul(q, enc.transpose(1, 2)) / math.sqrt(enc.size(-1)), dim=-1)
        pooled = torch.matmul(attn, enc).squeeze(1)              # [B,D]
        pooled = self.drop(pooled)
        y = self.head(pooled)                                     # [B,pred_len]
        return y

class LongShortDualTST_Hier(nn.Module):
    """
    Long+Short dual-stream encoder → single/dual decoder.
    입력:
      XL: [B, TL, F], XS: [B, TS, F]   (x_cat*는 무시, 시그니처 호환)
    출력(학습 루프 호환):
      outL: [B, long_term_pred_length]   # 보조 헤드
      outS: [B, short_term_pred_length]  # 최종 7일 예측
    """
    def __init__(
        self,
        long_term_length: int,
        short_term_length: int,
        short_term_pred_length: int,     # 최종 예측 horizon (예: 168)
        long_term_pred_length: int,      # 보조 헤드 horizon (예: 256)  ← ★ 추가
        num_features_long: int,
        num_features_short: int,
        target_idx: int = 0,
        d_model: int = 128,
        nhead: int = 8,
        d_ff: int = 512,
        depth: int = 3,
        patch_len_long: int = 24,
        stride_long: int | None = None,
        patch_len_short: int = 24,
        stride_short: int | None = None,
        dropout: float = 0.1,
        use_ci: bool = True,
    ):
        super().__init__()
        self.pred_len_short = short_term_pred_length
        self.pred_len_long  = long_term_pred_length    # ← ★ 추가
        self.target_idx = target_idx
        self.use_ci = use_ci

        # 정규화 + (옵션) 멀티변수→타깃단일 스트림
        self.normL = nn.LayerNorm(num_features_long)
        self.normS = nn.LayerNorm(num_features_short)
        self.mixL = None if use_ci else nn.Linear(num_features_long, 1)
        self.mixS = None if use_ci else nn.Linear(num_features_short, 1)

        # 패치 임베딩
        self.patchL = PatchEmbedding1D(d_model=d_model, patch_len=patch_len_long,  stride=stride_long)
        self.patchS = PatchEmbedding1D(d_model=d_model, patch_len=patch_len_short, stride=stride_short)
        self.pos    = SinusoidalPositionalEncoding(d_model)

        # 스트림 태깅
        self.emb_long  = nn.Parameter(torch.randn(1, 1, d_model))
        self.emb_short = nn.Parameter(torch.randn(1, 1, d_model))

        # 공유 인코더
        self.encoder = TransformerEncoder(d_model, nhead, d_ff, depth, dropout)

        # Long 컨텍스트 요약 → Short/Long 디코더 쿼리에 컨디셔닝
        self.ctx_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.ctx_mlp   = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        # 메모리 융합 게이트
        # self.fuse_logits = nn.Parameter(torch.zeros(2))
        self.fuse_logits = nn.Parameter(torch.tensor([-0.2, 0.2], dtype=torch.float))
        self.drop = nn.Dropout(dropout)

        # 디코더 레이어(공유)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True, activation='gelu'
        )

        # Short 디코더/헤드 (최종)
        self.decoder_short = nn.TransformerDecoder(dec_layer, num_layers=1)
        self.q_short = nn.Parameter(torch.randn(self.pred_len_short, d_model))
        self.pos_q   = SinusoidalPositionalEncoding(d_model)
        self.head_short = nn.Linear(d_model, 1)

        # Long 보조 디코더/헤드  ← ★ 추가
        self.decoder_long  = nn.TransformerDecoder(dec_layer, num_layers=1)
        self.q_long  = nn.Parameter(torch.randn(self.pred_len_long, d_model))
        self.head_long = nn.Linear(d_model, 1)

    def _to_target(self, x, norm, mix):
        x = norm(x)
        if self.use_ci:
            return x[:, :, self.target_idx:self.target_idx+1]   # [B,T,1]
        return mix(x) if mix is not None else x[:, :, :1]

    def _attn_pool(self, mem):  # [B,N,D] -> [B,D]
        q = self.ctx_query.expand(mem.size(0), -1, -1)             # [B,1,D]
        attn = torch.softmax(torch.matmul(q, mem.transpose(1, 2)) / math.sqrt(mem.size(-1)), dim=-1)  # [B,1,N]
        pooled = torch.matmul(attn, mem).squeeze(1)                # [B,D]
        return pooled

    def forward(self, XL, XS, XiL: torch.Tensor | None = None, XiS: torch.Tensor | None = None):
        # Long stream
        xL  = self._to_target(XL, self.normL, self.mixL)           # [B,TL,1]
        tokL = self.pos(self.patchL(xL)) + self.emb_long           # [B,NL,D]
        memL = self.encoder(tokL)                                  # [B,NL,D]

        # Short stream
        xS  = self._to_target(XS, self.normS, self.mixS)           # [B,TS,1]
        tokS = self.pos(self.patchS(xS)) + self.emb_short          # [B,NS,D]
        memS = self.encoder(tokS)                                  # [B,NS,D]

        # Long 컨텍스트 요약
        ctxL = self.ctx_mlp(self._attn_pool(memL))                 # [B,D]

        # 메모리 융합(길이 다를 수 있으므로 concat)
        w = torch.softmax(self.fuse_logits, dim=0)                 # [2]
        memory = self.drop(torch.cat([w[0] * memL, w[1] * memS], dim=1))   # [B, NL+NS, D]

        B = memory.size(0)

        # ---- Short(최종) 디코딩 ----
        qS = self.q_short.unsqueeze(0).expand(B, -1, -1)           # [B,PLs,D]
        qS = self.pos_q(qS)
        qS = qS + ctxL.unsqueeze(1)
        decS = self.decoder_short(tgt=qS, memory=memory)           # [B,PLs,D]
        outS = self.head_short(decS).squeeze(-1)                   # [B,PLs]

        # ---- Long(보조) 디코딩 ----
        qL = self.q_long.unsqueeze(0).expand(B, -1, -1)            # [B,PLl,D]
        qL = self.pos_q(qL)
        qL = qL + ctxL.unsqueeze(1)
        decL = self.decoder_long(tgt=qL, memory=memory)            # [B,PLl,D]
        outL = self.head_long(decL).squeeze(-1)                    # [B,PLl]

        # 학습 루프 호환: 항상 (outL, outS) 반환
        return outL, outS
