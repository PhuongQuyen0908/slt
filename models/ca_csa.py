"""
models/ca_csa.py

CA-CSA (Confidence-Aware Cross-Stream Attention) - Core contribution.
Drop-in replacement cho BaselineFusion trong models/baseline.py.

Architecture (theo Research Proposal):
  1. StreamTokenizer:          4 streams → 4 tokens (B, T, 4, d)
  2. LearnedConfidence:        stream features → reliability score [0,1]
  3. ConfidenceBiasAttention:  cross-stream attention + confidence bias
  4. CA_CSA_Block:             attention + FFN + residual (×2 layers)
  5. ConfidenceWeightedPooling: 4 tokens → 1 vector (B, T, d)
  6. Linear:                   (B, T, d) → (B, T, 768) cho T5

Key design: Confidence bias B[i,j] = log(c_j + eps)
  - Khi c_j cao: bias ≈ 0, attention bình thường
  - Khi c_j thấp: bias lớn âm, attention bị penalize
  - Log scale: smooth, gradient-friendly, precedent từ ALiBi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from transformers import T5ForConditionalGeneration
import math


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_STREAMS = 4
STREAM_NAMES = ["face", "lhand", "rhand", "pose"]

FACE_DIM = 384
HAND_DIM = 384
POSE_DIM = 14
T5_DIM = 768

CONF_EPS = 1e-6  # Tránh log(0)
CONF_MIN = 0.05  # Clip confidence để tránh extreme values
CONF_MAX = 0.95


# ---------------------------------------------------------------------------
# 1. Stream Tokenizer
# ---------------------------------------------------------------------------


class StreamTokenizer(nn.Module):
    """
    Project mỗi stream về cùng dimension d, thêm stream identity embedding.

    Tại sao cần:
      - Sau projection, mỗi stream có identity riêng biệt
      - Stream identity embedding encode vai trò ngôn ngữ học:
        face = grammatical markers, hand = lexical content
      - Prerequisite để cross-stream attention có nghĩa
    """

    def __init__(
        self,
        stream_dim: int = 256,
        face_dim: int = FACE_DIM,
        hand_dim: int = HAND_DIM,
        pose_dim: int = POSE_DIM,
    ):
        super().__init__()

        self.stream_dim = stream_dim

        # Stream-specific linear projections
        self.proj = nn.ModuleDict(
            {
                "face": nn.Linear(face_dim, stream_dim),
                "lhand": nn.Linear(hand_dim, stream_dim),
                "rhand": nn.Linear(hand_dim, stream_dim),
                "pose": nn.Linear(pose_dim, stream_dim),
            }
        )

        # Stream identity embeddings (4 × d)
        # Tương tự positional embedding nhưng encode linguistic role
        self.identity = nn.Embedding(NUM_STREAMS, stream_dim)

        # Register stream index mapping
        self.register_buffer(
            "stream_ids",
            torch.arange(NUM_STREAMS),  # [0, 1, 2, 3]
        )

    def forward(
        self,
        f_face: torch.Tensor,  # (B, T, 384)
        f_lhand: torch.Tensor,  # (B, T, 384)
        f_rhand: torch.Tensor,  # (B, T, 384)
        f_pose: torch.Tensor,  # (B, T, 14)
    ) -> torch.Tensor:
        """Returns: S of shape (B, T, 4, d)"""

        streams = [f_face, f_lhand, f_rhand, f_pose]
        tokens = []

        for i, (name, feat) in enumerate(zip(STREAM_NAMES, streams)):
            # Project
            s = self.proj[name](feat)  # (B, T, d)
            # Add stream identity
            s = s + self.identity(self.stream_ids[i])  # broadcast
            tokens.append(s)

        # Stack: (B, T, 4, d)
        S = torch.stack(tokens, dim=2)
        return S


# ---------------------------------------------------------------------------
# 2. Learned Confidence Module
# ---------------------------------------------------------------------------


class LearnedConfidence(nn.Module):
    """
    Predict reliability score của mỗi stream từ stream features.

    Tại sao cần Learned Confidence thay vì chỉ dùng MediaPipe score:
      - MediaPipe được train trên everyday handshapes, không phải sign language
      - Learned confidence capture sign-specific unreliability patterns
      - End-to-end training với translation objective

    Output: combined score = alpha * learned + (1-alpha) * mediapipe
    """

    def __init__(self, stream_dim: int = 256, alpha: float = 0.5):
        super().__init__()

        self.alpha = alpha

        # Lightweight MLP: stream_dim → 64 → 1
        self.scorer = nn.Sequential(
            nn.Linear(stream_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Temporal GRU smoothing (k=5 context window via hidden state)
        # Phân biệt real occlusion (nhiều frame liên tiếp thấp) vs detection noise (1 frame)
        self.temporal_gru = nn.GRU(
            input_size=1,
            hidden_size=1,
            num_layers=1,
            batch_first=True,
        )

    def forward(
        self,
        stream_token: torch.Tensor,  # (B, T, d) — AFTER tokenization
        mediapipe_conf: torch.Tensor,  # (B, T, 1) — MediaPipe confidence
    ) -> torch.Tensor:
        """Returns: temporally smoothed confidence (B, T, 1) in [CONF_MIN, CONF_MAX]"""

        # Run entirely in fp32 to prevent NaN in fp16 training
        orig_dtype = stream_token.dtype
        learned = self.scorer(stream_token.float())  # (B, T, 1) fp32
        combined = (
            self.alpha * learned + (1.0 - self.alpha) * mediapipe_conf.float()
        )  # (B, T, 1) fp32

        # Temporal GRU smoothing in fp32 (fp16 GRU over 512 steps → NaN)
        smoothed, _ = self.temporal_gru(combined)  # (B, T, 1) fp32

        # Clip và cast back
        smoothed = smoothed.clamp(CONF_MIN, CONF_MAX).to(orig_dtype)
        return smoothed


# ---------------------------------------------------------------------------
# 3. Confidence Bias Cross-Stream Attention
# ---------------------------------------------------------------------------


class ConfidenceBiasAttention(nn.Module):
    """
    Multi-head attention với additive confidence bias.

    Mechanism:
      B[i,j] = log(c_j + eps)
      Attention(Q,K,V,B) = softmax((QK^T/√d) + B) · V

    Tại sao log scale:
      - Smooth, differentiable → gradient flow tốt
      - Additive với log-space softmax (precedent: ALiBi)
      - Soft gating, không phải hard threshold
      - c_j=0.9 → B≈-0.1 (gần 0, ít ảnh hưởng)
      - c_j=0.1 → B≈-2.3 (penalty lớn)
    """

    def __init__(self, stream_dim: int = 256, num_heads: int = 4):
        super().__init__()

        assert stream_dim % num_heads == 0, (
            f"stream_dim={stream_dim} phải chia hết cho num_heads={num_heads}"
        )

        self.num_heads = num_heads
        self.head_dim = stream_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.W_Q = nn.Linear(stream_dim, stream_dim, bias=True)
        self.W_K = nn.Linear(stream_dim, stream_dim, bias=True)
        self.W_V = nn.Linear(stream_dim, stream_dim, bias=True)
        self.out = nn.Linear(stream_dim, stream_dim, bias=True)

    def forward(
        self,
        S: torch.Tensor,  # (B, T, 4, d)
        confidence_scores: torch.Tensor,  # (B, T, 4) — combined confidence
    ) -> torch.Tensor:
        """Returns: attended S of shape (B, T, 4, d)"""

        B, T, N, d = S.shape  # N = 4 streams

        # --- Build confidence bias matrix ---
        # B[i,j] = log(c_j + eps): stream i attending to stream j
        # c_j thấp → penalty khi attend đến stream j
        # Compute in fp32 to avoid fp16 underflow → NaN
        c = confidence_scores.float().clamp(CONF_MIN, CONF_MAX)  # (B, T, 4) fp32
        log_c = torch.log(c + CONF_EPS).clamp(-5.0, 0.0)  # (B, T, 4) fp32, bounded
        # Expand: bias[b,t,i,j] = log_c[b,t,j] (chỉ phụ thuộc vào key stream)
        bias = log_c.unsqueeze(-2).expand(B, T, N, N).to(S.dtype)  # cast back

        # --- Multi-head attention ---
        # Reshape (B, T, N, d) → (B, T, N, H, head_d) → (B, T, H, N, head_d)
        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(B, T, N, self.num_heads, self.head_dim).permute(
                0, 1, 3, 2, 4
            )  # (B, T, H, N, head_d)

        Q = split_heads(self.W_Q(S))  # (B, T, H, 4, head_d)
        K = split_heads(self.W_K(S))
        V = split_heads(self.W_V(S))

        # Attention scores - compute in fp32 for numerical stability
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, T, H, 4, 4)

        # Add confidence bias (broadcast over heads)
        bias_expanded = bias.unsqueeze(2).expand_as(attn)  # (B, T, H, 4, 4)
        attn = attn + bias_expanded

        attn = F.softmax(attn.float(), dim=-1).to(S.dtype)  # fp32 softmax → cast back

        # Apply attention
        out = torch.matmul(attn, V)  # (B, T, H, 4, head_d)

        # Merge heads
        out = out.permute(0, 1, 3, 2, 4).contiguous()  # (B, T, 4, H, head_d)
        out = out.view(B, T, N, d)  # (B, T, 4, d)

        return self.out(out)


# ---------------------------------------------------------------------------
# 4. CA-CSA Block (Attention + FFN + Residual)
# ---------------------------------------------------------------------------


class CACsaBlock(nn.Module):
    """
    Một CA-CSA block: confidence-bias attention + FFN, cả hai với residual.
    Paper dùng 2 blocks chồng nhau.

    Tại sao 2 blocks là đủ:
      - 4 streams là nhỏ → 2 layers attention đủ để mọi stream attend đến nhau
      - Tránh overfitting
      - Compute overhead negligible
    """

    def __init__(
        self,
        stream_dim: int = 256,
        num_heads: int = 4,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = ConfidenceBiasAttention(stream_dim, num_heads)
        self.norm1 = nn.LayerNorm(stream_dim)
        self.norm2 = nn.LayerNorm(stream_dim)

        ffn_dim = stream_dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(stream_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, stream_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        S: torch.Tensor,  # (B, T, 4, d)
        confidence_scores: torch.Tensor,  # (B, T, 4)
    ) -> torch.Tensor:
        """Returns: updated S of shape (B, T, 4, d)"""

        # Pre-norm + attention + residual
        S = S + self.attn(self.norm1(S), confidence_scores)

        # Pre-norm + FFN + residual
        # FFN apply độc lập cho từng stream token
        S = S + self.ffn(self.norm2(S))

        return S


# ---------------------------------------------------------------------------
# 5. Confidence-Weighted Pooling
# ---------------------------------------------------------------------------


class ConfidenceWeightedPooling(nn.Module):
    """
    Pool 4 stream tokens thành 1 vector, weighted by confidence.

    w = softmax([c_face, c_lhand, c_rhand, c_pose])
    f_fused = Σ w_i · s_i

    Tại sao không dùng mean pooling:
      - Mean treat streams như nhau bất kể reliability
      - Stream với confidence 0.1 vẫn contribute bằng stream 0.9
      - Confidence-weighted pooling là second level của confidence awareness:
        lần 1 ở attention (how streams interact)
        lần 2 ở pooling (how streams contribute to output)
    """

    def __init__(self):
        super().__init__()
        # Không có trainable params: trực tiếp dùng confidence scores

    def forward(
        self,
        S: torch.Tensor,  # (B, T, 4, d) — refined tokens
        confidence_scores: torch.Tensor,  # (B, T, 4)
    ) -> torch.Tensor:
        """Returns: (B, T, d) fused representation"""

        # Weights = softmax over stream dimension
        w = F.softmax(confidence_scores, dim=-1)  # (B, T, 4)
        w = w.unsqueeze(-1)  # (B, T, 4, 1)

        # Weighted sum
        fused = (w * S).sum(dim=2)  # (B, T, d)
        return fused


# ---------------------------------------------------------------------------
# 6. Full CA-CSA Fusion Module
# ---------------------------------------------------------------------------


class CACsaFusion(nn.Module):
    """
    Drop-in replacement cho BaselineFusion.

    Input:
      f_face, f_lhand, f_rhand: (B, T, 384)
      f_pose:                    (B, T, 14)
      mp_conf_face, mp_conf_lhand, mp_conf_rhand, mp_conf_pose: (B, T, 1)

    Output: (B, T, 768) ready cho T5

    Full pipeline:
      Tokenize → LearnedConf → ConfBiasAttn×2 → ConfWeightedPool → Linear
    """

    def __init__(
        self,
        stream_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_ratio: int = 4,
        alpha: float = 0.5,  # blend ratio learned vs mediapipe conf
        dropout: float = 0.1,
        t5_dim: int = T5_DIM,
    ):
        super().__init__()

        self.stream_dim = stream_dim

        # Step 1: Tokenization
        self.tokenizer = StreamTokenizer(stream_dim=stream_dim)

        # Step 2: Learned Confidence (một module per stream)
        self.confidence = nn.ModuleDict(
            {
                name: LearnedConfidence(stream_dim=stream_dim, alpha=alpha)
                for name in STREAM_NAMES
            }
        )

        # Step 3-4: CA-CSA blocks (×num_layers)
        self.blocks = nn.ModuleList(
            [
                CACsaBlock(
                    stream_dim=stream_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Step 5: Confidence-weighted pooling
        self.pooling = ConfidenceWeightedPooling()

        # Step 6: Output projection → T5 dim
        self.proj_out = nn.Linear(stream_dim, t5_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        f_face: torch.Tensor,  # (B, T, 384)
        f_lhand: torch.Tensor,  # (B, T, 384)
        f_rhand: torch.Tensor,  # (B, T, 384)
        f_pose: torch.Tensor,  # (B, T, 14)
        mp_conf_face: Optional[torch.Tensor] = None,  # (B, T, 1)
        mp_conf_lhand: Optional[torch.Tensor] = None,
        mp_conf_rhand: Optional[torch.Tensor] = None,
        mp_conf_pose: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Returns: (B, T, 768)"""

        B, T, _ = f_face.shape

        # Default: nếu không có MediaPipe confidence → dùng 0.5 (neutral)
        def default_conf(c):
            if c is None:
                return torch.full((B, T, 1), 0.5, device=f_face.device)
            return c

        mp_conf_face = default_conf(mp_conf_face)
        mp_conf_lhand = default_conf(mp_conf_lhand)
        mp_conf_rhand = default_conf(mp_conf_rhand)
        mp_conf_pose = default_conf(mp_conf_pose)

        # --- Step 1: Stream Tokenization ---
        S = self.tokenizer(f_face, f_lhand, f_rhand, f_pose)  # (B, T, 4, d)

        # --- Step 2: Learned Confidence ---
        # S[:,:,i,:] = token của stream i
        mp_confs = [mp_conf_face, mp_conf_lhand, mp_conf_rhand, mp_conf_pose]
        conf_scores = []

        for i, name in enumerate(STREAM_NAMES):
            c = self.confidence[name](S[:, :, i, :], mp_confs[i])  # (B, T, 1)
            conf_scores.append(c)

        # Stack: (B, T, 4)
        conf_scores = torch.cat(conf_scores, dim=-1)

        # --- Step 3-4: CA-CSA blocks ---
        for block in self.blocks:
            S = block(S, conf_scores)

        # --- Step 5: Confidence-Weighted Pooling ---
        fused = self.pooling(S, conf_scores)  # (B, T, d)

        # --- Step 6: Project → T5 ---
        out = self.proj_out(self.dropout(fused))  # (B, T, 768)
        return out

    def get_attention_weights(
        self,
        f_face: torch.Tensor,
        f_lhand: torch.Tensor,
        f_rhand: torch.Tensor,
        f_pose: torch.Tensor,
        mp_conf_face: Optional[torch.Tensor] = None,
        mp_conf_lhand: Optional[torch.Tensor] = None,
        mp_conf_rhand: Optional[torch.Tensor] = None,
        mp_conf_pose: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Debug/analysis: trả về attention weights và confidence scores.
        Dùng trong analysis/attn_viz.py.
        """
        B, T, _ = f_face.shape

        def default_conf(c):
            if c is None:
                return torch.full((B, T, 1), 0.5, device=f_face.device)
            return c

        mp_conf_face = default_conf(mp_conf_face)
        mp_conf_lhand = default_conf(mp_conf_lhand)
        mp_conf_rhand = default_conf(mp_conf_rhand)
        mp_conf_pose = default_conf(mp_conf_pose)

        S = self.tokenizer(f_face, f_lhand, f_rhand, f_pose)
        mp_confs = [mp_conf_face, mp_conf_lhand, mp_conf_rhand, mp_conf_pose]
        conf_scores_list = []

        for i, name in enumerate(STREAM_NAMES):
            c = self.confidence[name](S[:, :, i, :], mp_confs[i])
            conf_scores_list.append(c)

        conf_scores = torch.cat(conf_scores_list, dim=-1)  # (B, T, 4)

        return {
            "confidence_scores": conf_scores,  # (B, T, 4)
            "stream_names": STREAM_NAMES,
            "conf_min": conf_scores.min().item(),
            "conf_max": conf_scores.max().item(),
            "conf_mean": conf_scores.mean(dim=[0, 1]).tolist(),  # per-stream mean
        }


# ---------------------------------------------------------------------------
# Full CA-CSA Model
# ---------------------------------------------------------------------------


class SignMusketeersCACSA(nn.Module):
    """
    Full model với CA-CSA fusion. Interface giống SignMusketeersBaseline.
    """

    def __init__(
        self,
        t5_model_name: str = "google/t5-v1_1-base",
        stream_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_ratio: int = 4,
        alpha: float = 0.5,
        dropout: float = 0.1,
        label_smoothing: float = 0.2,
    ):
        super().__init__()

        # CA-CSA fusion (replaces BaselineFusion)
        self.fusion = CACsaFusion(
            stream_dim=stream_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_ratio=ffn_ratio,
            alpha=alpha,
            dropout=dropout,
        )

        print(f"[Model] Loading T5: {t5_model_name}...")
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)

        self.label_smoothing = label_smoothing

        # Count và log thêm params so với baseline
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        print(f"[CA-CSA] Fusion params: {fusion_params / 1e6:.2f}M")

    def forward(
        self,
        f_face: torch.Tensor,
        f_lhand: torch.Tensor,
        f_rhand: torch.Tensor,
        f_pose: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mp_conf_face: Optional[torch.Tensor] = None,
        mp_conf_lhand: Optional[torch.Tensor] = None,
        mp_conf_rhand: Optional[torch.Tensor] = None,
        mp_conf_pose: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        encoder_hidden = self.fusion(
            f_face=f_face,
            f_lhand=f_lhand,
            f_rhand=f_rhand,
            f_pose=f_pose,
            mp_conf_face=mp_conf_face,
            mp_conf_lhand=mp_conf_lhand,
            mp_conf_rhand=mp_conf_rhand,
            mp_conf_pose=mp_conf_pose,
        )

        outputs = self.t5(
            inputs_embeds=encoder_hidden,
            attention_mask=attention_mask,
            labels=labels,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "encoder_outputs": outputs.encoder_last_hidden_state,
        }

    @torch.no_grad()
    def generate(
        self,
        f_face: torch.Tensor,
        f_lhand: torch.Tensor,
        f_rhand: torch.Tensor,
        f_pose: torch.Tensor,
        attention_mask: torch.Tensor,
        mp_conf_face: Optional[torch.Tensor] = None,
        mp_conf_lhand: Optional[torch.Tensor] = None,
        mp_conf_rhand: Optional[torch.Tensor] = None,
        mp_conf_pose: Optional[torch.Tensor] = None,
        max_length: int = 128,
        num_beams: int = 5,
        **kwargs,
    ) -> torch.Tensor:

        from transformers.modeling_outputs import BaseModelOutput

        encoder_hidden = self.fusion(
            f_face=f_face,
            f_lhand=f_lhand,
            f_rhand=f_rhand,
            f_pose=f_pose,
            mp_conf_face=mp_conf_face,
            mp_conf_lhand=mp_conf_lhand,
            mp_conf_rhand=mp_conf_rhand,
            mp_conf_pose=mp_conf_pose,
        )

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

        return self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
