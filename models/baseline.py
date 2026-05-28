"""
models/baseline.py

Re-implementation của SignMusketeers Phase 2 (Section 3.3).
Fusion strategy gốc: concat 4 streams → Linear projection → T5

Architecture:
  - Face:    (B, T, 384) → Linear → (B, T, 256)
  - L.Hand:  (B, T, 384) → Linear → (B, T, 256)
  - R.Hand:  (B, T, 384) → Linear → (B, T, 256)
  - Pose:    (B, T, 14)  → Linear → (B, T, 128)
  - Concat:  (B, T, 256+256+256+128) = (B, T, 896)
  - Project: Linear(896 → 768) → input cho T5

Note về dimensions từ paper:
  - Paper nói face/hand project về 256, pose về 128
  - Concat = 256*3 + 128 = 896 (bukan 1166 seperti di conversation awal)
  - 1166 = 384*3 + 14 nếu không project trước khi concat
  - Paper Section 3.3 rõ ràng: project trước, concat sau
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Dict


# ---------------------------------------------------------------------------
# Dimensions (theo paper Section 3.3)
# ---------------------------------------------------------------------------

FACE_DIM = 384  # DINOv2-Small CLS token
HAND_DIM = 384
POSE_DIM = 14  # 7 landmarks × 2

FACE_PROJ = 256  # Sau stream-specific linear
HAND_PROJ = 256
POSE_PROJ = 128

CONCAT_DIM = FACE_PROJ + HAND_PROJ + HAND_PROJ + POSE_PROJ  # = 896
T5_DIM = 768  # T5-Base input dimension


# ---------------------------------------------------------------------------
# SignMusketeers Baseline Fusion
# ---------------------------------------------------------------------------


class BaselineFusion(nn.Module):
    """
    Fusion module của SignMusketeers gốc.
    Thay thế module này bằng CACSA trong ca_csa.py.

    Input (mỗi frame):
      f_face:  (B, T, 384) - precomputed features
      f_lhand: (B, T, 384)
      f_rhand: (B, T, 384)
      f_pose:  (B, T, 14)

    Output:
      fused: (B, T, 768) - ready cho T5 encoder
    """

    def __init__(
        self,
        face_dim: int = FACE_DIM,
        hand_dim: int = HAND_DIM,
        pose_dim: int = POSE_DIM,
        face_proj: int = FACE_PROJ,
        hand_proj: int = HAND_PROJ,
        pose_proj: int = POSE_PROJ,
        t5_dim: int = T5_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Stream-specific projections (trained from scratch)
        self.proj_face = nn.Linear(face_dim, face_proj)
        self.proj_lhand = nn.Linear(hand_dim, hand_proj)
        self.proj_rhand = nn.Linear(hand_dim, hand_proj)
        self.proj_pose = nn.Linear(pose_dim, pose_proj)

        concat_dim = face_proj + hand_proj + hand_proj + pose_proj

        # Final projection → T5 input dim
        self.proj_out = nn.Linear(concat_dim, t5_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        f_face: torch.Tensor,  # (B, T, 384)
        f_lhand: torch.Tensor,  # (B, T, 384)
        f_rhand: torch.Tensor,  # (B, T, 384)
        f_pose: torch.Tensor,  # (B, T, 14)
        **kwargs,  # Ignore confidence scores nếu có
    ) -> torch.Tensor:
        """Returns: (B, T, 768)"""

        s_face = self.proj_face(f_face)  # (B, T, 256)
        s_lhand = self.proj_lhand(f_lhand)  # (B, T, 256)
        s_rhand = self.proj_rhand(f_rhand)  # (B, T, 256)
        s_pose = self.proj_pose(f_pose)  # (B, T, 128)

        # Concat tất cả streams
        concat = torch.cat([s_face, s_lhand, s_rhand, s_pose], dim=-1)  # (B, T, 896)

        # Project → T5 dimension
        fused = self.proj_out(self.dropout(concat))  # (B, T, 768)
        return fused


# ---------------------------------------------------------------------------
# Full SignMusketeers Model (Baseline)
# ---------------------------------------------------------------------------


class SignMusketeersBaseline(nn.Module):
    """
    Full model: Fusion → T5 translation.

    Trong training: features đã được precompute, nên chỉ cần:
      1. Fusion module (trainable)
      2. T5 (fine-tuned)

    Encoders (DINOv2-F, DINOv2-H) KHÔNG nằm trong model này
    → xem models/encoders.py và pipeline/precompute.py
    """

    def __init__(
        self,
        t5_model_name: str = "google/t5-v1_1-base",
        fusion_dropout: float = 0.1,
        label_smoothing: float = 0.2,  # Theo paper Table 8
    ):
        super().__init__()

        # Fusion module
        self.fusion = BaselineFusion(dropout=fusion_dropout)

        # T5 model
        print(f"[Model] Loading T5: {t5_model_name}...")
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)

        self.label_smoothing = label_smoothing
        self.t5_dim = T5_DIM

        # Verify T5 hidden dim match
        assert self.t5.config.d_model == T5_DIM, (
            f"T5 d_model={self.t5.config.d_model} ≠ expected {T5_DIM}"
        )

    def forward(
        self,
        f_face: torch.Tensor,  # (B, T, 384) precomputed
        f_lhand: torch.Tensor,  # (B, T, 384)
        f_rhand: torch.Tensor,  # (B, T, 384)
        f_pose: torch.Tensor,  # (B, T, 14)
        attention_mask: torch.Tensor,  # (B, T) frame-level mask
        labels: Optional[torch.Tensor] = None,  # (B, L) token ids
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns dict với:
          - 'loss': scalar (nếu labels được cung cấp)
          - 'logits': (B, L, vocab_size)
          - 'encoder_outputs': để dùng lại khi generate
        """
        # 1. Fusion: 4 streams → (B, T, 768)
        encoder_hidden = self.fusion(
            f_face=f_face,
            f_lhand=f_lhand,
            f_rhand=f_rhand,
            f_pose=f_pose,
        )

        # 2. T5 forward
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
        max_length: int = 128,
        num_beams: int = 5,
        **kwargs,
    ) -> torch.Tensor:
        """
        Beam search generation.
        Theo paper: max_length=128, num_beams=5 (Table 8)
        """
        encoder_hidden = self.fusion(
            f_face=f_face,
            f_lhand=f_lhand,
            f_rhand=f_rhand,
            f_pose=f_pose,
        )

        # Tạo encoder outputs object cho T5 generate
        from transformers.modeling_outputs import BaseModelOutput

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

        generated = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

        return generated

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
