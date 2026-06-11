import torch
from torch import nn


class ConcatFusion(nn.Module):
    """Concatenate audio/video embeddings and project them to class logits.

    The baseline CLIP+CLAP model expects fusion modules to return both logits and
    the fused feature representation so callers can optionally save features.
    """

    def __init__(self, input_dim, output_dim, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, audio_features, video_features):
        fused_features = torch.cat((audio_features, video_features), dim=-1)
        logits = self.classifier(self.dropout(fused_features))
        return logits, fused_features
