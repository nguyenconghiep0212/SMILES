import torch
import torch.nn as nn

from src.models.cnn_encoder import ResNet18Encoder
from src.models.transformer_decoder import SmilesTransformerDecoder


class ImageToSmilesModel(nn.Module):
    def __init__(self, vocab_size, max_len=256):
        super().__init__()
        self.encoder = ResNet18Encoder(pretrained=True)
        self.decoder = SmilesTransformerDecoder(vocab_size=vocab_size, max_len=max_len)

    def forward(self, images, tgt_tokens):
        """
        images: (B, 3, 256, 256)
        tgt_tokens: (B, T) - input to decoder

        returns logits: (B, T, vocab_size)
        """
        memory = self.encoder(images)
        logits = self.decoder(tgt_tokens, memory)
        return logits