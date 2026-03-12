import torch
import torch.nn as nn

# generate SMILES tokens autoregressively.
class SmilesTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=4, max_len=256):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.positional_embed = nn.Embedding(max_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.output_layer = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, tgt, memory):
        """
        tgt shape: (B, T)
        memory shape: (B, S, 512) from encoder

        Returns logits: (B, T, vocab_size)
        """

        B, T = tgt.shape
        positions = torch.arange(0, T, device=tgt.device).unsqueeze(0)

        tgt_emb = self.token_embed(tgt) + self.positional_embed(positions)

        # Mask: decoder cannot see future signals
        tgt_mask = torch.triu(torch.ones(T, T, device=tgt.device) == 1, diagonal=1)
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))

        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask
        )

        return self.output_layer(output)
