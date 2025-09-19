import torch
from .seq2seq import myTransformer

class Seq2SeqModel(myTransformer):
    def __init__(self, d_model=512):
        super().__init__(d_model=d_model)

    @staticmethod
    def _gen_subseq_mask(sz, device):
        # standard causal mask for decoder
        m = torch.full((sz, sz), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    def forward(self, src, tgt):
        # src: [B,62,5]    (EEG DE features)
        # tgt: [B,6,9216]  (video latents, flattened)

        B, C, Bn = src.shape
        assert C == 62 and Bn == 5, f"EEG expected [B,62,5], got {src.shape}"
        assert tgt.dim() == 3 and tgt.shape[1] == 6, f"Latent expected [B,6,9216], got {tgt.shape}"

        # Add fake sequence dimension: [B,1,62,5]
        src = src.unsqueeze(1)

        # EEG embedding: [B*1,1,62,5] -> [B,1,embed]
        src = self.eeg_embedding(src.reshape(B, 1, 62, 5)).reshape(B, 1, -1)

        # Video latent embedding: [B,6,9216] -> [B,6,embed]
        tgt = self.img_embedding(tgt)

        # Positional encodings
        src = self.positional_encoding(src)   # [B,1,embed]
        tgt = self.positional_encoding(tgt)   # [B,6,embed]

        # Causal mask for decoder (over 6 frames)
        tgt_mask = self._gen_subseq_mask(tgt.size(1), device=tgt.device)

        # Transformer encoder-decoder
        memory = self.transformer_encoder(src)                              # [B,1,embed]
        hidden = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)   # [B,6,embed]

        # Predictor → [B,6,9216]
        out = self.predictor(hidden)
        return out
