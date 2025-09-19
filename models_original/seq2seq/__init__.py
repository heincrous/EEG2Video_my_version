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
        # src: [B, 62, 5]   (DE features: 62 channels × 5 bands)
        # tgt: [B, 6, 9216] (video latents already flattened)

        if src.dim() == 3:
            # EEG: [B,62,5] -> add F=1 dimension -> [B,1,62,5]
            B, C, Bn = src.shape
            F = 1
            src = src.unsqueeze(1)
        elif src.dim() == 4:
            # EEG: [B,F,62,5]
            B, F, C, Bn = src.shape
        else:
            raise ValueError(f"Unexpected EEG input shape: {src.shape}")

        assert C == 62 and Bn == 5, f"EEG expected [*,*,62,5], got {src.shape}"

        if tgt.dim() == 3:
            # tgt: [B,6,9216] (already flattened)
            B2, F2, D = tgt.shape
            assert B2 == B, "Batch size mismatch between EEG and video latents"
            F = F2
        elif tgt.dim() == 5:
            # tgt: [B,F,4,36,64]
            B2, F2, C2, H, W = tgt.shape
            assert C2 == 4, f"Latent expected 4 channels, got {C2}"
            F = F2
            tgt = tgt.reshape(B, F, -1)
        else:
            raise ValueError(f"Unexpected latent input shape: {tgt.shape}")

        # EEG embedding: [B*F,1,62,5] -> [B,F,embed]
        src = self.eeg_embedding(src.reshape(B * F, 1, 62, 5)).reshape(B, F, -1)

        # Latent embedding: [B,F,9216] -> [B,F,embed]
        tgt = self.img_embedding(tgt)

        # Positional encodings
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # Causal mask for decoder over sequence length F
        tgt_mask = self._gen_subseq_mask(F, device=tgt.device)

        # Transformer encoder-decoder
        memory = self.transformer_encoder(src)                             # [B,F,embed]
        hidden = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)  # [B,F,embed]

        # Predict 9216 per frame (4*36*64)
        out = self.predictor(hidden)                                       # [B,F,9216]
        return out
