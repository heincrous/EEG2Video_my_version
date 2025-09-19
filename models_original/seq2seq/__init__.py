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
        # src: [B,F,62,100]   (time-window EEG)
        # tgt: [B,6,9216] or [B,F,4,36,64] (video latents)

        # Unpack EEG dimensions
        B, F, C, T = src.shape
        assert C == 62 and T == 100, f"EEG expected [B,F,62,100], got {src.shape}"

        # EEG embedding: [B*F,1,62,100] -> [B,F,embed]
        src = self.eeg_embedding(src.reshape(B * F, 1, 62, 100)).reshape(B, F, -1)

        # Video latents
        if tgt.dim() == 5:   # [B,F,4,36,64]
            tgt = tgt.reshape(B, tgt.size(1), -1)
        elif tgt.dim() == 3: # [B,6,9216]
            pass  # already flattened
        else:
            raise ValueError(f"Unexpected latent input shape: {tgt.shape}")

        # Latent embedding: [B,F,9216] -> [B,F,embed]
        tgt = self.img_embedding(tgt)

        # Positional encodings
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # Causal mask for decoder
        tgt_mask = self._gen_subseq_mask(tgt.size(1), device=tgt.device)

        # Transformer encoder-decoder
        memory = self.transformer_encoder(src)                             # [B,F,embed]
        hidden = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)  # [B,F,embed]

        # Predictor → [B,F,9216]
        out = self.predictor(hidden)
        return out
