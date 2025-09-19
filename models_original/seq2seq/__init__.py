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
        # src: [B, F, 62, 100]
        # tgt: [B, F, 4, 36, 64]
        B, F, C, T = src.shape
        assert C == 62 and T == 100, f"EEG expected [*,*,62,100], got {src.shape}"
        assert tgt.dim() == 5 and tgt.size(2) == 4, f"Latent expected [*,*,4,36,64], got {tgt.shape}"

        # EEG embedding: [B*F,1,62,100] -> [B,F,embed]
        src = self.eeg_embedding(src.reshape(B * F, 1, 62, 100)).reshape(B, F, -1)

        # Latent embedding: flatten -> img embedding -> [B,F,embed]
        tgt = tgt.reshape(B, F, -1)
        tgt = self.img_embedding(tgt)

        # Positional encodings
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # Causal mask for decoder over F
        tgt_mask = self._gen_subseq_mask(F, device=tgt.device)

        # Transformer
        memory = self.transformer_encoder(src)                  # [B,F,embed]
        hidden  = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)  # [B,F,embed]

        # Predict 9216 per frame (4*36*64)
        out = self.predictor(hidden)                            # [B,F,9216]
        return out
