from .seq2seq import myTransformer

class Seq2SeqModel(myTransformer):
    def __init__(self, d_model=512):
        super().__init__(d_model=d_model)

    def forward(self, src, tgt):
        # EEG embedding: reshape to [B*frames,1,62,100] then back
        src = self.eeg_embedding(src.reshape(src.shape[0] * src.shape[1], 1, 62, 100)).reshape(src.shape[0], src.shape[1], -1)
        # Target embedding: flatten [B,frames,4,36,64] -> [B,frames,9216]
        tgt = tgt.reshape(tgt.shape[0], tgt.shape[1], -1)
        tgt = self.img_embedding(tgt)

        # Add positional encoding
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # Mask for autoregression
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # Transformer encode/decode
        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)

        # Latent regression head (9216)
        return self.predictor(output)
