from .seq2seq import myTransformer

class Seq2SeqModel(myTransformer):
    def __init__(self, d_model=512):
        super().__init__(d_model=d_model)

    def forward(self, src, tgt):
        # Run the encoder steps manually, not the parent forward
        src = self.eeg_embedding(src.reshape(src.shape[0] * src.shape[1], 1, 62, 100)).reshape(src.shape[0], 7, -1)
        tgt = tgt.reshape(tgt.shape[0], tgt.shape[1], tgt.shape[2] * tgt.shape[3] * tgt.shape[4])
        tgt = self.img_embedding(tgt)

        # Add positional encodings
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # Mask for autoregression
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # Transformer
        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)

        # Instead of txtpredictor, use latent predictor
        return self.predictor(output)
