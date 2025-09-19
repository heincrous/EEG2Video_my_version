from .seq2seq import myTransformer

class Seq2SeqModel(myTransformer):
    def __init__(self, d_model=512):
        super().__init__(d_model=d_model)

    def forward(self, src, tgt):
        # Call the parent forward
        out = super().forward(src, tgt)

        # Parent forward may return logits (B,13) or embeddings before predictor
        if isinstance(out, tuple):
            pred = out[0]
        else:
            pred = out

        # If the output looks like classification logits, re-map it
        if pred.shape[-1] == 13:
            # Take the decoder hidden states before classification
            # In parent code: 'out' just before txtpredictor is the hidden reps
            # Here we assume 'pred' is decoder hidden reps, so project to 9216
            return self.predictor(pred)
        else:
            return pred
