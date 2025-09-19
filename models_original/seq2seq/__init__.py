from .seq2seq import myTransformer

class Seq2SeqModel(myTransformer):
    def __init__(self, d_model=512):
        super().__init__(d_model=d_model)

    def forward(self, src, tgt):
        # Call parent forward
        out = super().forward(src, tgt)
        if isinstance(out, tuple):
            pred, _ = out  # if tuple, take first
        else:
            pred = out

        # Force latent regression head (9216 per frame)
        return self.predictor(pred)
