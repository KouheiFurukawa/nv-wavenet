import torch
from .cpc import CPC

def audio_model():
    strides = [4, 4, 2, 2, 2, 2]
    filter_sizes = [8, 8, 4, 4, 4, 4]
    padding = [2, 2, 2, 2, 2, 1]
    genc_hidden = 128
    gar_hidden = 128

    model = Model(
        strides=strides,
        filter_sizes=filter_sizes,
        padding=padding,
        genc_hidden=genc_hidden,
        gar_hidden=gar_hidden,
    )
    return model


class Model(torch.nn.Module):
    def __init__(
        self, strides, filter_sizes, padding, genc_hidden, gar_hidden,
    ):
        super(Model, self).__init__()

        self.strides = strides
        self.filter_sizes = filter_sizes
        self.padding = padding
        self.genc_input = 1
        self.genc_hidden = genc_hidden
        self.gar_hidden = gar_hidden

        self.model = CPC(
            strides,
            filter_sizes,
            padding,
            self.genc_input,
            genc_hidden,
            gar_hidden,
        )

    def forward(self, x):
        """Forward through the network"""

        loss, accuracy, _, z = self.model(x)
        return loss
