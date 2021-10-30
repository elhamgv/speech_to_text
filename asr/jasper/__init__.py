class JasperEncoderConfig:
    def _init__(self) -> None:
        self.num_blocks = None
        self.num_sub_blocks = None
        self.preprocess_block = None
        self.block = None


class JasperDecoderConfig:
    def __init__(self, num_classes: int) -> None:
        super(JasperDecoderConfig, self).__init__()
        self.num_classes = num_classes
        self.block = {
            'in_channels': (768, 896, 1024),
            'out_channels': (896, 1024, num_classes),
            'kernel_size': (29, 1, 1),
            'dilation': (2, 1, 1),
            'dropout_p': (0.4, 0.4, 0.0)
        }


from asr.jasper.model import Jasper