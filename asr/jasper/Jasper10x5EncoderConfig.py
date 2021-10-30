

class JasperEncoderConfig:
    def _init__(self) -> None:
        self.num_blocks = None
        self.num_sub_blocks = None
        self.preprocess_block = None
        self.block = None


class Jasper10x5EncoderConfig(JasperEncoderConfig):
    def __init__(self, num_blocks: int, num_sub_blocks: int) -> None:
        super(JasperEncoderConfig, self).__init__()
        self.num_blocks = num_blocks
        self.num_sub_blocks = num_sub_blocks
        self.preprocess_block = {
            'in_channels': 80,
            'out_channels': 256,
            'kernel_size': 11,
            'stride': 2,
            'dilation': 1,
            'dropout_p': 0.2,
        }
        self.block = {
            'in_channels': (256, 256, 256, 384, 384, 512, 512, 640, 640, 768),
            'out_channels': (256, 256, 384, 384, 512, 512, 640, 640, 768, 768),
            'kernel_size': (11, 11, 13, 13, 17, 17, 21, 21, 25, 25),
            'dilation': [1] * 10,
            'dropout_p': (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3),
        }