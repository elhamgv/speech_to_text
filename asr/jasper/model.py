import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple

from asr.jasper.decoder import JasperDecoder
from asr.jasper.encoder import JasperEncoder
from asr.jasper.Jasper5x3EncoderConfig import Jasper5x3EncoderConfig
from asr.jasper.Jasper10x5EncoderConfig import Jasper10x5EncoderConfig
from asr.jasper import JasperDecoderConfig

class Jasper(nn.Module):
    """
    Jasper: An End-to-End Convolutional Neural Acoustic Model
    Jasper (Just Another Speech Recognizer), an ASR model comprised of 54 layers proposed by NVIDIA.
    Jasper achieved sub 3 percent word error rate (WER) on the LibriSpeech dataset.
    More details: https://arxiv.org/pdf/1904.03288.pdf

    Args:
        num_classes (int): number of classification
        version (str): version of jasper. Marked as BxR: B - number of blocks, R - number of sub-blocks
        device (torch.device): device - 'cuda' or 'cpu'

    Inputs: inputs, input_lengths, residual
        - **inputs**: tensor contains input sequence vector
        - **input_lengths**: tensor contains sequence lengths
        - **residual**: tensor contains residual vector

    Returns: output, output_lengths
        - **output**: tensor contains output sequence vector
        - **output**: tensor contains output sequence lengths
    """

    def __init__(self, num_classes: int, version: str = '10x5', device: torch.device = 'cuda') -> None:
        super(Jasper, self).__init__()
        supported_versions = {
            '10x5': {
                'encoder_config': Jasper10x5EncoderConfig(num_blocks=10, num_sub_blocks=5),
                'decoder_config': JasperDecoderConfig(num_classes),
            },
            '5x3': {
                'encoder_config': Jasper5x3EncoderConfig(num_blocks=5, num_sub_blocks=3),
                'decoder_config': JasperDecoderConfig(num_classes),
            },
        }
        assert version.lower() in supported_versions.keys(), "Unsupported Version: {}".format(version)

        self.encoder = JasperEncoder(config=supported_versions[version]['encoder_config'], device=device)
        self.decoder = JasperDecoder(config=supported_versions[version]['decoder_config'], device=device)
        self.device = device

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        inputs (torch.FloatTensor): (batch_size, sequence_length, dimension)
        input_lengths (torch.LongTensor): (batch_size)
        """
        encoder_outputs, output_lengths = self.encoder(inputs.transpose(1, 2), input_lengths)
        output, output_lengths = self.decoder(encoder_outputs, output_lengths)
        return output, output_lengths

    def greedy_search(self, inputs: Tensor, input_lengths: Tensor, device: str):
        with torch.no_grad():
            output, output_lengths = self.forward(inputs.transpose(1, 2), input_lengths)
            return output.max(-1)[1]
