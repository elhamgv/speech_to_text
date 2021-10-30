import torch
from asr.jasper import Jasper
BATCH_SIZE, SEQ_LENGTH, DIM = 3, 12345, 80

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

inputs = torch.rand(BATCH_SIZE, SEQ_LENGTH, DIM).to(device)  # BxTxD
input_lengths = torch.LongTensor([SEQ_LENGTH, SEQ_LENGTH - 10, SEQ_LENGTH - 20]).to(device)

# Jasper 10x3 Model Test
model = Jasper(num_classes=10, version='10x5', device=device).to(device)
output, output_lengths = model(inputs, input_lengths)

# Jasper 5x3 Model Test
model = Jasper(num_classes=10, version='5x3', device=device).to(device)
output, output_lengths = model(inputs, input_lengths)