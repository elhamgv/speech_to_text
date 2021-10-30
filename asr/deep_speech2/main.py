import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from utils.transformation import TextTransform
from network.bidirectional_gru import BidirectionalGRU
from network.residual_cnn import ResidualCNN
from asr.deep_speech2.pre_processing import data_processing
from metrics.asr_metrics import cer,wer
from comet_ml import Experiment

train_dataset = torchaudio.datasets.LIBRISPEECH("./", url="train-clean-100", download=True)
test_dataset = torchaudio.datasets.LIBRISPEECH("./", url="test-clean", download=True)

train_audio_transforms = nn.Sequential(
  torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
  torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
  torchaudio.transforms.TimeMasking(time_mask_param=100))

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

text_transform = TextTransform()


def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
  arg_maxes = torch.argmax(output, dim=2)
  decodes = []
  targets = []
  for i, args in enumerate(arg_maxes):
    decode = []
    targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
    for j, index in enumerate(args):
      if index != blank_label:
        if collapse_repeated and j != 0 and index == args[j - 1]:
          continue
        decode.append(index.item())
    decodes.append(text_transform.int_to_text(decode))
  return decodes, targets


class SpeechRecognitionModel(nn.Module):

  def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
    super(SpeechRecognitionModel, self).__init__()
    n_feats = n_feats // 2
    self.cnn = nn.Conv2d(
      1, 32, 3, stride=stride, padding=3 // 2)  # cnn for extracting heirachal features

    # n residual cnn layers with filter size of 32
    self.rescnn_layers = nn.Sequential(
      *[
        ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
        for _ in range(n_cnn_layers)
      ])
    self.fully_connected = nn.Linear(n_feats * 32, rnn_dim)
    self.birnn_layers = nn.Sequential(
      *[
        BidirectionalGRU(
          rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
          hidden_size=rnn_dim,
          dropout=dropout,
          batch_first=i == 0) for i in range(n_rnn_layers)
      ])
    self.classifier = nn.Sequential(
      nn.Linear(rnn_dim * 2, rnn_dim),  # birnn returns rnn_dim*2
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(rnn_dim, n_class))

  def forward(self, x):
    x = self.cnn(x)
    x = self.rescnn_layers(x)
    sizes = x.size()
    x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
    x = x.transpose(1, 2)  # (batch, time, feature)
    x = self.fully_connected(x)
    x = self.birnn_layers(x)
    x = self.classifier(x)
    return x


class IterMeter(object):
  """keeps track of total iterations"""

  def __init__(self):
    self.val = 0

  def step(self):
    self.val += 1

  def get(self):
    return self.val


def train(
    model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment):
  model.train()
  data_len = len(train_loader.dataset)
  with experiment.train():
    for batch_idx, _data in enumerate(train_loader):
      spectrograms, labels, input_lengths, label_lengths = _data
      spectrograms, labels = spectrograms.to(device), labels.to(device)

      optimizer.zero_grad()

      output = model(spectrograms)  # (batch, time, n_class)
      output = F.log_softmax(output, dim=2)
      output = output.transpose(0, 1)  # (time, batch, n_class)

      loss = criterion(output, labels, input_lengths, label_lengths)
      loss.backward()

      experiment.log_metric('loss', loss.item(), step=iter_meter.get())
      experiment.log_metric('learning_rate', scheduler.get_lr(), step=iter_meter.get())

      optimizer.step()
      scheduler.step()
      iter_meter.step()
      if batch_idx % 100 == 0 or batch_idx == data_len:
        print(
          'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(spectrograms), data_len, 100. * batch_idx / len(train_loader),
            loss.item()))


def test(model, device, test_loader, criterion, epoch, iter_meter, experiment):
  print('\nevaluating...')
  model.eval()
  test_loss = 0
  test_cer, test_wer = [], []
  with torch.no_grad():
    for i, _data in enumerate(test_loader):
      spectrograms, labels, input_lengths, label_lengths = _data
      spectrograms, labels = spectrograms.to(device), labels.to(device)

      output = model(spectrograms)  # (batch, time, n_class)
      output = F.log_softmax(output, dim=2)
      output = output.transpose(0, 1)  # (time, batch, n_class)

      loss = criterion(output, labels, input_lengths, label_lengths)
      test_loss += loss.item() / len(test_loader)

      decoded_preds, decoded_targets = GreedyDecoder(
        output.transpose(0, 1), labels, label_lengths)
      for j in range(len(decoded_preds)):
        test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
        test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

  avg_cer = sum(test_cer) / len(test_cer)
  avg_wer = sum(test_wer) / len(test_wer)
  experiment.log_metric('test_loss', test_loss, step=iter_meter.get())
  experiment.log_metric('cer', avg_cer, step=iter_meter.get())
  experiment.log_metric('wer', avg_wer, step=iter_meter.get())

  print(
    'Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(
      test_loss, avg_cer, avg_wer))


def main(learning_rate,batch_size,epochs,train_url,test_url,experiment):
  hparams = {
    "n_cnn_layers": 3,
    "n_rnn_layers": 5,
    "rnn_dim": 512,
    "n_class": 29,
    "n_feats": 128,
    "stride": 2,
    "dropout": 0.1,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs
  }

  experiment.log_parameters(hparams)

  use_cuda = torch.cuda.is_available()
  torch.manual_seed(7)
  device = torch.device("cuda" if use_cuda else "cpu")

  if not os.path.isdir("./data"):
    os.makedirs("./data")

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  train_loader = data.DataLoader(dataset=train_dataset,
                                 batch_size=hparams['batch_size'],
                                 shuffle=True,
                                 collate_fn=lambda x: data_processing(x, 'train', train_audio_transforms,
                                                                      valid_audio_transforms, text_transform),
                                 **kwargs)
  test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'valid', train_audio_transforms,
                                                                     valid_audio_transforms, text_transform),
                                **kwargs)

  model = SpeechRecognitionModel(
    hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'], hparams['n_class'],
    hparams['n_feats'], hparams['stride'], hparams['dropout']).to(device)

  '''model.load_state_dict(torch.load("deepspeech2_1.params"))'''

  print(model)
  print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

  optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
  criterion = nn.CTCLoss(blank=28).to(device)
  scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=hparams['learning_rate'],
    steps_per_epoch=int(len(train_loader)),
    epochs=hparams['epochs'],
    anneal_strategy='linear')

  iter_meter = IterMeter()
  for epoch in range(1, epochs + 1):
    train(
      model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment)
    test(model, device, test_loader, criterion, epoch, iter_meter, experiment)


if __name__ == "__main__":
  print('hi')

comet_api_key = "" # add your api key here
project_name = "speechrecognition"
experiment_name = "speechrecognition-colab"

if comet_api_key:
  experiment = Experiment(api_key=comet_api_key, project_name=project_name, parse_args=False)
  experiment.set_name(experiment_name)
  experiment.display()
else:
  experiment = Experiment(api_key='dummy_key', disabled=True)

learning_rate = 5e-4
batch_size = 10
epochs = 1
libri_train_set = "train-clean-100"
libri_test_set = "test-clean"
"dastorat"
main(learning_rate,batch_size,epochs,libri_train_set,libri_test_set,experiment)
