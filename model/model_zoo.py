import torch
import torch.nn as nn
from .resnext import resnet101
import re

class ResnextClassifier():
  def __init__(
    self,
    weights_path,
    use_softmax=True,
    n_classes=100,
    sample_size=112,
    sample_duration=64
  ):
    self.weights_path = weights_path
    self.n_classes, self.sample_size, self.sample_duration = n_classes, sample_size, sample_duration
    model = resnet101(sample_size=self.sample_size, sample_duration=self.sample_duration)
    model.fc = nn.Linear(model.fc.in_features, self.n_classes)
    data = torch.load(self.weights_path, map_location='cpu')
    renamed_data = dict()
    for key in data:
        renamed_key = re.sub('^0\.', '', key)
        renamed_data[renamed_key] = data[key]
    model.load_state_dict(renamed_data)
    # add a softmax layer
    if use_softmax:
      model = nn.Sequential(model, nn.Softmax(dim=1))
    model.eval()
    self.model = model.cpu()

  def predict(self, video):
    if(video.shape[1:] != (3, self.sample_duration, self.sample_size, self.sample_size)):
      raise ValueError('The shape of the sample video should be (batch_size, 3, {0}, {1}, {1}).\
                        The shape received was {2}'.format(self.sample_duration, self.sample_size, video.shape))
    with torch.set_grad_enabled(False):
      return self.model(video.cpu())
