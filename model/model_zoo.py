import torch
import torch.nn as nn
from .resnext import resnet101

class ResnextClassifier():
  def __init__(self, weights_path, use_softmax=True):
    self.n_classes, self.sample_size, self.sample_duration = 100, 112, 64
    model = resnet101(sample_size=self.sample_size, sample_duration=self.sample_duration)
    model.fc = nn.Linear(model.fc.in_features, self.n_classes)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))

    # add a softmax layer
    if use_softmax:
      model = nn.Sequential(model, nn.Softmax(dim=1))
    self.model = model.cpu()

  def predict(self, video):
    if(video.shape[1:] != (3, self.sample_duration, self.sample_size, self.sample_size)):
      raise ValueError('The shape of the sample video should be (batch_size, 3, {0}, {1}, {1}).\
                        The shape received was {2}'.format(self.sample_duration, self.sample_size, video.shape))
    with torch.set_grad_enabled(False):
      return self.model(video.cpu())
