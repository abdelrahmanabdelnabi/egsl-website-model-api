from .temporal_transforms import TemporalCenterCrop
import numpy as np
import torch

class ResNextTransformer(object):

  def __init__(self, spatial_transform, sample_duration, sample_size):
    self.transform = spatial_transform
    self.sample_duration = sample_duration
    self.temporal_transform = TemporalCenterCrop(self.sample_duration)
    self.sample_size = sample_size

  def __call__(self, frames):
    indices = self.temporal_transform(list(range(len(frames))))
    frames = np.array(frames)[indices]
    frames = self.transform(frames, self.sample_size)
    return torch.stack(frames).view(3, self.sample_duration, self.sample_size, self.sample_size)
