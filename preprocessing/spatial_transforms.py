from torchvision import transforms
import numpy as np
import random

input_size = 112
sample_duration = 64

def validation_spatial_transform(frames, input_size=112):
    transform = transforms.Compose([
        transforms.CenterCrop(input_size*4),
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])
    return [transform(frame.to_image()) for frame in frames]

def training_spatial_transform(frames, input_size):
    """
    Applies one random transfomation to all of the frames
    of the same video.
    """
    translation = np.random.uniform(0, 0.07)
    crop_size = input_size*random.randrange(3, 7)
    # color jitter params
    b = np.random.uniform(0.7, 1.3)
    h = np.random.uniform(-0.2, 0.2)
    c = np.random.uniform(0.9, 1.1)
    s = np.random.uniform(0.8, 1.2)

    def transform(img):
        transforms.functional.affine(
            img,
            angle=0,
            translate=(translation, translation),
            scale=1,
            shear=0,
        )
        img = transforms.CenterCrop(crop_size)(img)
        img = transforms.functional.resize(img, input_size)
        img = transforms.functional.adjust_brightness(img, b)
        img = transforms.functional.adjust_contrast(img, c)
        img = transforms.functional.adjust_hue(img, h)
        img = transforms.functional.adjust_saturation(img, s)
        tensor = transforms.functional.to_tensor(img)
        return tensor

    return [transform(frame.to_image()) for frame in frames]
