import numpy as np
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
from datetime import datetime

import os, os.path

path = 'Training_Input'
num_files = len(os.listdir(path))

seq = iaa.Sequential([

    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 70% of all images.
    iaa.Sometimes(
        0.7,
        iaa.OneOf([
            iaa.GaussianBlur((8, 8)),
            iaa.AverageBlur(k=(6, 13)),
            iaa.MedianBlur(k=(5, 13)),
        ])
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.5, 1.3)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.4), per_channel=0.2),

    # Improve or worsen the contrast of images.
    iaa.LinearContrast((0.8, 2), per_channel=0.5)


], random_order=True) # apply augmenters in random order

for i in range(num_files-1):
    img = Image.open('Training_Input/input_{0}_1.png'.format(i+1)).convert('RGB')
    img_array = np.array(img)
    ia.seed(int(datetime.utcnow().timestamp()))
    images = [img_array, img_array, img_array, img_array]
    images_aug = seq(images=images)
    for k in range(len(images_aug)):
        img_aug = Image.fromarray(np.uint8(images_aug[k]))
        img_aug.save('Training_Input_Augmented/input_{0}_{1}.png'.format(i+1, k+2))
