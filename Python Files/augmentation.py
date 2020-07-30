import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
from datetime import datetime
import progressbar
import os
import shutil


def generate():
    """
    Function to generate augmented versions of each of the training examples
    :return: True when finished
    """

    path = '../Training Data/Training_Input'
    num_files = len(os.listdir(path))

    if not os.path.exists("../Training Data/Training_Input_Augmented"):
        os.makedirs("../Training Data/Training_Input_Augmented")
    if not os.path.exists("../Training Data/Training_Output_Augmented"):
        os.makedirs("../Training Data/Training_Output_Augmented")

    # Sequence of augmentation steps
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

    with progressbar.ProgressBar(max_value=num_files) as bar:
        for i in range(1,num_files+1):
            img = Image.open('../Training Data/Training_Input/input_{0}.png'.format(i)).convert('RGB')  # open image to augment
            img_array = np.array(img) # convert to array
            img.close() # close img to free up memory
            ia.seed(int(datetime.utcnow().timestamp())) # generate random seed
            images = [img_array, img_array, img_array, img_array] # generate 4 augmented images
            images_aug = seq(images=images) # perform augmentation sequence on each of the images
            for k in range(1,len(images_aug)+1): # for each of the augmented images
                img_aug = Image.fromarray(np.uint8(images_aug[k-1]))  # convert to image
                img_aug.save('../Training Data/Training_Input_Augmented/input_{0}.png'.format(num_files + k + i)) # save augmented inout image
                img_aug.close()
                shutil.copy("../Training Data/Training_Output/output_{0}.png".format(i), "../Training Data/Training_Output_Augmented/output_{0}.png".format(num_files + k + i))
                # copy training output for each of the augmented images
            bar.update(i-1)

    return True


