# -----------------------------------------------------------
# Single Quantum Dot Simulator that is used to Generate Training Examples for a CNN.
#
# (C) 2020 Joel Pendleton, London, UK
# Released under MIT license
# email joel.pendleton@quantummotion.tech
# -----------------------------------------------------------

import multiprocessing
from simulation import QuantumDot
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from datetime import datetime
from PIL import Image
import shutil
import os
from tqdm import tqdm
from multiprocessing import Pool

class Helper:
    """
    This is a class for augmention and generation of  traning examples (helper class)

    Attributes:
        seq (object): the sequence of augmentation steps to perform on existing training examples.
    """

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
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.4), per_channel=0.2),

        # Improve or worsen the contrast of images.
        iaa.LinearContrast((0.8, 2), per_channel=0.5)

    ], random_order=True)  # apply augmenters in random order

    def __init__(self):
        """
       The constructor for Helper class.
       """
        if not os.path.exists("../Training Data/Training_Input_Augmented"):
            os.makedirs("../Training Data/Training_Input_Augmented")
        if not os.path.exists("../Training Data/Training_Output_Augmented"):
            os.makedirs("../Training Data/Training_Output_Augmented")
        if not os.path.exists("../Training Data/Training_Input"):
            os.makedirs("../Training Data/Training_Input")
        if not os.path.exists("../Training Data/Training_Output"):
            os.makedirs("../Training Data/Training_Output")

        self.number_of_examples_created = len(os.listdir('../Training Data/Training_Input'))
        self.num_processes = int(multiprocessing.cpu_count() * 0.8)  # number of logical processors to utilise


    '''Define function to run mutiple processors and pool the results together'''

    def simulate(self, i):
        """
        The function to create a simulation and produce a training example using QuantumDot class defined in simulation.py.

        Parameters:
            i (int): the current iteration of the simulations (file name suffix).
        Returns:
            i (int): current iteration of the simulations.
        """
        dot_i = QuantumDot()
        dot_i.simulate(i)
        return i

    def run_imap_multiprocessing(self, func, argument_list, num_processes):
        """
        The function to create a multiprocessing instance, and simulate examples using multithreading.

        Parameters:
            func: the function to perform multithreading on.
            argument_list (int): a list of integers representing the different integers to iterate over.
            num_processes (int): the number of logical processors to be utilised.
        Returns:
            result_list_tqdm
        """
        pool = Pool(processes=num_processes)

        result_list_tqdm = []

        for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
            result_list_tqdm.append(result)

        return result_list_tqdm

    def generate_examples(self, number_of_examples):
        """
        The function to generate simulation training examples.

        Parameters:
            number_of_examples (int): the number of training examples to generate
        Returns:
            True upon completion
        """

        #  imap: It only support functions with one dynamic argument
        func = self.simulate
        argument_list = list(range(1, number_of_examples + 1))

        # print("Running imap multiprocessing for single-argument functions ...")
        result_list = self.run_imap_multiprocessing(func=func, argument_list=argument_list, num_processes=self.num_processes)
        assert result_list == argument_list
        return True


    def augment_examples(self):
        """
         The function to augment simulation training examples.

         Returns:
             True upon completion
         """

        #  imap: It only support functions with one dynamic argument
        func = self.augment
        self.number_of_examples_created = len(os.listdir('../Training Data/Training_Input'))
        argument_list = list(range(1, self.number_of_examples_created + 1))

        result_list = self.run_imap_multiprocessing(func=func, argument_list=argument_list, num_processes=self.num_processes)

        assert result_list == argument_list
        return True

    def augment(self, i):
        """
         The function to augment a single simulation training example.

         Parameters:
            i (int): the number of the current iteration
         Returns:
            i (int): the number of the current iteration
         """

        img = Image.open('../Training Data/Training_Input/input_{0}.png'.format(i)).convert(
            'RGB')  # open image to augment
        img_array = np.array(img)  # convert to array
        img.close()  # close img to free up memory
        ia.seed(int(datetime.utcnow().timestamp()))  # generate random seed
        images = [img_array, img_array, img_array, img_array]  # generate 4 augmented images
        images_aug = Helper.seq(images=images)  # perform augmentation sequence on each of the images
        counter = 4
        for k in range(1, len(images_aug) + 1):  # for each of the augmented images
            current_image_number = self.number_of_examples_created + k + counter * (i - 1)  # current augmented image number to be saved
            img_aug = Image.fromarray(np.uint8(images_aug[k - 1]))  # convert to image
            img_aug.save('../Training Data/Training_Input_Augmented/input_{0}.png'.format(
                current_image_number))  # save augmented inout image
            img_aug.close()
            shutil.copy("../Training Data/Training_Output/output_{0}.png".format(i),
                        "../Training Data/Training_Output_Augmented/output_{0}.png".format(current_image_number))
            # copy training output for each of the augmented images

        return i
