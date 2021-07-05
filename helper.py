# -----------------------------------------------------------
# Single Quantum Dot Simulator that is used to Generate Training Examples for a CNN.
#
# (C) 2020 Joel Pendleton, London, UK
# Released under MIT license
# email joel.pendleton@quantummotion.tech
# -----------------------------------------------------------

import multiprocessing
from simulation import QuantumDot
import os
from tqdm import tqdm
from multiprocessing import Pool

class Helper:
    """
    This is a class for augmention and generation of  traning examples (helper class)

    Attributes:
        seq (object): the sequence of augmentation steps to perform on existing training examples.
    """

    def __init__(self):
        """
       The constructor for Helper class.
       """
        if not os.path.exists("data/train/image"):
            os.makedirs("./data/train/image")
        if not os.path.exists("./data/train/annotation"):
            os.makedirs("./data/train/annotation")

        if not os.path.exists("./data/val/image"):
            os.makedirs("data/val/image")
        if not os.path.exists("./data/val/annotation"):
            os.makedirs("./data/val/annotation")

        self.number_of_examples_created = len(os.listdir('./data/train/image'))
        self.num_processes = int(multiprocessing.cpu_count() * 0.6)  # number of logical processors to utilise

    'Define function to run mutiple processors and pool the results together'

    def simulate(self, i):
        """
        The function to create a simulation and produce a training example using QuantumDot class defined in simulation.py.

        Parameters:
            i (int): the current iteration of the simulations (file name suffix).
        Returns:
            i (int): current iteration of the simulations.
        """
        dot_i = QuantumDot()
        simulate = dot_i.simulate(i, self.noise, self.path)

        while simulate==False: # if there are no annotations produced try simulate again
            dot_i = QuantumDot()
            simulate = dot_i.simulate(i, self.noise, self.path)
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
        #print(noise)
        #  imap: It only support functions with one dynamic argument
        func = self.simulate
        argument_list = list(range(1, number_of_examples + 1))

        # print("Running imap multiprocessing for single-argument functions ...")
        result_list = self.run_imap_multiprocessing(func=func, argument_list=argument_list, num_processes=self.num_processes)
        assert result_list == argument_list
        return True


