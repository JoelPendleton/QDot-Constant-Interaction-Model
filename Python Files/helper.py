# -----------------------------------------------------------
# Single Quantum Dot Simulator that is used to Generate Training Examples for a CNN.
#
# (C) 2020 Joel Pendleton, London, UK
# Released under MIT license
# email joel.pendleton@quantummotion.tech
# -----------------------------------------------------------

import multiprocessing
#from multiprocessing import Pool
import time
from simulation import QuantumDot
from multiprocessing import Pool
from multiprocessing import freeze_support
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from datetime import datetime
from PIL import Image
import progressbar
import shutil
import os



'''Define function to run mutiple processors and pool the results together'''
def run_multiprocessing(func, number_of_examples, n_processors):
    with Pool(processes=n_processors) as pool:
        return pool.map(func, number_of_examples)

def task(k):

    current_simulaton = QuantumDot()
    current_simulaton.simulate(k)

def generate(number_of_examples):
    with progressbar.ProgressBar(max_value=number_of_examples) as bar:

        'set up parameters required by the task'
        n_processors = 4
        number_of_examples = list(range(1, number_of_examples))

        '''
        pass the task function, followed by the parameters to processors
        '''
        run_multiprocessing(task, number_of_examples, n_processors)

