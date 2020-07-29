# Quantum Dot Constant Interaction Model Simulation
Software to simulate single quantum dot systems, using the constant interaction model. Theory of model is described in [Spins in few-electron quantum dots
](https://arxiv.org/pdf/cond-mat/0610433.pdf).

**simulation.py** generates coulomb diamonds for 1000 randomised single quantum dot systems, and it also outputs the respective edges of each diamond. 
This code was developed to generate training data for an edge detector, which could be further utilised to perform automated read-outs of the paramters of coulomb diamonds.


**augmentation.py** generates 4 augmented versions for each of the training examples generated by simulation.py. This gives more varied examples for training a ML edge detection algorithm.

Example of simulation output:

![myImage](https://github.com/JoelPendleton/QDot-Constant-Interaction-Model/blob/master/simulation_example.png)
