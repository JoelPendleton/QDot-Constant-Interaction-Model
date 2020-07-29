# Single Quantum Dot Simulation using the Constant Interaction Model 
Software to simulate single quantum dot systems, using the constant interaction model. Theory of model is described in [Spins in few-electron quantum dots
](https://arxiv.org/pdf/cond-mat/0610433.pdf)

**simulation.py** generates coulomb diamonds for 1000 randomised single quantum dot systems, and it also outputs the respective edges of each diamond. 
This code was developed to generate training data for an edge detector, which could be further utilised to perform automated read-outs of the paramters of coulomb diamonds.


**augmentation.py** generates 4 augmented versions for each of the training examples generated by simulation.py. This gives more varied examples for training a ML edge detection algorithm.

To generate training examples run the **generate_examples.py** script in the command line using one of the following arguments / flags:
* **-t** generates training examples.
* **-a** generates augmented versions of existing training examples.
* **-b** generates training examples and augmented versions of the training examples.
for the **-t** and **-b** arguments you need to also pass another argument, the number of training examples you wish to generate.
E.g. 
```python generate_examples.py -t 100```
This generates 100 training examples.

Example of simulation output:

<img src="https://github.com/JoelPendleton/QDot-Constant-Interaction-Model/blob/master/simulation_example.png " width="500">

