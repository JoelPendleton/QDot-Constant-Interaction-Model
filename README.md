# Single Quantum Dot Simulation using the Constant Interaction Model 
Software to simulate single quantum dot systems, using the constant interaction model. Theory of model is described in [Spins in few-electron quantum dots
](https://arxiv.org/pdf/cond-mat/0610433.pdf)

**simulation.py** generates coulomb diamonds for 1000 randomised single quantum dot systems, and it also outputs the respective edges of each diamond. 
This code was developed to generate training data for an edge detector, which could be further utilised to perform automated read-outs of the paramters of coulomb diamonds.


**augmentation.py** generates 4 augmented versions for each of the training examples generated by simulation.py. This gives more varied examples for training a ML edge detection algorithm.

To generate training examples run the **main.py** script in the command line using one of the following arguments / flags:
* **--simulate** generates training examples.
* **--augment** generates augmented versions of existing training examples.
* **--both** generates training examples and augmented versions of the training examples.
for the **--simulate** and **--both** arguments you need to also pass another argument, the number of training examples you wish to generate.

E.g. 
```python main.py --simulate 100```
This generates 100 training examples.

**Example of training example generated:**

<table>
   <tbody>
      <tr>
       <td>Input</td>
       <td>Output</td>
     </tr> 
     <tr>
       <td><img src="https://gitlab.com/QSD/dot-analysis-hub/QDot-Constant-Interaction-Model/-/raw/master/simulation_example_input.png" width="500"></td>
       <td><img src="https://gitlab.com/QSD/dot-analysis-hub/QDot-Constant-Interaction-Model/-/raw/master/simulation_example_output.png" width="500"></td>
     </tr > 
  </tbody>
</table>


