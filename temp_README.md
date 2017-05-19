1. The structure is organized in the following way:

- conig/*.txt is supposed to specify all hyperparameters for
the specific experiment. For instance, value_based_config.txt
provides specification to train DQN

- experiments/experiments.py contains an implementation
of specific experiments that are to be run

- environment/* has wrappers to deal with 2 players and
provide interface that can be used by libraries and simulator 
that can be used to speed up training

- models/networks.py has various neural nets

- tasks/ has a specific Malmo tasks (only pig_chase for now)


To run the code with Malmo please follow description in th original
README.md

2. To check simulation, you can run:

```python main.py -e simulation -c value_based_config.txt```

What would happen then is:

experiments.py/simulation:

- creates learner specified in config and defined in experiments.py/create_value_based_learner

- creates env specified in config and defined in env_simulator.py

- runs experiment with ```chainerrl``` utils

This should generate results folder with results.
For me, it converged to normalized reward of 0.4 - 0.42
in 160k - 200k steps with p_focused set to 1. 

3. You could look in 

ai_challenge/tasks/pig_chase/environment/extensions.py

This defines how we build a state from observation received in 
game. Namely method ```build``` in subclass of ```MalmoStateBuilder```
should implement how to preprocess the state. New observation
from environment is updated in property ```environment.world_observations```
and this needs to be transformed to numpy array that will be an
input to the model. You could run simulation as specified in (2)
and print these things to see what they contain.

The code could be quite messy at the moment, you will know much 
better how to structure this properly.