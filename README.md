Before running the code create a virtualenv
using python2.7 and activate it. Then run:

```pip install -r requirements.txt```

and

```pip install -e .```

Point 4 specifies how to evaluate the trained agent.

1. The structure is organized in the following way:

- ai_challenge/conig/*.txt is supposed to specify all hyperparameters for
the specific experiment. For instance, value_based_config.txt
provides specification to train DQN

- experiments/experiments.py contains an implementation
of specific experiments that can be run

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
For us, it converges to normalized reward of 0.3 - 0.32
in 160k - 200k steps with probability of playing with focused agent
set to 0.75. 

3. You could look in 

```ai_challenge/tasks/pig_chase/environment/extensions.py```

This defines how we build a state from observation received in 
game. Namely method ```build``` in subclass of ```MalmoStateBuilder```
should implement how to preprocess the state. New observation
from environment is updated in property ```environment.world_observations```
and this needs to be transformed to numpy array that will be an
input to the model. You could run simulation as specified in (2)
and print these things to see what they contain.

4. To evaluate trained agent, go to

```ai_challenge/tasks/pig_chase/scripts```

and run 

```python evaluate_trained_agent.py -c eval_config.txt -r 1000```

where the argument after ```-r``` flag reps specifies the number of episodes
to evaluate. This will generate the submission file
 in the same directory. 
Please note that the agent uses ```CustomStateBuilder```
defined in ```ai_challenge/tasks/pig_chase/environment/extensions.py```

To evaluate the agent on simulation, run:

```python main.py -e eval_simulation -c eval_config.txt ```