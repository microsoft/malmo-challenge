## Writing your first experiment

The framework is designed to give you the flexibility you need to design and run your experiment. 
In this section you will see how easy it is to write a simple Atari/DQN experiment based on CNTK backend.


### Using with Microsoft Cognitive Network ToolKit (CNTK)
To be able use CNTK from the framework, you will need first to install CNTK from the 
official repository [release page](https://github.com/Microsoft/CNTK/releases). Pick the 
right distribution according to your OS / Hardware configuration and plans to use distributed
training sessions.

The CNTK Python binding can be installed by running the installation script 
([more information here](https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-your-machine)).
After following the installation process you should be able to import CNTK.

___Note that every time you will want to run experiment with CNTK you will need to activate the cntk-pyXX environment.___

### Getting started

First of all, you need to import all the dependencies :  
```python
from malmopy.agent.qlearner import QLearnerAgent, TemporalMemory
from malmopy.model.cntk import QNeuralNetwork
from malmopy.environment.gym import GymEnvironment 
 
# In this example we will use the Breakout-v3 environment.
env = GymEnvironment('Breakout-v3', monitoring_path='/directory/where/to/put/records')
 
# Q Neural Network needs a Replay Memory to randomly sample minibatch.
memory = TemporalMemory(1000000, (84, 84), 4)
 
#Here a simple Deep Q Neural Network backed by CNTK runtime
model = QNeuralNetwork((4, 84, 84), env.available_actions, device_id=-1)
 
# We provide the number of action available, our model and the memory
agent = QLearnerAgent("DQN Agent", env.available_actions, model, memory, 0.99, 32)
  
reward = 0
done = False
  
# Remplace range by xrange if running Python 2
while True:

    # Reset environment if needed
    if env.done:
        current_state = env.reset()

    action = agent.act(current_state, reward, done, True)    
    new_state, reward, done = env.do(action)
```

## Some comments:
- The GymEnvironment monitoring_path is used to record short epsiode videos of the agent
- Temporal Memory generates a sample w.r.t to the history_length previous state
  - For example with history_length = 4 a sample is [s(t-3), s(t-2), s(t-1), s(t)]
- QNeuralNetwork input_shape is the shape of a sample from the TemporalMemory (history_length, width, height)
- QNeuralNetwork output_shape is the number of actions available for the environment (one neuron per action)
- QNeuralNetwork device_id == -1 indicate 'Run on CPU', anything >=0 refers to a GPU device ID
