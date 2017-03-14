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
Follow the installation process after what you should be able to activate the cntk-pyXX environment
(where XX is the python version).  

___Note that every time you will want to run experiment with CNTK you will need to activate the cntk-pyXX environment.___

### Getting started

First of all, you need to import all the dependencies :  
```python
from malmopy.agent.qlearner import DQNAgent, TemporalMemory
from malmopy.model.cntk import DeepQNeuralNetwork
from malmopy.environment.gym import GymEnvironment 
```

Secondly, create the environment to run experiment on: 
```python
# OpenAIEnvironment is wrapper around OpenAI Gym environment and provides all methods/properties you need to interact with
# In this example we will use the Breakout-v3 environment.
# monitoring_path is an optional parameter which lets you specify the directory where records will be saved. 
env = GymEnvironment('Breakout-v3', monitoring_path=/directory/where/to/put/records)
```

Now, we are ready to specify the agent and the model that will be used during the experiment:
```python
# First we specify the memory that we will use. Lets use the same than the DeepMind's Nature Paper
# That is to say, a Temporal Memory which provides the four latest frames as input to our network
# We define a 1.000.000 samples memory with each sample of shape 84x84 and the latest 4 stacked frames 

memory = TemporalMemory(1000000, 4, (84, 84), False)
```

```python
#model, here a simple Deep Q Neural Network backed by CNTK runtime
# First parameter is the input shape (4 last frames, 84, 84)
# Second parameter is the number of actions our agent can do (number of neurons on the last layer)
# device_id < 0 indicate "Use CPU", if > 0 it indicates the ID of the GPU to use

model = DeepQNeuralNetwork((4, 84, 84), env.available_actions, device_id=-1)
```

Create our DQN Agent:
```python
# We provide the number of action available, our model and the memory
agent = DQNAgent("DQN Agent", env.available_actions, model, memory)
```

We can now run the experiment inside a loop:
```python
# Remplace range by xrange if running Python 2
for it in range(20000000):
    
    # Assume that an epoch is 250.000 actions done by the agent 
    epoch = int(it / 250000)
    print('Starting epoch %d' % epoch)

    # Reset environment if needed
    if env.done:
        env.reset()

    current_state = env.state

    action = agent.act(it, current_state, True)
    new_state, reward = env.do(action)
    agent.observe(current_state, action, new_state, reward, env.done)

    # Let's the agent warming up with 50.000 actions
    if it >= 50000:
        agent.learn(it)
```
