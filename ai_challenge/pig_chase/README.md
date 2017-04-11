# Malmo Collaborative AI Challenge - Pig Chase

This repository contains Malmo Collaborative AI challenge task definition. The challenge task takes the form of a collaborative mini game, called Pig Chase.

![Screenshot of the pig chase game](pig-chase-overview.png?raw=true "Screenshot of the Pig Chase game")

## Overview of the game

Two Minecraft agents and a pig are wandering a small meadow. The agents have two choices:

- _Catch the pig_ (i.e., the agents pinch or corner the pig, and no escape path is available), and receive a high reward (25 points)
- _Give up_ and leave the pig pen through the exits to the left and right of the pen, marked by blue squares, and receive a small reward (5 points)

The pig chased is inspired by the variant of the _stag hunt_ presented in [Yoshida et al. 2008]. The [stag hunt](https://en.wikipedia.org/wiki/Stag_hunt) is a classical game theoretic game formulation that captures conflicts between collaboration and individual safety.

[Yoshida et al. 2008] Yoshida, Wako, Ray J. Dolan, and Karl J. Friston. ["Game theory of mind."](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000254) PLoS Comput Biol 4.12 (2008): e1000254.


## How to play (human players)

To familiarize yourself with the game, we recommend that you play it yourself. The following instructions allow you to play the game with a "focused agent". A baseline agent that tries to move towards the pig whenever possible.

### Prerequisites

* Install the [Malmo Platform](https://github.com/Microsoft/malmo) and the `malmopy` framework as described under [Installation](../../README.md#installation), and verify that you can run the Malmo platform and python example agents

### Steps

* Start two instances of the Malmo Client on ports `10000` and `10001`
* `cd malmo-challenge/ai_challenge/pig_chase`
* `python pig_chase_human_vs_agent.py`

Wait for a few seconds for the human player interface to appear.

Note: the script assumes that two Malmo clients are running on the default ports on localhost. You can specify alternative clients on the command line. See the script's usage instructions (`python pig_chase_human_vs_agent.py -h`) for details.

### How to play

* The game is played over 10 rounds at a time. Goal is to accumulate the highest score over these 10 rounds.
* In each round a "collaborator" agent is selected to play with you. Different collaborators may have different behaviors.
* Once the game has started, use the left/right arrow keys to turn, and the forward/backward keys to move. You can see your agent move in the first person view, and shown as a red arrow in the top-down rendering on the left.
* You and your collaborator move in turns and try to catch the pig (25 points if caught). You can give up on catching the pig in the current round by moving to the blue "exit squares" (5 points). You have a maximum of 25 steps available, and will get -1 point for each step taken.

## Run your first experiment

An example experiment is provided in `pig_chase_baseline.py`. To run it, start two instances of the Malmo Client as [above](#steps). Then run:

```
python pig_chase_baseline.py
```

Depending on whether `tensorboard` is available on your system, this script will output performance statistics to either tensorboard or to console. If using tensorboard, you can plot the stored data by pointing a tensorboard instance to the results folder:

```
cd ai_challenge/pig_chase
tensorboard --logdir=results --port=6006
```

You can then navigate to http://127.0.0.1:6006 to view the results.

The baseline script runs a `FocusedAgent` by default - it uses a simple planning algorithm to find a shortest path to the pig. You can also run a `RandomAgent` baseline. Switch agents using the command line arguments:

```
python pig_chase_baseline.py -t random
```

For additional command line options, see the usage instructions: `python pig_chase_baseline.py -h`.

## Evaluate your agent

We provide a commodity evaluator PigChaseEvaluator, which allows you to quickly evaluate
performances of your agent.

PigChaseEvaluator takes 2 arguments:
- agent_100k : Your agent trained with 100k steps (100k train calls) 
- agent_500k : Your agent trained with 500k steps (500k train calls)

To evaluate your agent:

``` python
# Creates an agent trained with 100k train calls
my_agent_100k = MyCustomAgent()

# Creates an agent trained with 500k train calls
my_agent_500k = MyCustomAgent()

# You can pass a custom StateBuilder for your agent.
# It will be used by the environment to generate state for your agent
eval = PigChaseEvaluator(my_agent_100k, my_agent_500k, MyStateBuilder())

# Run and save
eval.run()
eval.save('My experiment 1', 'path/to/save.json')
```


## Next steps

To participate in the Collaborative AI Challenge, implement and train an agent that can effectively collaborate with any collaborator. Your agent can use either the first-person visual view, or the symbolic view (as demonstrated in the `FocusedAgent`). You can use any AI/learning approach you like - originality of the chose approach is part of the criteria for the challenge prizes. Can you come up with an agent learns to outperform the A-star baseline agent? Can an agent learn to play with a copy of itself? Can it outperform your own (human) score?

For more inspiration, you can look at more [code samples](../../samples/README.md) or learn how to [run experiments on Azure using docker](../../docker/README.md).



