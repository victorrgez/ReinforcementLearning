## Reinforcement Learning to-do list:

- [X] Solve CartPole with a Single Network
- [X] Solve CartPole with a Single Network using Memory Replay
- [X] Solve CartPole with a Double Network (An Online Model plus a Target Network) to stabilise the algorithm
- [X] Solve Lunar Lander (Discrete)
- [X] Learn how to implement the gradients for an Actor-Critic Model with Tensorflow
- [X] Solve Lunar Lander (Continuous) with Actor-Critic Model
- [ ] Solve Bipedal Walker with DDPG (Deep Deterministic Policy Gradient)
- [ ] \(Optional) Solve Bipedal Walker Hardcore (fine-tuning the weights from normal Bipedal Walker)
- [ ] \(Optional) Solve Atari Breakout. It would probably take more than a week computing each training loop...

## 1. Cart Pole
Random Agent with no training:

![CartPoleRandomAgent](cartPole/tactics/CartPoleRandom.gif)

Links to scripts:

[Single Network](cartPole/cartPole1SingleNetwork.ipynb), [Single Network with Memory Replay](cartPole/cartPole2WithExperienceReplaySaveBestWeights.ipynb), [Double Network](cartPole/cartPole3DoubleDQN.ipynb)

The agent has two different actions: Moving left (-1) and moving right (1). Cart Pole was found to develop two strategies. The first one involves moving sharply to the opposite side, managing to mantain its position in the middle of the environment:

![CartPoleTactic1](cartPole/tactics/CartPoleTactic1.gif)

The second one is based on subtle changes of direction, which in the long term would result in getting out of the screen. However, the limit of 500 steps per episode helps reduce the evolutive pression of the environment towards the tactic 1 given that normally the environment ends before the agents goes out of the screen (only around 10% of the times this happens before 500 steps):

![CartPoleTactic2](cartPole/tactics/CartPoleTactic2.gif)

You can test these two tactics by yourself with the following script. You need to have installed OpenAIGym (https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30?gi=cdb9345d454c). The weights file (available in the cartPole folder) should be located in the same folder than the following testing script:

[Testing CartPole Weights](cartPole/cartPole0TestingCartPoleWeights.ipynb)

## 2. Lunar Lander
Agent with no training:

![OnlyAction3](lunarLander/tactics/1AgentOnlyAction3.gif)

There are two version of this environment: Discrete and Continuous. Links to scripts:

In the discrete one, the agent can choose between doing nothing, firing main engine, firing left engine and firing right engine. In the continuous one, the action space comprises two float values which indicate the power in the main engine (from -1 to 0 means powered off) and the relation between left and right engine power (values close to -1: left, values close to 1: right).

At first, the agent crashes quickly every time, so it needs to change its behaviour. After around 100 episodes, it learns to hover mid-air so it doesn't feel the -100 points penalisation for crashing:

![Hovering](lunarLander/tactics/2AgentHoversInTheAir.gif)

It's only after hundreds of episodes of exploration that it learns that sometimes, touching the ground it's not negative, instead, it results in a +100 points reward for landing with a proper speed and angle:

![TrainedAgent](lunarLander/tactics/3AgentHasLearnt.gif)

As with other RL environments, these learning algorithms can be unstable when certain past-experiences dissapear from the memory array. When catastrophic learning happens in Lunar Lander, the agent stops remembering that landing with a great speed or a bad angle results in a -100 points penalisation. For this reason, it is adviced to do regular checkpoints of the agent weights to recover the best version of the agent obtained during the learning curve.

## 3. Bipedal Walker

... (Working on it)
