## Reinforcement Learning to-do list:

- [X] Solve CartPole with a Single Network
- [X] Solve CartPole with a Single Network using Memory Replay
- [X] Solve CartPole with a Double Network (An Online Model plus a Target Network) to stabilise the algorithm
- [X] Solve Lunar Lander (Discrete)
- [X] Learn how to implement the gradients for an Actor-Critic Model with Tensorflow
- [X] Solve Lunar Lander (Continuous) with Actor-Critic Model
- [X] Code my own environment (SNAKE) and look for proper inputs so that the AI can generalise to different boards
- [X] Solve Pendulum by Changing from Epsilon-greedy exploration strategies to noise-adding strategies
- [X] Try Bipedal Walker with DDPG (Deep Deterministic Policy Gradient). There is not enough exploration and it's difficult to escape from local optima
- [X] Implement a simple Genetic Algorithm capable of finding the minimum of a function (Beale function)
- [X] Use the Genetic Algorithm to solve the Acrobot Environment (also CartPole again)
- [X] Solve Mountain Car with a Genetic Algorithm (this environment is more challenging than it seems)
- [X] Try again Bipedal Walker with the Genetic Algorithm. Gets a better performance although it is not optimal
- [X] Try Augmented Random Search for solving Bipedal Walker. The results are worse than with genetic algorithm

- [ ] \(Optional) Solve an Atari Environment from screen pixels. It would probably take more than a week on a CPU for each hyperparameter trial...

## 1. Cart Pole
Random Agent with no training:

![CartPoleRandomAgent](cartPole/tactics/CartPoleRandom.gif)

Links to scripts:

[Cart Pole Single Network](cartPole/cartPole1SingleNetwork.ipynb), [Cart Pole Single Network with Memory Replay](cartPole/cartPole2WithExperienceReplaySaveBestWeights.ipynb), [Cart Pole Double Network](cartPole/cartPole3DoubleDQN.ipynb)

The agent has two different actions: Moving left (-1) and moving right (1). Cart Pole was found to develop two strategies. The first one involves moving sharply to the opposite side, managing to mantain its position in the middle of the environment:

![CartPoleTactic1](cartPole/tactics/CartPoleTactic1.gif)

The second one is based on subtle changes of direction, which in the long term would result in getting out of the screen. However, the limit of 500 steps per episode helps reduce the evolutive pressure of the environment towards the tactic 1 given that normally the environment ends before the agents goes out of the screen (only around 10% of the times this happens before 500 steps):

![CartPoleTactic2](cartPole/tactics/CartPoleTactic2.gif)

You can test these two tactics by yourself with the following script. You need to install OpenAIGym (https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30?gi=cdb9345d454c). The weights file (available in the cartPole/tactics folder) should be located in the same folder than the following testing script:

[Testing CartPole Weights](cartPole/cartPole0TestingCartPoleWeights.ipynb)

## 2. Lunar Lander
Agent with no training:

![OnlyAction3](lunarLander/tactics/1AgentOnlyAction3.gif)

There are two version of this environment: Discrete and Continuous. Links to scripts:

[Lunar Lander Discrete](lunarLander/lunarLander2Discrete.ipynb), [Lunar Lander Continuous](lunarLander/lunarLander3Continuous.ipynb)

In the discrete one, the agent can choose between doing nothing, firing main engine, firing left engine and firing right engine. In the continuous one, the action space comprises two float values which indicate the power in the main engine (from -1 to 0 means powered off) and the relation between left and right engine power (values close to -1: left, values close to 1: right). The Actor-Critic algorithm was implemented to solve the continuous environment. The actor is a neural network that chooses for each state which output should be executed. The critic takes as input both the state and the action to be executed and grades it, judging how much reward the agent is gonna obtain in the future. Not only the critic network is trained with Q-learning, its gradient is also passed backwards to the actor so the weights are changed in the opposite direction (to maximise instead of minimising the grade to be obtained by the critic)

At first, the agent crashes quickly every time, so it needs to change its behaviour. After around 100 episodes, it learns to hover mid-air so it doesn't feel the -100 points penalisation for crashing:

![Hovering](lunarLander/tactics/2AgentHoversInTheAir.gif)

It's only after hundreds of episodes of exploration that it learns that sometimes, touching the ground it's not negative, instead, it results in a +100 points reward for landing with a proper speed and angle:

![TrainedAgent](lunarLander/tactics/3AgentHasLearnt.gif)

As with other RL environments, these learning algorithms can be unstable when certain past-experiences dissapear from the memory array. When catastrophic learning happens in Lunar Lander, the agent stops remembering that landing with a great speed or a bad angle results in a -100 points penalisation. For this reason, it is advised to do regular checkpoints of the agent weights to recover the best version of the agent obtained during the learning curve.

For testing the trained weights, the weights file (available in the lunarLander/tactics folder) should be located in the same folder than the following testing scripts:

[Testing Lunar Lander Discrete](lunarLander/lunarLander0TestingDiscrete.ipynb), [Testing Lunar Lander Continuous](lunarLander/lunarLander1TestingContinuous.ipynb)


## 3. Coding my Own Environment (SNAKE)

The Snake game was coded from scratch in order to obtain a flexible environment. In this section, different kinds of inputs and outputs for the Neural Network were tried such as passing the board of the game or the pixels obtained after rendering the board with Matplotlib. Both versions of the snake game were tested: with an Open Area in which the snake can go outside the borders and appear in the opposite side of the screen; and with a wall that kills the snake if it tries to go out. Using an Open Area made it more difficult to implement snake sensors and there were more situations in which the snake would fall into a loop in the initial training episodes:

![SnakeLoop](snake/tactics/loopSnake.gif)

After many trials, it was decided to use a closed area and equip the Snake with sensors which would help the AI generalise to smaller and bigger boards. Therefore, the snake can see in eight directions for which it will know the proximity of a wall, an apple or a snake body part (the proximity is given as 1/distance), making a total of 24 input numbers. Furthermore, the snake is given a one-hot vector for the direction in which it is moving plus a smelling sense to know if the apple is South, North, East or West (for all those cases in which there is not a straigth line between the snake and the apple):

![TrainedSnake](snake/tactics/trainedSnake.gif)

Once the snake had been trained in a 9-by-9 board, it was moved to boards with different shapes in which it was tested (without further training or changing any code). The AI proved to have excellent generalisation. This is due to the use of sensors, if the snake had been trained from screen pixels or directly from the board, it would have had to start learning from scratch each time it was moved to a new board shape:

![SmallerBoard](snake/tactics/smallSnake.gif)
![DifferentBoard](snake/tactics/differentBoardSnake.gif)

The snake is not perfect, however; it could be improved in the future by adding extra sensors which would let the snake know how many body parts are found in each direction (so far it can only know how close is the first one, and it often gets confused once its length is around one-third of the total squares of the board).

[Snake Custom Environment](snake/SnakeCustomEnvironmentDefinitive.ipynb), [Testing Snake](snake/TestingSnake.ipynb)

## 4. From Epsilon-Greedy exploration to noise-adding exploration strategies (Pendulum and Bipedal Walker)

Although Lunar-Lander continuous environment was solved using a naive Epsilon-greedy exploration strategy, this is not the preferred way for continuous action spaces. In these environments, Momentum is an important factor for the proper movement of the agent, which cannot be kept if randonmly changing actions at every time step. Two kinds of noise are commonly used: Gaussian Noise and Ornstein–Uhlenbeck noise. Both of these noise functions are stationary (its distribution doesn't change over time), Markovian (the noise only depends on the current state of the environment) and Stocastic while maintaining a Normal Distribution.

[Solving Pendulum](pendulum/Pendulum.ipynb), [Testing Pendulum](pendulum/TestingPendulum.ipynb)

The Pendulum environment was easily solved using Gaussian Noise (random agent on the left, trained agent on the right):

![RandomPendulum](pendulum/tactics/randomPendulum.gif)
![TrainedPendulum](pendulum/tactics/learnedPendulum.gif)

The Bipedal Walker, however; is a much more difficult environment. Not only does the state contain 24 different values, but also, the action space is quite complex with 4 different continuous actions which represent the force to apply at each agent torque. The greatest challenge in this enviroment is to escape from local optima. The agent learns really fast to remain static on the floor in order not to fall to the ground and lose points. If the reward function is changed in order to penalise maintaining the agent static on the floor, the agent will now learn how to fall as fast as quickly so it doesn't get penalised by applying forces before ending the episode (the reward function penalises applying forces in order to obtain an efficient robot that walks with little energy).

![RandomBipedal1](bipedalWalker/tactics/randomBipedal1.gif)
![RandomBipedal2](bipedalWalker/tactics/randomBipedal2.gif)
![RandomBipedal3](bipedalWalker/tactics/randomBipedal3.gif)

After many trials, a different approach was tried, Genetic Algorithms.

## 5. Genetic Algorithms. From a simple function to Bipedal Walker optimization.

Genetic algorithms are an important branch of optimization algorithms, based on how Natural Selection works. Agents, which are initially random, need to be evaluated with custom fitness functions (for example, the reward they get on the environment) in order to select the fittest ones. As with evolution, only the fittest ones will "reproduce" by obtaining other "children" agents whose parameters are intermediate between those of both "parents". These new agents will undergo some random mutations to improve the "genetic" variability of the Neural Network weights. After enough generations of agents, the Reinforcement Learning environments can be solved with quite impressive scores, however; the number of generations, and number of agents per generation quickly scales the computation cost of these algorithms.

[Finding the minimum of a function with a Genetic Algorithm](bealeFunction/GeneticBealeFunction.ipynb)

This algorithm was able to find the minimum of the Beale function in seconds (only two parameters: X and Y). Afterwards, CartPole, was also solved in around 10 minutes. This is quite impressive when compared to the experiments with DQNs which would take hours to finish. Furthermore, when using a genetic algorithm, catastrophic forgetting is usually avoided as the Elitism Selection gets rid of those agents whose mutations result in lower scores.

[Genetic CartPole](cartPole/cartPole4GeneticAlgorithm.ipynb), [Genetic Acrobot](acrobot/GeneticAcrobot.ipynb), [Testing Acrobot](acrobot/TestingAcrobot.ipynb)

Following that, a new environment was solved: Acrobot. In this environment, the agent has to manage to get some part of its body above the line as fast as possible (random agent on the left, trained agent on the right):

![RandomAcrobot](acrobot/tactics/randomAcrobot.gif)
![TrainedAcrobot](acrobot/tactics/learnedAcrobot.gif)

The Mountain Car environment seems to be a rather simple environment, however; its reward function can prevent many great Reinforcement Learning algorithms from solving it. The Mountain Car reward function gives -1 each time step and 0 when the car reaches the goal. This means that the car has to randonmly go back and forth in the mountain till it gets to the top of the mountain (which is highly unlikely) just to learn that the agent is not always going to get a -1 reward. Other alternative reward functions such as measuring how close the car gets to the goal would fail given that the car has to learn to go left (far from the goal) in order to build momentum.

[Genetic Mountain Car](mountainCar/GeneticMountainCar.ipynb), [Testing Mountain Car](mountainCar/TestingMountainCar.ipynb)

Since almost no random agent can possibly obtain a great reward at the end of the episode (all of them would get -200 after 200 time steps), reward function engineering was carried out. The fitness function considers both the maximum velocity reached for each car and the closest they had been to the goal during the episode. After some generations, an optimal agent is obtained (even though it is not optimal with regard to the official fitness function, since we are not considering how many step it takes to reach the goal as long as it finishes before 200 steps). Random agent on the left, trained agent on the right:

![RandomCar](mountainCar/tactics/randomCar.gif)
![TrainedCar](mountainCar/tactics/learnedCar.gif)

Finally, the genetic algorithm was implemented in the Bipedal Walker environment. A custom reward function was coded in order to keep throughout evolution those agents which reached the farthest during the assigned time. In order to make natural selection faster, the environment ended after only 500 steps considering that the fastest agent would have enough time to make its advantage worth. The results are promissing even though this is a really computationally-expensive experiment. Ideally, this algorithm should be run for more generations, with more agents per generation and taking into account the performance across different episodes (given that the random surface in each episode affects the speed of the agent). Here are shown different techniques developed through the evolution of random agents with selection and crossover/mutation of the fittest ones:

[Genetic Bipedal Walker](bipedalWalker/GeneticBipedalWalker.ipynb), [Testing Bipedal Walker](bipedalWalker/TestingBipedalWalker.ipynb)

![BipedalWalker1](bipedalWalker/tactics/tactic1.gif)
![BipedalWalker2](bipedalWalker/tactics/tactic2.gif)
![BipedalWalker3](bipedalWalker/tactics/tactic3.gif)

Augmented Random Search is an algorithm which explores by adding random noise to the weights of the agent (the noise we had described in a previous section had been added to the actions instead). Applying the same perturbation in both directions (positive and negative), a derivative with high variance can be computed with respect to the parameters of the agent. This technique was tried on Bipedal Walker, however; it obtained worse results than with Genetic Algorithms (results were still better than with Actor-Critic nonetheless). This is due to the surface variability in each episode, altogether with the need of obtaining a good first random agent to train (if it gets static on the floor, a small perturbation in the weights is not likely going to change its behaviour) and a reward function that doesn't encourage the agent to explore (penalising the forces applied). Using a custom reward function didn't help either, since the agent would slowly learn to jump and fall as far as possible as a consequence of these small derivatives.

## 6. Extra environments.

Some additional environments might be added in this section. However, they will not have fully-trained agents given that most of these complex challenges would mean days or even weeks running in order to try each hyperparameter configuration on a CPU.

Pong (from ram). The input consists of 128 bytes and the output is discrete: Up or Down.

[Genetic Algorithm Pong](pong/Pong.ipynb)

Random agent on the left, agent selected by Genetic Algorithm on the right (computer:orange, agent:green):

![randomPong](pong/tactics/randomPong.gif)
![learnedPong](pong/tactics/learnedPong.gif)
