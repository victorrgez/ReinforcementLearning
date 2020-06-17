## Reinforcement Learning to-do list:

- [X] Solve CartPole with a Single Network
- [X] Solve CartPole with a Single Network using Memory Replay
- [X] Solve CartPole with a Double Network (An Online Model plus a Target Network) to stabilise the algorithm
- [ ] Solve Lunar Lander with discrete Action Space
- [ ] Solve Lunar Lander with continuous Action Space
- [ ] Solve Bipedal Walker
- [ ] Solved Bipedal Waker Hardcore
- [ ] Implement Action Critic Algorithm ?
- [ ] \(Optional) Solve Atari Breakout. It would probably take more than a week computing each training loop...

## 1. Cart Pole
Random Agent with no training:

![CartPoleRandomAgent](cartPole/tactics/CartPoleRandom.gif)

Links to scripts:

[Single Network](cartPole/cartPole1SingleNetwork.ipynb), [Single Network with Memory Replay](cartPole/cartPole2WithExperienceReplaySaveBestWeights.ipynb), [Double Network](cartPole/cartPole3DoubleDQN.ipynb)

Cart Pole was found to develop two strategies. The first one involves moving sharply to the opposite side, managing to mantain its position in the middle of the environment:

![CartPoleTactic1](cartPole/tactics/CartPoleTactic1.gif)

The second one is based on subtle changes of direction, which in the long term would result in getting out of the screen. However, the limit of 500 steps per episode helps reduce the evolutive pression of the environment towards the tactic 1 given that normally the environment ends before the agents goes out of the screen (only around 10% of the times this happens before 500 steps):

![CartPoleTactic2](cartPole/tactics/CartPoleTactic2.gif)


## 2. Lunar Lander
TO BE ADDED....
