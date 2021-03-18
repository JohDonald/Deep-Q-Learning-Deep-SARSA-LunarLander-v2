# Deep SARSA and Deep Q-Learning - LunarLander-v2

### Environment
In this project I seek to solve the Lunar Lander environment from the OpenAI gym library. This is a 2 dimensional environment where the aim is to teach a Lunar Module to land safely on a landing pad which is fixed at point (0,0). The agent has 3 thrusters: one on the bottom and one on each side of the module. Thus at each time step the agent has 4 possible actions to chose from: Fire each of the thrusters or do nothing. The rewards given to the agent are dependent on a number of factors: Firing the bottom thruster incurs a reward of -0.3, whilst firing the side thrusters incurs a reward of -0.03. If the agent lands safely on the landing pad a reward of +100 points is given, furthermore each leg of the module making contact with the ground is rewarded +10 points. A terminal state has been reached when the agent either lands or crashes. In order to detect a terminal state one can extract a state vector which indicates the position of the agent, its current velocities and environment landing flags to indicate if the legs have contact with the ground. One may also extract an RGB array representing a picture of the agent in the environment. Finally, to solve this problem and to determine whether the algorithm applied to the problem has converged, the agent should average 200 points across multiple episodes.

### Deep SARSA Network
Deep SARSA combines the SARSA on-policy reinforcement learning algorithm with deep learning in order to estimate state action values and build an optimal policy for a given agent. The Q network is initiated/state action values are estimated by the neural network and the SARSA algorithm coupled with experienced based replay then updates the network values.

For the Deep SARSA algorithm we initiate the Q network using a multi-layer neural network to estimate the state-action values. Having built the agent and the replay buffer class, the SARSA update code is then run. 

For each episode the agent begins its course taking steps in accordance with the epsilon-greedy policy, storing each transition to the replay buffer for each step until the environment is terminal. 

At the end of each episode 128 updates are performed, in which for each update a minibatch of 64 observed transitions are randomly sampled from the replay buffer. These values are then used to calculate Q(s,a) target values. The target values are then passed through the loss function, MSE, where we then use the Adam Optimizer to perform a gradient descent to optimize the weights of the neural network by minimizing the loss of the Q(s,a) values and the target Q(s,a) values.

The algorithm used in this report takes inspiration from the pseudo code of Zhao, Wang, Shao and Zhu (2016), however bears some differences particularly in the transition storing and the loss function update. 

Rather than just storing the state, action, reward and new state to be sampled from the replay buffer at each timestep of the episode, we generate and include the next action to be taken in the new state following the e-greedy policy and include this in the transition stored: i.e. (s,a,r,s’,a’). Zhao, et al. (2016) instead generate the new action associated with the new state after it has sampled the transitions from the replay buffer whilst sampling at each time step, whereas the algorithm in this project samples at the end of each episode 128 times whilst updating the weights of the Q network 128 times.

Finally, I also implement an epsilon decay in the training of the agent. This means that after every episode in training, epsilon (the probability of taking a random action) decreases by the decay rate. In this case, epsilon is initiated  at value 1 (making the policy completely random) and is reduced by 0.5% after every episode until it reaches 0.2, where it remains constant. This is to encourage exploratory actions from the agent in earlier episodes to speed up the agent’s learning.

### Deep Q-Network (DQN) 
A basic Q Learning algorithm is implemented to train the agent. Much like with the Deep SARSA I use a multi-layer neural network to estimate the Q table and a replay buffer to sample the information of what happened in the episode and to update/train the neural network. 

Again, an epsilon decay is introduced encouraging early episode environment exploration.



### Sources:

Zhao,D.,Wang,H.,ShaoK.,andZhuY.,(2016),DeepReinforcementLearningwithExperience Replay Based on Sarsa , IEEE Symposium Series on Computational Intelligence (SSCI), Athens, 2016, pp. 1-6, doi: 10.1109/SSCI.2016.7849837.

OpenAI Gym LunarLander-v2: https://gym.openai.com/envs/LunarLander-v2/
