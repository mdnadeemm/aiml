# Chapter 20: Introduction to Reinforcement Learning

## Learning Objectives
By the end of this chapter, students will be able to:
- Understand the fundamental concepts of reinforcement learning
- Implement basic RL algorithms from scratch
- Apply RL techniques to solve sequential decision problems
- Evaluate and compare different RL approaches
- Design RL systems for real-world applications

## Table of Contents
1. [Introduction to Reinforcement Learning](#introduction)
2. [RL Framework and Components](#framework)
3. [Markov Decision Processes](#mdp)
4. [Value Functions and Bellman Equations](#value-functions)
5. [Dynamic Programming](#dynamic-programming)
6. [Monte Carlo Methods](#monte-carlo)
7. [Temporal Difference Learning](#temporal-difference)
8. [Q-Learning](#q-learning)
9. [Policy Gradient Methods](#policy-gradient)
10. [Implementation Examples](#implementation)

## 1. Introduction to Reinforcement Learning {#introduction}

Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. Unlike supervised learning, RL doesn't require labeled data but learns through trial and error, receiving rewards or punishments for its actions.

### Key Characteristics of RL:
- **Learning through interaction**: Agent learns by taking actions and observing outcomes
- **Delayed rewards**: Actions may have consequences that appear much later
- **Exploration vs. exploitation**: Balance between trying new actions and using known good actions
- **Sequential decision making**: Current actions affect future states and rewards

### RL vs. Other Learning Paradigms:

| Aspect | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
|--------|-------------------|----------------------|----------------------|
| Data | Labeled examples | Unlabeled data | Interactive environment |
| Feedback | Immediate, correct answers | No explicit feedback | Delayed rewards/punishments |
| Goal | Predict outputs | Find patterns | Maximize cumulative reward |
| Examples | Classification, Regression | Clustering, Dimensionality reduction | Game playing, Robotics |

### Applications of RL:
- **Game Playing**: Chess, Go, video games
- **Robotics**: Robot navigation, manipulation
- **Autonomous Vehicles**: Path planning, decision making
- **Finance**: Algorithmic trading, portfolio management
- **Healthcare**: Treatment optimization, drug discovery
- **Recommendation Systems**: Personalized content delivery

## 2. RL Framework and Components {#framework}

### The Agent-Environment Interface:

```python
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Environment(ABC):
    """Abstract base class for RL environments"""
    
    @abstractmethod
    def reset(self):
        """Reset environment to initial state"""
        pass
    
    @abstractmethod
    def step(self, action):
        """Take action and return (next_state, reward, done, info)"""
        pass
    
    @abstractmethod
    def get_state_space(self):
        """Return the state space"""
        pass
    
    @abstractmethod
    def get_action_space(self):
        """Return the action space"""
        pass

class Agent(ABC):
    """Abstract base class for RL agents"""
    
    @abstractmethod
    def choose_action(self, state):
        """Choose action given current state"""
        pass
    
    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """Update agent's knowledge"""
        pass

class RLFramework:
    """General RL training framework"""
    
    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent
        self.episode_rewards = []
        self.episode_lengths = []
    
    def train(self, num_episodes=1000, max_steps=200):
        """Train the agent"""
        for episode in range(num_episodes):
            state = self.environment.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Agent chooses action
                action = self.agent.choose_action(state)
                
                # Environment responds
                next_state, reward, done, info = self.environment.step(action)
                
                # Agent learns
                self.agent.update(state, action, reward, next_state, done)
                
                # Update tracking variables
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
    
    def plot_results(self):
        """Plot training results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot episode rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        
        # Plot moving average
        window_size = 100
        if len(self.episode_rewards) >= window_size:
            moving_avg = np.convolve(self.episode_rewards, 
                                   np.ones(window_size)/window_size, mode='valid')
            ax2.plot(moving_avg)
            ax2.set_title(f'Moving Average Reward (window={window_size})')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Average Reward')
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
```

### Key RL Components:

1. **Agent**: The learner/decision maker
2. **Environment**: The world the agent interacts with
3. **State (S)**: Current situation of the agent
4. **Action (A)**: What the agent can do
5. **Reward (R)**: Feedback from the environment
6. **Policy (π)**: Agent's behavior strategy
7. **Value Function (V)**: Expected future reward

## 3. Markov Decision Processes {#mdp}

A Markov Decision Process (MDP) provides the mathematical framework for RL. An MDP is defined by the tuple (S, A, P, R, γ):

- **S**: Set of states
- **A**: Set of actions
- **P**: Transition probabilities P(s'|s,a)
- **R**: Reward function R(s,a,s')
- **γ**: Discount factor (0 ≤ γ ≤ 1)

### Markov Property:
The future depends only on the current state, not on the history of how we got there:
```
P(St+1 = s' | St = s, At = a, St-1, At-1, ..., S0, A0) = P(St+1 = s' | St = s, At = a)
```

### Example: Simple Grid World MDP

```python
class GridWorld(Environment):
    """Simple grid world environment"""
    
    def __init__(self, height=4, width=4, start=(0,0), goal=(3,3)):
        self.height = height
        self.width = width
        self.start = start
        self.goal = goal
        self.state = start
        
        # Actions: 0=up, 1=right, 2=down, 3=left
        self.actions = [(-1,0), (0,1), (1,0), (0,-1)]
        
    def reset(self):
        """Reset to start state"""
        self.state = self.start
        return self.state
    
    def step(self, action):
        """Take action and return (next_state, reward, done, info)"""
        # Calculate next position
        dy, dx = self.actions[action]
        next_y = max(0, min(self.height-1, self.state[0] + dy))
        next_x = max(0, min(self.width-1, self.state[1] + dx))
        
        next_state = (next_y, next_x)
        
        # Calculate reward
        if next_state == self.goal:
            reward = 10.0
            done = True
        else:
            reward = -0.1  # Small negative reward for each step
            done = False
        
        self.state = next_state
        return next_state, reward, done, {}
    
    def get_state_space(self):
        """Return state space size"""
        return self.height * self.width
    
    def get_action_space(self):
        """Return number of actions"""
        return len(self.actions)
    
    def state_to_index(self, state):
        """Convert 2D state to 1D index"""
        return state[0] * self.width + state[1]
    
    def index_to_state(self, index):
        """Convert 1D index to 2D state"""
        return (index // self.width, index % self.width)
    
    def render(self, policy=None):
        """Visualize the grid world"""
        grid = np.zeros((self.height, self.width))
        
        # Mark goal
        grid[self.goal] = 2
        
        # Mark current position
        grid[self.state] = 1
        
        print("Grid World:")
        print("0: Empty, 1: Agent, 2: Goal")
        for row in grid:
            print([int(x) for x in row])
        
        if policy is not None:
            print("\nPolicy:")
            policy_symbols = ['^', '>', 'v', '<']
            for i in range(self.height):
                row = ""
                for j in range(self.width):
                    if (i, j) == self.goal:
                        row += "G "
                    else:
                        state_idx = self.state_to_index((i, j))
                        action = np.argmax(policy[state_idx])
                        row += policy_symbols[action] + " "
                print(row)
```

## 4. Value Functions and Bellman Equations {#value-functions}

### State Value Function:
The value of a state under policy π:
```
V^π(s) = E[Gt | St = s, π]
```

Where Gt is the return (cumulative discounted reward):
```
Gt = Rt+1 + γRt+2 + γ²Rt+3 + ... = Σ(γ^k * Rt+k+1)
```

### Action Value Function (Q-function):
The value of taking action a in state s under policy π:
```
Q^π(s,a) = E[Gt | St = s, At = a, π]
```

### Bellman Equations:

```python
class ValueFunction:
    """Implementation of value functions and Bellman equations"""
    
    def __init__(self, num_states, num_actions, gamma=0.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        
        # Initialize value functions
        self.V = np.zeros(num_states)
        self.Q = np.zeros((num_states, num_actions))
    
    def bellman_expectation_v(self, policy, transition_probs, rewards):
        """Bellman expectation equation for state values"""
        V_new = np.zeros(self.num_states)
        
        for s in range(self.num_states):
            for a in range(self.num_actions):
                for s_prime in range(self.num_states):
                    V_new[s] += (policy[s, a] * 
                               transition_probs[s, a, s_prime] * 
                               (rewards[s, a, s_prime] + self.gamma * self.V[s_prime]))
        
        return V_new
    
    def bellman_expectation_q(self, policy, transition_probs, rewards):
        """Bellman expectation equation for action values"""
        Q_new = np.zeros((self.num_states, self.num_actions))
        
        for s in range(self.num_states):
            for a in range(self.num_actions):
                for s_prime in range(self.num_states):
                    expected_future_value = 0
                    for a_prime in range(self.num_actions):
                        expected_future_value += policy[s_prime, a_prime] * self.Q[s_prime, a_prime]
                    
                    Q_new[s, a] += (transition_probs[s, a, s_prime] * 
                                  (rewards[s, a, s_prime] + self.gamma * expected_future_value))
        
        return Q_new
    
    def bellman_optimality_v(self, transition_probs, rewards):
        """Bellman optimality equation for state values"""
        V_new = np.zeros(self.num_states)
        
        for s in range(self.num_states):
            action_values = []
            for a in range(self.num_actions):
                value = 0
                for s_prime in range(self.num_states):
                    value += (transition_probs[s, a, s_prime] * 
                            (rewards[s, a, s_prime] + self.gamma * self.V[s_prime]))
                action_values.append(value)
            V_new[s] = max(action_values)
        
        return V_new
    
    def bellman_optimality_q(self, transition_probs, rewards):
        """Bellman optimality equation for action values"""
        Q_new = np.zeros((self.num_states, self.num_actions))
        
        for s in range(self.num_states):
            for a in range(self.num_actions):
                for s_prime in range(self.num_states):
                    Q_new[s, a] += (transition_probs[s, a, s_prime] * 
                                  (rewards[s, a, s_prime] + self.gamma * np.max(self.Q[s_prime])))
        
        return Q_new
    
    def policy_from_q(self):
        """Extract greedy policy from Q-values"""
        policy = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            best_action = np.argmax(self.Q[s])
            policy[s, best_action] = 1.0
        return policy
```

## 5. Dynamic Programming {#dynamic-programming}

Dynamic Programming (DP) methods solve MDPs when the model (transition probabilities and rewards) is known.

### Policy Evaluation:

```python
class PolicyEvaluation:
    """Policy evaluation using iterative method"""
    
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.num_states = env.get_state_space()
        self.num_actions = env.get_action_space()
    
    def evaluate_policy(self, policy, max_iterations=1000):
        """Evaluate a given policy"""
        V = np.zeros(self.num_states)
        
        for iteration in range(max_iterations):
            V_old = V.copy()
            
            for s in range(self.num_states):
                v = 0
                state_2d = self.env.index_to_state(s)
                
                for a in range(self.num_actions):
                    action_prob = policy[s, a]
                    
                    # Simulate taking action
                    self.env.state = state_2d
                    next_state_2d, reward, done, _ = self.env.step(a)
                    next_s = self.env.state_to_index(next_state_2d)
                    
                    v += action_prob * (reward + self.gamma * V_old[next_s] * (not done))
                
                V[s] = v
            
            # Check for convergence
            if np.max(np.abs(V - V_old)) < self.theta:
                print(f"Policy evaluation converged after {iteration + 1} iterations")
                break
        
        return V

class PolicyIteration:
    """Policy iteration algorithm"""
    
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.num_states = env.get_state_space()
        self.num_actions = env.get_action_space()
        self.policy_evaluator = PolicyEvaluation(env, gamma, theta)
    
    def policy_improvement(self, V):
        """Improve policy based on value function"""
        policy = np.zeros((self.num_states, self.num_actions))
        
        for s in range(self.num_states):
            action_values = []
            state_2d = self.env.index_to_state(s)
            
            for a in range(self.num_actions):
                # Simulate taking action
                self.env.state = state_2d
                next_state_2d, reward, done, _ = self.env.step(a)
                next_s = self.env.state_to_index(next_state_2d)
                
                action_value = reward + self.gamma * V[next_s] * (not done)
                action_values.append(action_value)
            
            # Choose best action
            best_action = np.argmax(action_values)
            policy[s, best_action] = 1.0
        
        return policy
    
    def run(self, max_iterations=100):
        """Run policy iteration"""
        # Initialize random policy
        policy = np.ones((self.num_states, self.num_actions)) / self.num_actions
        
        for iteration in range(max_iterations):
            # Policy evaluation
            V = self.policy_evaluator.evaluate_policy(policy)
            
            # Policy improvement
            new_policy = self.policy_improvement(V)
            
            # Check for convergence
            if np.array_equal(policy, new_policy):
                print(f"Policy iteration converged after {iteration + 1} iterations")
                break
            
            policy = new_policy
        
        return policy, V

class ValueIteration:
    """Value iteration algorithm"""
    
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.num_states = env.get_state_space()
        self.num_actions = env.get_action_space()
    
    def run(self, max_iterations=1000):
        """Run value iteration"""
        V = np.zeros(self.num_states)
        
        for iteration in range(max_iterations):
            V_old = V.copy()
            
            for s in range(self.num_states):
                action_values = []
                state_2d = self.env.index_to_state(s)
                
                for a in range(self.num_actions):
                    # Simulate taking action
                    self.env.state = state_2d
                    next_state_2d, reward, done, _ = self.env.step(a)
                    next_s = self.env.state_to_index(next_state_2d)
                    
                    action_value = reward + self.gamma * V_old[next_s] * (not done)
                    action_values.append(action_value)
                
                V[s] = max(action_values)
            
            # Check for convergence
            if np.max(np.abs(V - V_old)) < self.theta:
                print(f"Value iteration converged after {iteration + 1} iterations")
                break
        
        # Extract optimal policy
        policy = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            action_values = []
            state_2d = self.env.index_to_state(s)
            
            for a in range(self.num_actions):
                self.env.state = state_2d
                next_state_2d, reward, done, _ = self.env.step(a)
                next_s = self.env.state_to_index(next_state_2d)
                
                action_value = reward + self.gamma * V[next_s] * (not done)
                action_values.append(action_value)
            
            best_action = np.argmax(action_values)
            policy[s, best_action] = 1.0
        
        return policy, V
```

## 6. Monte Carlo Methods {#monte-carlo}

Monte Carlo methods learn from complete episodes without requiring a model of the environment.

```python
class MonteCarloAgent(Agent):
    """Monte Carlo agent for policy evaluation and control"""
    
    def __init__(self, num_states, num_actions, gamma=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-values and visit counts
        self.Q = np.zeros((num_states, num_actions))
        self.returns = {}  # Dictionary to store returns for each (state, action) pair
        self.episode_buffer = []  # Store current episode
        
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, done):
        """Store experience in episode buffer"""
        self.episode_buffer.append((state, action, reward))
        
        if done:
            self._update_from_episode()
            self.episode_buffer = []
    
    def _update_from_episode(self):
        """Update Q-values using first-visit Monte Carlo"""
        G = 0  # Return
        visited_sa_pairs = set()
        
        # Process episode backwards
        for t in reversed(range(len(self.episode_buffer))):
            state, action, reward = self.episode_buffer[t]
            G = self.gamma * G + reward
            
            # First-visit MC
            if (state, action) not in visited_sa_pairs:
                visited_sa_pairs.add((state, action))
                
                # Update returns
                if (state, action) not in self.returns:
                    self.returns[(state, action)] = []
                self.returns[(state, action)].append(G)
                
                # Update Q-value (average of returns)
                self.Q[state, action] = np.mean(self.returns[(state, action)])

class MonteCarloES:
    """Monte Carlo Exploring Starts"""
    
    def __init__(self, num_states, num_actions, gamma=0.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        
        self.Q = np.random.random((num_states, num_actions))
        self.policy = np.zeros((num_states, num_actions))
        self.returns = {}
        
        # Initialize greedy policy
        for s in range(num_states):
            best_action = np.argmax(self.Q[s])
            self.policy[s, best_action] = 1.0
    
    def generate_episode(self, env, max_steps=200):
        """Generate episode with exploring starts"""
        # Random starting state and action
        start_state = np.random.randint(self.num_states)
        start_action = np.random.randint(self.num_actions)
        
        env.state = env.index_to_state(start_state)
        episode = []
        
        # Take first action
        next_state_2d, reward, done, _ = env.step(start_action)
        next_state = env.state_to_index(next_state_2d)
        episode.append((start_state, start_action, reward))
        
        if done:
            return episode
        
        # Follow policy for rest of episode
        state = next_state
        for step in range(max_steps - 1):
            action = np.argmax(self.policy[state])
            next_state_2d, reward, done, _ = env.step(action)
            next_state = env.state_to_index(next_state_2d)
            
            episode.append((state, action, reward))
            
            if done:
                break
            
            state = next_state
        
        return episode
    
    def update_from_episode(self, episode):
        """Update Q-values and policy from episode"""
        G = 0
        visited_sa_pairs = set()
        
        # Process episode backwards
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            
            if (state, action) not in visited_sa_pairs:
                visited_sa_pairs.add((state, action))
                
                # Update returns
                if (state, action) not in self.returns:
                    self.returns[(state, action)] = []
                self.returns[(state, action)].append(G)
                
                # Update Q-value
                self.Q[state, action] = np.mean(self.returns[(state, action)])
                
                # Update policy (greedy)
                best_action = np.argmax(self.Q[state])
                self.policy[state] = 0
                self.policy[state, best_action] = 1.0
    
    def train(self, env, num_episodes=5000):
        """Train using Monte Carlo ES"""
        for episode in range(num_episodes):
            episode_data = self.generate_episode(env)
            self.update_from_episode(episode_data)
            
            if (episode + 1) % 1000 == 0:
                print(f"Episode {episode + 1} completed")
```

## 7. Temporal Difference Learning {#temporal-difference}

Temporal Difference (TD) learning combines ideas from Monte Carlo and Dynamic Programming.

### TD(0) for Policy Evaluation:

```python
class TDAgent(Agent):
    """Temporal Difference learning agent"""
    
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        self.V = np.zeros(num_states)  # State values
        self.Q = np.zeros((num_states, num_actions))  # Action values
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.Q[state])
    
    def td_prediction(self, state, reward, next_state, done):
        """TD(0) prediction for state values"""
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.V[next_state]
        
        td_error = td_target - self.V[state]
        self.V[state] += self.alpha * td_error
        
        return td_error
    
    def update(self, state, action, reward, next_state, done):
        """Update using TD learning"""
        # Update state value
        td_error = self.td_prediction(state, reward, next_state, done)
        
        # Update action value (SARSA-style update would go here)
        pass

class SARSAAgent(Agent):
    """SARSA (State-Action-Reward-State-Action) agent"""
    
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.Q = np.zeros((num_states, num_actions))
        self.last_state = None
        self.last_action = None
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, done):
        """SARSA update"""
        if self.last_state is not None:
            # Choose next action using current policy
            if done:
                next_action = 0  # Doesn't matter, next_q will be 0
                next_q = 0
            else:
                next_action = self.choose_action(next_state)
                next_q = self.Q[next_state, next_action]
            
            # SARSA update
            td_target = reward + self.gamma * next_q
            td_error = td_target - self.Q[self.last_state, self.last_action]
            self.Q[self.last_state, self.last_action] += self.alpha * td_error
        
        self.last_state = state
        self.last_action = action
```

## 8. Q-Learning {#q-learning}

Q-Learning is an off-policy TD control algorithm that learns the optimal action-value function.

```python
class QLearningAgent(Agent):
    """Q-Learning agent"""
    
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.Q = np.zeros((num_states, num_actions))
        
        # Exploration schedule
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, done):
        """Q-Learning update"""
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])
        
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_policy(self):
        """Extract greedy policy"""
        policy = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            best_action = np.argmax(self.Q[s])
            policy[s, best_action] = 1.0
        return policy

class DoubleQLearningAgent(Agent):
    """Double Q-Learning agent to reduce overestimation bias"""
    
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Two Q-tables
        self.Q1 = np.zeros((num_states, num_actions))
        self.Q2 = np.zeros((num_states, num_actions))
        
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def choose_action(self, state):
        """Epsilon-greedy using sum of Q-tables"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            Q_combined = self.Q1[state] + self.Q2[state]
            return np.argmax(Q_combined)
    
    def update(self, state, action, reward, next_state, done):
        """Double Q-Learning update"""
        if np.random.random() < 0.5:
            # Update Q1
            if done:
                td_target = reward
            else:
                best_action = np.argmax(self.Q1[next_state])
                td_target = reward + self.gamma * self.Q2[next_state, best_action]
            
            td_error = td_target - self.Q1[state, action]
            self.Q1[state, action] += self.alpha * td_error
        else:
            # Update Q2
            if done:
                td_target = reward
            else:
                best_action = np.argmax(self.Q2[next_state])
                td_target = reward + self.gamma * self.Q1[next_state, best_action]
            
            td_error = td_target - self.Q2[state, action]
            self.Q2[state, action] += self.alpha * td_error
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

## 9. Policy Gradient Methods {#policy-gradient}

Policy gradient methods directly optimize the policy without maintaining value functions.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """Neural network policy"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

class REINFORCEAgent(Agent):
    """REINFORCE policy gradient agent"""
    
    def __init__(self, state_size, action_size, hidden_size=128, lr=0.01, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        # Policy network
        self.policy_net = PolicyNetwork(state_size, hidden_size, action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def choose_action(self, state):
        """Choose action using policy network"""
        if isinstance(state, tuple):
            # Convert 2D state to 1D for neural network
            state_tensor = torch.FloatTensor([state[0], state[1]]).unsqueeze(0)
        else:
            state_tensor = torch.FloatTensor([state]).unsqueeze(0)
        
        action_probs = self.policy_net(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        # Store for training
        self.episode_states.append(state_tensor)
        self.episode_actions.append(action)
        
        return action.item()
    
    def update(self, state, action, reward, next_state, done):
        """Store reward and update if episode is done"""
        self.episode_rewards.append(reward)
        
        if done:
            self._update_policy()
            self._reset_episode()
    
    def _update_policy(self):
        """Update policy using REINFORCE algorithm"""
        # Calculate returns
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Convert to tensor and normalize
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for i in range(len(self.episode_states)):
            state = self.episode_states[i]
            action = self.episode_actions[i]
            Return = returns[i]
            
            action_probs = self.policy_net(state)
            action_dist = torch.distributions.Categorical(action_probs)
            log_prob = action_dist.log_prob(action)
            
            policy_loss.append(-log_prob * Return)
        
        # Update policy network
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
    
    def _reset_episode(self):
        """Reset episode memory"""
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

class ActorCriticAgent(Agent):
    """Actor-Critic agent"""
    
    def __init__(self, state_size, action_size, hidden_size=128, lr=0.01, gamma=0.99):
        self.gamma = gamma
        
        # Actor network (policy)
        self.actor = PolicyNetwork(state_size, hidden_size, action_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
    
    def choose_action(self, state):
        """Choose action using actor network"""
        if isinstance(state, tuple):
            state_tensor = torch.FloatTensor([state[0], state[1]]).unsqueeze(0)
        else:
            state_tensor = torch.FloatTensor([state]).unsqueeze(0)
        
        action_probs = self.actor(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return action.item()
    
    def update(self, state, action, reward, next_state, done):
        """Update actor and critic networks"""
        # Convert states to tensors
        if isinstance(state, tuple):
            state_tensor = torch.FloatTensor([state[0], state[1]]).unsqueeze(0)
        else:
            state_tensor = torch.FloatTensor([state]).unsqueeze(0)
        
        if isinstance(next_state, tuple):
            next_state_tensor = torch.FloatTensor([next_state[0], next_state[1]]).unsqueeze(0)
        else:
            next_state_tensor = torch.FloatTensor([next_state]).unsqueeze(0)
        
        # Calculate TD target and error
        current_value = self.critic(state_tensor)
        if done:
            td_target = reward
        else:
            next_value = self.critic(next_state_tensor)
            td_target = reward + self.gamma * next_value
        
        td_error = td_target - current_value
        
        # Update critic
        critic_loss = td_error.pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        action_probs = self.actor(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        log_prob = action_dist.log_prob(torch.tensor(action))
        
        actor_loss = -log_prob * td_error.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
```

## 10. Implementation Examples {#implementation}

### Complete Training Example:

```python
def train_and_compare_agents():
    """Train and compare different RL agents"""
    # Create environment
    env = GridWorld(height=4, width=4)
    num_states = env.get_state_space()
    num_actions = env.get_action_space()
    
    # Create agents
    agents = {
        'Q-Learning': QLearningAgent(num_states, num_actions),
        'SARSA': SARSAAgent(num_states, num_actions),
        'Double Q-Learning': DoubleQLearningAgent(num_states, num_actions),
        'Monte Carlo': MonteCarloAgent(num_states, num_actions)
    }
    
    # Training parameters
    num_episodes = 2000
    results = {}
    
    for agent_name, agent in agents.items():
        print(f"\nTraining {agent_name}...")
        framework = RLFramework(env, agent)
        framework.train(num_episodes)
        results[agent_name] = framework.episode_rewards
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    for agent_name, rewards in results.items():
        # Calculate moving average
        window_size = 100
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(moving_avg, label=agent_name)
    
    plt.title('RL Agents Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return results

def demonstrate_policy():
    """Demonstrate learned policy"""
    env = GridWorld(height=4, width=4)
    agent = QLearningAgent(env.get_state_space(), env.get_action_space())
    
    # Train agent
    framework = RLFramework(env, agent)
    framework.train(1000)
    
    # Extract and display policy
    policy = agent.get_policy()
    env.render(policy)
    
    # Test policy
    state = env.reset()
    total_reward = 0
    steps = 0
    
    print("\nTesting learned policy:")
    while steps < 20:
        state_idx = env.state_to_index(state)
        action = agent.choose_action(state_idx)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        print(f"Step {steps}: State {state}, Reward {reward:.2f}")
        
        if done:
            break
    
    print(f"Total reward: {total_reward:.2f}")

# Run examples
if __name__ == "__main__":
    # Compare different agents
    results = train_and_compare_agents()
    
    # Demonstrate learned policy
    demonstrate_policy()
```

## Summary

This chapter introduced the fundamentals of reinforcement learning:

1. **RL Framework**: Agent-environment interaction and key components
2. **MDPs**: Mathematical foundation for RL problems
3. **Value Functions**: State and action value functions with Bellman equations
4. **Dynamic Programming**: Model-based methods for known environments
5. **Monte Carlo**: Learning from complete episodes
6. **Temporal Difference**: Learning from individual steps
7. **Q-Learning**: Off-policy control algorithm
8. **Policy Gradients**: Direct policy optimization methods

### Key Takeaways:
- RL learns through trial and error interaction
- Balance exploration vs exploitation is crucial
- Different algorithms suit different problem types
- Value-based and policy-based methods each have advantages
- Practical implementation requires careful tuning

---

## Exercises

1. Implement and compare SARSA vs Q-Learning on different environments
2. Create a custom environment and solve it with multiple RL algorithms
3. Implement experience replay for Q-Learning
4. Design a multi-armed bandit problem and solve it
5. Build a simple game AI using reinforcement learning

---

*Master reinforcement learning through hands-on implementation of algorithms and environments.* 