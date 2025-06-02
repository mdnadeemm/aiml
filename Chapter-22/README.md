# Chapter 22: Training Models through Trial and Error

## Learning Objectives
By the end of this chapter, students will be able to:
- Understand temporal difference learning and Q-learning algorithms
- Implement Monte Carlo methods for reinforcement learning
- Apply function approximation techniques for large state spaces
- Design exploration strategies for effective learning
- Build practical RL agents for complex environments

## Table of Contents
1. [Introduction to Trial and Error Learning](#introduction)
2. [Monte Carlo Methods](#monte-carlo)
3. [Temporal Difference Learning](#temporal-difference)
4. [Q-Learning and SARSA](#q-learning)
5. [Function Approximation](#function-approximation)
6. [Exploration vs Exploitation](#exploration)

## 1. Introduction to Trial and Error Learning {#introduction}

Trial and error learning forms the core of reinforcement learning, where agents learn optimal behavior through direct interaction with their environment without explicit supervision.

### Key Principles

**Learning from Experience**: Agents improve performance through repeated interactions
**Delayed Rewards**: Actions may have consequences that manifest later
**Exploration vs Exploitation**: Balancing trying new actions vs using known good actions
**Credit Assignment**: Determining which actions led to rewards or penalties

### Learning Approaches

**Model-Free Learning**: Learn directly from experience without building environment model
**Model-Based Learning**: Learn environment model and use it for planning
**On-Policy Learning**: Learn about policy being followed
**Off-Policy Learning**: Learn about different policy from the one being followed

```python
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict, deque
import seaborn as sns

class RLAgent:
    """Base class for reinforcement learning agents"""
    
    def __init__(self, n_actions, learning_rate=0.1, discount_factor=0.9):
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = 0.1  # Exploration rate
        
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        raise NotImplementedError
        
    def update(self, state, action, reward, next_state, done):
        """Update agent's knowledge"""
        raise NotImplementedError
        
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair"""
        raise NotImplementedError

class MultiArmedBandit:
    """Multi-armed bandit environment for basic RL demonstration"""
    
    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        # True action values (unknown to agent)
        self.true_values = np.random.normal(0, 1, n_arms)
        self.optimal_action = np.argmax(self.true_values)
        
    def step(self, action):
        """Execute action and return reward"""
        # Reward is sampled from normal distribution around true value
        reward = np.random.normal(self.true_values[action], 1)
        return reward
    
    def visualize_arms(self):
        """Visualize the bandit arms and their true values"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(range(self.n_arms), self.true_values, alpha=0.7)
        bars[self.optimal_action].set_color('red')
        bars[self.optimal_action].set_alpha(1.0)
        
        ax.set_title('Multi-Armed Bandit: True Action Values')
        ax.set_xlabel('Action')
        ax.set_ylabel('True Value')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Highlight optimal action
        ax.text(self.optimal_action, self.true_values[self.optimal_action] + 0.1, 
                'Optimal', ha='center', fontweight='bold', color='red')
        
        plt.show()
        
        print(f"Optimal action: {self.optimal_action}")
        print(f"Optimal value: {self.true_values[self.optimal_action]:.3f}")

# Demonstrate multi-armed bandit
bandit = MultiArmedBandit(n_arms=10)
bandit.visualize_arms()
```

## 2. Monte Carlo Methods {#monte-carlo}

Monte Carlo methods learn value functions and optimal policies from complete episodes of experience, using actual returns rather than estimates.

### Key Concepts

**Episode**: Complete sequence from start to terminal state
**Return**: Total discounted reward from a time step to episode end
**First-Visit MC**: Update values using first visit to each state in episode
**Every-Visit MC**: Update values using every visit to each state

### Monte Carlo Policy Evaluation

For policy π, estimate V^π(s) by averaging returns from visits to state s:
V(s) ← V(s) + α[G_t - V(s)]

where G_t is the return following first visit to s at time t.

```python
class MonteCarloAgent(RLAgent):
    """Monte Carlo learning agent"""
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.9):
        super().__init__(n_actions, learning_rate, discount_factor)
        self.n_states = n_states
        
        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))
        
        # Track visits for first-visit MC
        self.returns = defaultdict(list)
        
        # Current episode data
        self.episode_data = []
        
    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def store_experience(self, state, action, reward):
        """Store experience for current episode"""
        self.episode_data.append((state, action, reward))
    
    def end_episode(self):
        """Process completed episode using Monte Carlo update"""
        # Calculate returns for each step
        G = 0
        visited = set()
        
        # Process episode backwards
        for t in reversed(range(len(self.episode_data))):
            state, action, reward = self.episode_data[t]
            G = self.gamma * G + reward
            
            # First-visit Monte Carlo
            if (state, action) not in visited:
                visited.add((state, action))
                
                # Update Q-value
                old_q = self.q_table[state, action]
                self.q_table[state, action] += self.alpha * (G - old_q)
        
        # Clear episode data
        self.episode_data = []
    
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair"""
        return self.q_table[state, action]

class GridWorldEnvironment:
    """Simple grid world for RL demonstrations"""
    
    def __init__(self, height=4, width=4):
        self.height = height
        self.width = width
        self.n_states = height * width
        self.n_actions = 4  # up, down, left, right
        
        # Define terminal states and rewards
        self.terminal_states = {3, 7}  # Top-right, middle-right
        self.rewards = {3: 1.0, 7: -1.0}  # Goal: +1, Trap: -1
        
        # Action effects
        self.action_deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        
        self.reset()
    
    def reset(self):
        """Reset environment to start state"""
        self.state = 0  # Start at top-left
        return self.state
    
    def step(self, action):
        """Execute action and return (next_state, reward, done)"""
        if self.state in self.terminal_states:
            return self.state, 0, True
        
        # Calculate new position
        row, col = divmod(self.state, self.width)
        delta = self.action_deltas[action]
        new_row = max(0, min(self.height - 1, row + delta[0]))
        new_col = max(0, min(self.width - 1, col + delta[1]))
        
        new_state = new_row * self.width + new_col
        
        # Determine reward
        if new_state in self.rewards:
            reward = self.rewards[new_state]
        else:
            reward = -0.1  # Small penalty for each step
        
        # Check if episode is done
        done = new_state in self.terminal_states
        
        self.state = new_state
        return new_state, reward, done
    
    def visualize_policy(self, agent, title="Policy"):
        """Visualize learned policy"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        policy_grid = np.zeros((self.height, self.width))
        value_grid = np.zeros((self.height, self.width))
        
        # Extract policy and values
        for state in range(self.n_states):
            row, col = divmod(state, self.width)
            if state not in self.terminal_states:
                best_action = np.argmax(agent.q_table[state])
                policy_grid[row, col] = best_action
                value_grid[row, col] = np.max(agent.q_table[state])
        
        # Create visualization
        im = ax.imshow(value_grid, cmap='viridis')
        
        # Action arrows
        action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        
        for row in range(self.height):
            for col in range(self.width):
                state = row * self.width + col
                
                if state in self.terminal_states:
                    if state in self.rewards and self.rewards[state] > 0:
                        text = 'GOAL'
                        color = 'yellow'
                    else:
                        text = 'TRAP'
                        color = 'red'
                else:
                    action = int(policy_grid[row, col])
                    text = action_symbols[action]
                    color = 'white'
                
                ax.text(col, row, text, ha='center', va='center', 
                       color=color, fontsize=12, fontweight='bold')
        
        ax.set_title(title)
        plt.colorbar(im, label='State Value')
        plt.show()

def train_monte_carlo_agent():
    """Train Monte Carlo agent on grid world"""
    print("=== MONTE CARLO LEARNING ===")
    
    env = GridWorldEnvironment()
    agent = MonteCarloAgent(env.n_states, env.n_actions, learning_rate=0.1)
    
    n_episodes = 1000
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        # Run episode
        while True:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            agent.store_experience(state, action, reward)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done or steps > 100:  # Prevent infinite episodes
                break
        
        # Update agent after episode
        agent.end_episode()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Decay exploration
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Episode rewards
    window = 50
    smoothed_rewards = [np.mean(episode_rewards[max(0, i-window):i+1]) 
                       for i in range(len(episode_rewards))]
    
    axes[0].plot(episode_rewards, alpha=0.3, label='Raw')
    axes[0].plot(smoothed_rewards, label=f'{window}-episode average')
    axes[0].set_title('Learning Progress: Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Episode lengths
    smoothed_lengths = [np.mean(episode_lengths[max(0, i-window):i+1]) 
                       for i in range(len(episode_lengths))]
    
    axes[1].plot(episode_lengths, alpha=0.3, label='Raw')
    axes[1].plot(smoothed_lengths, label=f'{window}-episode average')
    axes[1].set_title('Learning Progress: Episode Length')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps to Completion')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Show final policy
    env.visualize_policy(agent, "Monte Carlo Learned Policy")
    
    return agent

# Train Monte Carlo agent
mc_agent = train_monte_carlo_agent()
```

## 3. Temporal Difference Learning {#temporal-difference}

Temporal Difference (TD) learning combines ideas from Monte Carlo and dynamic programming, learning from each step rather than waiting for episode completion.

### TD(0) Algorithm

**Update Rule**: V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]

**TD Error**: δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)

### Advantages over Monte Carlo
- **Online Learning**: Can learn before episode completion
- **Lower Variance**: Updates based on estimates rather than complete returns
- **Faster Learning**: Often converges faster than Monte Carlo methods

```python
class TDAgent(RLAgent):
    """Temporal Difference learning agent"""
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.9):
        super().__init__(n_actions, learning_rate, discount_factor)
        self.n_states = n_states
        
        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))
        
        # Track TD errors for analysis
        self.td_errors = []
        
    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """TD(0) update"""
        # Current Q-value
        current_q = self.q_table[state, action]
        
        # Target Q-value
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        
        # TD error
        td_error = target_q - current_q
        self.td_errors.append(abs(td_error))
        
        # Update Q-value
        self.q_table[state, action] += self.alpha * td_error
    
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair"""
        return self.q_table[state, action]

def train_td_agent():
    """Train TD learning agent"""
    print("\n=== TEMPORAL DIFFERENCE LEARNING ===")
    
    env = GridWorldEnvironment()
    agent = TDAgent(env.n_states, env.n_actions, learning_rate=0.1)
    
    n_episodes = 1000
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # TD update
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done or steps > 100:
                break
        
        episode_rewards.append(total_reward)
        
        # Decay exploration
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
    
    # Compare with Monte Carlo
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Learning curves comparison
    window = 50
    td_smoothed = [np.mean(episode_rewards[max(0, i-window):i+1]) 
                   for i in range(len(episode_rewards))]
    
    axes[0].plot(td_smoothed, label='TD Learning', linewidth=2)
    axes[0].set_title('TD vs Monte Carlo Learning')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Average Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # TD errors
    if agent.td_errors:
        window_errors = 100
        smoothed_errors = [np.mean(agent.td_errors[max(0, i-window_errors):i+1]) 
                          for i in range(len(agent.td_errors))]
        
        axes[1].plot(smoothed_errors)
        axes[1].set_title('TD Error Over Time')
        axes[1].set_xlabel('Update Step')
        axes[1].set_ylabel('Average |TD Error|')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Show learned policy
    env.visualize_policy(agent, "TD Learned Policy")
    
    return agent

# Train TD agent
td_agent = train_td_agent()
```

## Summary

This chapter covered trial and error learning methods in reinforcement learning:

### Key Concepts:
1. **Monte Carlo Methods**: Learning from complete episodes and actual returns
2. **Temporal Difference Learning**: Online learning with bootstrapping
3. **Exploration Strategies**: Balancing exploration and exploitation
4. **Value Function Approximation**: Handling large state spaces

### Practical Applications:
- **Game Playing**: Learning strategies through self-play
- **Robotics**: Learning motor skills and navigation
- **Resource Management**: Optimizing allocation strategies
- **Financial Trading**: Learning trading strategies from market data

### Best Practices:
- Choose appropriate learning rates for stable convergence
- Implement effective exploration strategies
- Monitor learning progress and adjust hyperparameters
- Consider function approximation for large state spaces
- Validate learned policies in test environments

---

## Exercises

1. **Bandit Algorithms**: Implement and compare different bandit algorithms
2. **TD Variants**: Explore TD(λ) and other temporal difference methods
3. **Function Approximation**: Use neural networks for value function approximation
4. **Custom Environment**: Design and solve a new RL environment
5. **Hyperparameter Tuning**: Optimize learning parameters for better performance

---

*Trial and error learning enables agents to discover optimal behaviors through direct interaction with their environment, forming the foundation of modern reinforcement learning systems.* 