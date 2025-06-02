# Chapter 21: Markov Decision Processes

## Learning Objectives
By the end of this chapter, students will be able to:
- Understand the mathematical framework of Markov Decision Processes (MDPs)
- Implement dynamic programming algorithms for solving MDPs
- Apply value iteration and policy iteration methods
- Analyze convergence properties and optimality conditions
- Design MDP models for real-world sequential decision problems

## Table of Contents
1. [Introduction to MDPs](#introduction)
2. [Mathematical Framework](#mathematical-framework)
3. [Value Functions and Bellman Equations](#value-functions)
4. [Dynamic Programming Algorithms](#dynamic-programming)
5. [Policy Evaluation and Improvement](#policy-methods)
6. [Practical Applications](#applications)

## 1. Introduction to MDPs {#introduction}

Markov Decision Processes provide a mathematical framework for modeling sequential decision-making problems where outcomes are partly random and partly under the control of a decision maker.

### Key Components

**States (S)**: All possible situations the agent can be in
**Actions (A)**: Set of all possible actions available to the agent
**Transition Probabilities (P)**: P(s'|s,a) - probability of reaching state s' from state s taking action a
**Rewards (R)**: R(s,a,s') - immediate reward for transitioning from s to s' via action a
**Discount Factor (γ)**: Value between 0 and 1 that determines importance of future rewards

### Markov Property

The future state depends only on the current state and action, not on the sequence of events that led to the current state:
**P(S_{t+1} = s' | S_t = s_t, A_t = a_t, S_{t-1} = s_{t-1}, ...) = P(S_{t+1} = s' | S_t = s_t, A_t = a_t)**

### Applications
- **Robotics**: Path planning and autonomous navigation
- **Finance**: Portfolio optimization and algorithmic trading
- **Healthcare**: Treatment planning and resource allocation
- **Gaming**: AI opponents and strategic decision making
- **Operations Research**: Inventory management and scheduling

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
from typing import Dict, List, Tuple, Optional

class GridWorldMDP:
    """Grid World environment for MDP demonstrations"""
    
    def __init__(self, height=4, width=4):
        self.height = height
        self.width = width
        self.n_states = height * width
        self.n_actions = 4  # up, down, left, right
        
        # Action mapping
        self.actions = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        self.action_deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        
        # Initialize MDP components
        self.setup_environment()
        
    def setup_environment(self):
        """Set up the grid world with rewards and terminal states"""
        # Create reward matrix
        self.rewards = np.full((self.height, self.width), -0.1)  # Small negative reward for each step
        
        # Terminal states with rewards
        self.rewards[0, 3] = 1.0   # Goal state
        self.rewards[1, 3] = -1.0  # Penalty state
        
        # Obstacles (walls)
        self.obstacles = {(1, 1), (1, 2)}
        for obs in self.obstacles:
            self.rewards[obs] = 0  # No reward for obstacles
        
        # Terminal states
        self.terminal_states = {(0, 3), (1, 3)}
        
    def state_to_position(self, state):
        """Convert state index to (row, col) position"""
        return (state // self.width, state % self.width)
    
    def position_to_state(self, row, col):
        """Convert (row, col) position to state index"""
        return row * self.width + col
    
    def get_valid_actions(self, state):
        """Get valid actions from a given state"""
        if self.state_to_position(state) in self.terminal_states:
            return []
        return list(range(self.n_actions))
    
    def transition(self, state, action):
        """Get next state and reward for taking action in state"""
        pos = self.state_to_position(state)
        
        # If terminal state, stay in same state
        if pos in self.terminal_states:
            return state, 0
        
        # Calculate new position
        delta = self.action_deltas[action]
        new_row = max(0, min(self.height - 1, pos[0] + delta[0]))
        new_col = max(0, min(self.width - 1, pos[1] + delta[1]))
        new_pos = (new_row, new_col)
        
        # Check if new position is obstacle
        if new_pos in self.obstacles:
            new_pos = pos  # Stay in current position
        
        new_state = self.position_to_state(new_pos[0], new_pos[1])
        reward = self.rewards[new_pos]
        
        return new_state, reward
    
    def visualize_environment(self):
        """Visualize the grid world environment"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create display grid
        display_grid = self.rewards.copy()
        
        # Mark obstacles
        for obs in self.obstacles:
            display_grid[obs] = -2
        
        # Create heatmap
        im = ax.imshow(display_grid, cmap='RdYlGn', vmin=-2, vmax=1)
        
        # Add text annotations
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) in self.obstacles:
                    text = 'WALL'
                    color = 'white'
                elif (i, j) in self.terminal_states:
                    if self.rewards[i, j] > 0:
                        text = 'GOAL\n+1'
                        color = 'black'
                    else:
                        text = 'TRAP\n-1'
                        color = 'white'
                else:
                    text = f'{self.rewards[i, j]:.1f}'
                    color = 'black'
                
                ax.text(j, i, text, ha='center', va='center', 
                       color=color, fontweight='bold')
        
        ax.set_title('Grid World Environment')
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        plt.colorbar(im, label='Reward')
        plt.show()

# Initialize and visualize grid world
env = GridWorldMDP()
env.visualize_environment()
```

## 2. Mathematical Framework {#mathematical-framework}

### Formal Definition

An MDP is defined as a 5-tuple (S, A, P, R, γ) where:
- **S**: Finite set of states
- **A**: Finite set of actions  
- **P**: State transition probability function
- **R**: Reward function
- **γ**: Discount factor ∈ [0,1]

### Policy

A **policy π** is a mapping from states to actions:
- **Deterministic Policy**: π: S → A
- **Stochastic Policy**: π: S × A → [0,1]

### Value Functions

**State Value Function**: V^π(s) = E_π[G_t | S_t = s]
where G_t = Σ_{k=0}^∞ γ^k R_{t+k+1} is the expected return

**Action Value Function**: Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]

### Bellman Equations

**State Value Bellman Equation**:
V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]

**Action Value Bellman Equation**:
Q^π(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ Σ_{a'} π(a'|s')Q^π(s',a')]

```python
class MDPSolver:
    """MDP solver with various algorithms"""
    
    def __init__(self, env):
        self.env = env
        self.gamma = 0.9  # Discount factor
        self.theta = 1e-6  # Convergence threshold
        
    def compute_transition_probabilities(self):
        """Compute transition probability matrix"""
        n_states = self.env.n_states
        n_actions = self.env.n_actions
        
        # Initialize P[s][a] = [(prob, next_state, reward), ...]
        self.P = {}
        
        for s in range(n_states):
            self.P[s] = {}
            for a in range(n_actions):
                next_state, reward = self.env.transition(s, a)
                self.P[s][a] = [(1.0, next_state, reward)]  # Deterministic transitions
    
    def policy_evaluation(self, policy, max_iterations=1000):
        """Evaluate a given policy using iterative policy evaluation"""
        V = np.zeros(self.env.n_states)
        
        for iteration in range(max_iterations):
            V_old = V.copy()
            
            for s in range(self.env.n_states):
                if self.env.state_to_position(s) in self.env.terminal_states:
                    continue
                    
                v = 0
                action = policy[s]
                for prob, next_state, reward in self.P[s][action]:
                    v += prob * (reward + self.gamma * V_old[next_state])
                V[s] = v
            
            # Check convergence
            if np.max(np.abs(V - V_old)) < self.theta:
                print(f"Policy evaluation converged after {iteration + 1} iterations")
                break
        
        return V
    
    def value_iteration(self, max_iterations=1000):
        """Find optimal policy using value iteration"""
        V = np.zeros(self.env.n_states)
        policy = np.zeros(self.env.n_states, dtype=int)
        
        iteration_values = []
        
        for iteration in range(max_iterations):
            V_old = V.copy()
            
            for s in range(self.env.n_states):
                if self.env.state_to_position(s) in self.env.terminal_states:
                    continue
                
                action_values = []
                for a in range(self.env.n_actions):
                    q_value = 0
                    for prob, next_state, reward in self.P[s][a]:
                        q_value += prob * (reward + self.gamma * V_old[next_state])
                    action_values.append(q_value)
                
                V[s] = max(action_values)
                policy[s] = np.argmax(action_values)
            
            iteration_values.append(V.copy())
            
            # Check convergence
            if np.max(np.abs(V - V_old)) < self.theta:
                print(f"Value iteration converged after {iteration + 1} iterations")
                break
        
        return V, policy, iteration_values
    
    def policy_iteration(self, max_iterations=100):
        """Find optimal policy using policy iteration"""
        # Initialize random policy
        policy = np.random.choice(self.env.n_actions, self.env.n_states)
        
        for iteration in range(max_iterations):
            # Policy Evaluation
            V = self.policy_evaluation(policy)
            
            # Policy Improvement
            policy_stable = True
            new_policy = policy.copy()
            
            for s in range(self.env.n_states):
                if self.env.state_to_position(s) in self.env.terminal_states:
                    continue
                
                old_action = policy[s]
                action_values = []
                
                for a in range(self.env.n_actions):
                    q_value = 0
                    for prob, next_state, reward in self.P[s][a]:
                        q_value += prob * (reward + self.gamma * V[next_state])
                    action_values.append(q_value)
                
                new_policy[s] = np.argmax(action_values)
                
                if old_action != new_policy[s]:
                    policy_stable = False
            
            policy = new_policy
            
            if policy_stable:
                print(f"Policy iteration converged after {iteration + 1} iterations")
                break
        
        # Final evaluation
        V = self.policy_evaluation(policy)
        return V, policy
    
    def visualize_policy(self, policy, title="Policy"):
        """Visualize policy as arrows in grid"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Arrow symbols for actions
        action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        
        # Create base grid
        display_grid = np.zeros((self.env.height, self.env.width))
        
        for i in range(self.env.height):
            for j in range(self.env.width):
                if (i, j) in self.env.obstacles:
                    display_grid[i, j] = -1
                elif (i, j) in self.env.terminal_states:
                    if self.env.rewards[i, j] > 0:
                        display_grid[i, j] = 1
                    else:
                        display_grid[i, j] = -0.5
        
        im = ax.imshow(display_grid, cmap='RdYlGn', vmin=-1, vmax=1)
        
        # Add policy arrows
        for i in range(self.env.height):
            for j in range(self.env.width):
                state = self.env.position_to_state(i, j)
                
                if (i, j) in self.env.obstacles:
                    text = 'WALL'
                    color = 'white'
                elif (i, j) in self.env.terminal_states:
                    if self.env.rewards[i, j] > 0:
                        text = 'GOAL'
                        color = 'black'
                    else:
                        text = 'TRAP'
                        color = 'white'
                else:
                    action = policy[state]
                    text = action_symbols[action]
                    color = 'black'
                
                ax.text(j, i, text, ha='center', va='center', 
                       color=color, fontsize=16, fontweight='bold')
        
        ax.set_title(title)
        ax.set_xticks(range(self.env.width))
        ax.set_yticks(range(self.env.height))
        plt.colorbar(im)
        plt.show()
    
    def visualize_values(self, V, title="State Values"):
        """Visualize state values as heatmap"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Reshape values to grid
        value_grid = V.reshape(self.env.height, self.env.width)
        
        im = ax.imshow(value_grid, cmap='viridis')
        
        # Add value annotations
        for i in range(self.env.height):
            for j in range(self.env.width):
                state = self.env.position_to_state(i, j)
                
                if (i, j) in self.env.obstacles:
                    text = 'WALL'
                    color = 'white'
                else:
                    text = f'{V[state]:.2f}'
                    color = 'white' if V[state] < 0 else 'black'
                
                ax.text(j, i, text, ha='center', va='center', 
                       color=color, fontweight='bold')
        
        ax.set_title(title)
        ax.set_xticks(range(self.env.width))
        ax.set_yticks(range(self.env.height))
        plt.colorbar(im, label='Value')
        plt.show()

# Solve MDP using different algorithms
solver = MDPSolver(env)
solver.compute_transition_probabilities()

# Value Iteration
print("=== VALUE ITERATION ===")
V_opt, policy_opt, value_history = solver.value_iteration()
solver.visualize_values(V_opt, "Optimal State Values (Value Iteration)")
solver.visualize_policy(policy_opt, "Optimal Policy (Value Iteration)")

# Policy Iteration
print("\n=== POLICY ITERATION ===")
V_pi, policy_pi = solver.policy_iteration()
solver.visualize_values(V_pi, "Optimal State Values (Policy Iteration)")
solver.visualize_policy(policy_pi, "Optimal Policy (Policy Iteration)")
```

## 3. Value Functions and Bellman Equations {#value-functions}

### Optimal Value Functions

The **optimal state value function** V*(s) is the maximum value function over all policies:
V*(s) = max_π V^π(s)

The **optimal action value function** Q*(s,a) is defined as:
Q*(s,a) = max_π Q^π(s,a)

### Bellman Optimality Equations

**Bellman Optimality Equation for V***:
V*(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV*(s')]

**Bellman Optimality Equation for Q***:
Q*(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ max_{a'} Q*(s',a')]

### Properties

**Uniqueness**: The optimal value function is unique
**Optimality**: Any policy that is greedy with respect to V* is optimal
**Consistency**: V*(s) = max_a Q*(s,a)

```python
class BellmanAnalysis:
    """Analysis of Bellman equations and convergence properties"""
    
    def __init__(self, solver):
        self.solver = solver
        self.env = solver.env
        
    def demonstrate_bellman_backup(self):
        """Demonstrate single Bellman backup operation"""
        print("=== BELLMAN BACKUP DEMONSTRATION ===")
        
        # Choose a specific state for demonstration
        demo_state = 5  # Middle of grid
        demo_pos = self.env.state_to_position(demo_state)
        
        print(f"Demonstrating Bellman backup for state {demo_state} at position {demo_pos}")
        
        # Initialize value function
        V = np.random.rand(self.env.n_states)
        
        print(f"Current value V({demo_state}) = {V[demo_state]:.3f}")
        
        # Calculate Q-values for all actions
        q_values = []
        action_details = []
        
        for action in range(self.env.n_actions):
            q_value = 0
            action_name = self.env.actions[action]
            
            for prob, next_state, reward in self.solver.P[demo_state][action]:
                contribution = prob * (reward + self.solver.gamma * V[next_state])
                q_value += contribution
                
                next_pos = self.env.state_to_position(next_state)
                action_details.append({
                    'action': action_name,
                    'next_state': next_state,
                    'next_pos': next_pos,
                    'prob': prob,
                    'reward': reward,
                    'next_value': V[next_state],
                    'contribution': contribution
                })
            
            q_values.append(q_value)
            print(f"Q({demo_state}, {action_name}) = {q_value:.3f}")
        
        # Bellman backup
        new_value = max(q_values)
        best_action = np.argmax(q_values)
        
        print(f"\nBellman backup:")
        print(f"V({demo_state}) ← max_a Q({demo_state}, a) = {new_value:.3f}")
        print(f"Best action: {self.env.actions[best_action]}")
        
        # Visualize backup operation
        self.visualize_backup_operation(demo_state, V, q_values, action_details)
    
    def visualize_backup_operation(self, state, V, q_values, action_details):
        """Visualize the Bellman backup operation"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left plot: Q-values for each action
        actions = [self.env.actions[i] for i in range(len(q_values))]
        bars = axes[0].bar(actions, q_values, alpha=0.7)
        
        # Highlight best action
        best_idx = np.argmax(q_values)
        bars[best_idx].set_color('red')
        bars[best_idx].set_alpha(1.0)
        
        axes[0].set_title(f'Q-values for State {state}')
        axes[0].set_ylabel('Q-value')
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(q_values):
            axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # Right plot: Value function visualization
        value_grid = V.reshape(self.env.height, self.env.width)
        im = axes[1].imshow(value_grid, cmap='viridis')
        
        # Highlight current state
        pos = self.env.state_to_position(state)
        axes[1].add_patch(plt.Rectangle((pos[1]-0.4, pos[0]-0.4), 0.8, 0.8, 
                                       fill=False, color='red', linewidth=3))
        
        # Add value annotations
        for i in range(self.env.height):
            for j in range(self.env.width):
                s = self.env.position_to_state(i, j)
                color = 'white' if V[s] < np.median(V) else 'black'
                axes[1].text(j, i, f'{V[s]:.2f}', ha='center', va='center', 
                           color=color, fontsize=8)
        
        axes[1].set_title(f'Value Function (State {state} highlighted)')
        plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
    
    def analyze_convergence(self, value_history):
        """Analyze convergence properties of value iteration"""
        print("\n=== CONVERGENCE ANALYSIS ===")
        
        if not value_history:
            print("No convergence history available")
            return
        
        # Calculate value differences between iterations
        value_differences = []
        max_changes = []
        
        for i in range(1, len(value_history)):
            diff = np.abs(value_history[i] - value_history[i-1])
            value_differences.append(diff)
            max_changes.append(np.max(diff))
        
        # Plot convergence
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Maximum value change per iteration
        axes[0].plot(max_changes, 'b-', marker='o', linewidth=2)
        axes[0].set_title('Convergence of Value Iteration')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Maximum Value Change')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        
        # Value function evolution for selected states
        selected_states = [0, 5, 10, 15]  # Sample states
        for state in selected_states:
            if state < self.env.n_states:
                values = [V[state] for V in value_history]
                pos = self.env.state_to_position(state)
                axes[1].plot(values, label=f'State {state} {pos}', marker='o')
        
        axes[1].set_title('Value Function Evolution')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('State Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print convergence statistics
        print(f"Total iterations: {len(value_history)}")
        print(f"Final maximum change: {max_changes[-1]:.2e}")
        print(f"Convergence rate (geometric): {max_changes[-1]/max_changes[0]:.2e}")

# Demonstrate Bellman operations and convergence
analysis = BellmanAnalysis(solver)
analysis.demonstrate_bellman_backup()
analysis.analyze_convergence(value_history)
```

## Summary

This chapter covered the fundamental concepts of Markov Decision Processes:

### Key Concepts:
1. **MDP Framework**: States, actions, transitions, rewards, and policies
2. **Value Functions**: State and action value functions with Bellman equations
3. **Dynamic Programming**: Value iteration and policy iteration algorithms
4. **Optimality**: Convergence properties and optimal policies

### Practical Applications:
- **Robotics**: Autonomous navigation and task planning
- **Finance**: Portfolio optimization and risk management
- **Healthcare**: Treatment sequencing and resource allocation
- **Gaming**: Strategic AI and decision making

### Best Practices:
- Choose appropriate discount factors based on problem horizon
- Consider computational complexity for large state spaces
- Validate convergence criteria and stopping conditions
- Design reward functions that encourage desired behaviors
- Handle stochastic transitions and partial observability

---

## Exercises

1. **Custom MDP**: Design and solve an MDP for a specific application domain
2. **Algorithm Comparison**: Compare convergence rates of value vs policy iteration
3. **Sensitivity Analysis**: Analyze how discount factor affects optimal policies
4. **Large State Spaces**: Implement approximation methods for complex MDPs
5. **Real-World Application**: Model a practical decision problem as an MDP

---

*Markov Decision Processes provide the mathematical foundation for sequential decision making under uncertainty, enabling optimal policy computation for complex real-world problems.* 