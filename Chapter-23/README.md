# Chapter 23: Applications of Reinforcement Learning

## Learning Objectives
By the end of this chapter, students will be able to:
- Identify suitable domains for reinforcement learning applications
- Implement RL solutions for game playing and robotics
- Apply RL to financial trading and resource optimization
- Design reward functions for real-world problems
- Evaluate RL system performance in practical settings

## Table of Contents
1. [Game Playing and Strategic AI](#game-playing)
2. [Robotics and Control Systems](#robotics)
3. [Financial Applications](#finance)
4. [Resource Management](#resource-management)
5. [Recommendation Systems](#recommendation)
6. [Healthcare and Treatment Planning](#healthcare)

## 1. Game Playing and Strategic AI {#game-playing}

Reinforcement learning has achieved remarkable success in game playing, from classic board games to complex real-time strategy games.

### Classic Board Games

**Chess**: Deep Blue and AlphaZero revolutionized chess playing
**Go**: AlphaGo's victory over world champions demonstrated RL's potential
**Poker**: Handling imperfect information and bluffing strategies

### Real-Time Strategy Games

**StarCraft II**: Complex multi-agent environments with partial observability
**Dota 2**: Team coordination and long-term strategic planning
**Chess Variants**: Exploring new game mechanics and strategies

```python
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque, defaultdict

class TicTacToeEnv:
    """Tic-Tac-Toe environment for RL demonstration"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset the game board"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.winner = 0
        return self.get_state()
    
    def get_state(self):
        """Get current board state as tuple"""
        return tuple(self.board.flatten())
    
    def get_valid_actions(self):
        """Get list of valid actions (empty positions)"""
        actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    actions.append(i * 3 + j)
        return actions
    
    def step(self, action):
        """Make a move and return (state, reward, done)"""
        if self.done:
            return self.get_state(), 0, True
        
        # Convert action to board position
        row, col = divmod(action, 3)
        
        # Check if move is valid
        if self.board[row, col] != 0:
            return self.get_state(), -10, True  # Invalid move penalty
        
        # Make the move
        self.board[row, col] = self.current_player
        
        # Check for winner
        reward = 0
        if self.check_winner():
            self.done = True
            if self.winner == self.current_player:
                reward = 10  # Win
            else:
                reward = -10  # Should not happen in valid game
        elif len(self.get_valid_actions()) == 0:
            self.done = True
            reward = 1  # Draw
        
        # Switch player
        self.current_player = 3 - self.current_player
        
        return self.get_state(), reward, self.done
    
    def check_winner(self):
        """Check if there's a winner"""
        # Check rows
        for row in self.board:
            if np.all(row == row[0]) and row[0] != 0:
                self.winner = row[0]
                return True
        
        # Check columns
        for col in self.board.T:
            if np.all(col == col[0]) and col[0] != 0:
                self.winner = col[0]
                return True
        
        # Check diagonals
        if np.all(np.diag(self.board) == self.board[0, 0]) and self.board[0, 0] != 0:
            self.winner = self.board[0, 0]
            return True
        
        if np.all(np.diag(np.fliplr(self.board)) == self.board[0, 2]) and self.board[0, 2] != 0:
            self.winner = self.board[0, 2]
            return True
        
        return False
    
    def render(self):
        """Display the current board"""
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        print("\n   0   1   2")
        for i in range(3):
            print(f"{i}  {symbols[self.board[i,0]]} | {symbols[self.board[i,1]]} | {symbols[self.board[i,2]]}")
            if i < 2:
                print("  -----------")

class QLearningAgent:
    """Q-Learning agent for game playing"""
    
    def __init__(self, player_id, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.player_id = player_id
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Q-table: state -> action -> value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # For training analysis
        self.wins = 0
        self.losses = 0
        self.draws = 0
    
    def get_action(self, state, valid_actions, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Get Q-values for valid actions
        q_values = [self.q_table[state][action] for action in valid_actions]
        
        if not q_values:
            return random.choice(valid_actions)
        
        # Choose action with highest Q-value
        max_q = max(q_values)
        best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state, next_valid_actions):
        """Update Q-value using Q-learning rule"""
        current_q = self.q_table[state][action]
        
        if next_valid_actions:
            max_next_q = max([self.q_table[next_state][a] for a in next_valid_actions])
        else:
            max_next_q = 0
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

def train_tic_tac_toe_agents():
    """Train two Q-learning agents to play Tic-Tac-Toe"""
    print("=== TRAINING TIC-TAC-TOE AGENTS ===")
    
    env = TicTacToeEnv()
    agent1 = QLearningAgent(1, epsilon=0.3)  # X player
    agent2 = QLearningAgent(2, epsilon=0.3)  # O player
    
    n_episodes = 10000
    results = {'agent1_wins': [], 'agent2_wins': [], 'draws': []}
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_history = []
        
        while not env.done:
            current_agent = agent1 if env.current_player == 1 else agent2
            valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                break
            
            action = current_agent.get_action(state, valid_actions)
            next_state, reward, done = env.step(action)
            
            # Store experience
            episode_history.append({
                'agent': current_agent,
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'next_valid_actions': env.get_valid_actions()
            })
            
            state = next_state
        
        # Update Q-values based on game outcome
        final_reward_1 = 0
        final_reward_2 = 0
        
        if env.winner == 1:
            final_reward_1 = 10
            final_reward_2 = -10
            agent1.wins += 1
            agent2.losses += 1
        elif env.winner == 2:
            final_reward_1 = -10
            final_reward_2 = 10
            agent2.wins += 1
            agent1.losses += 1
        else:
            final_reward_1 = 1
            final_reward_2 = 1
            agent1.draws += 1
            agent2.draws += 1
        
        # Backpropagate rewards
        for experience in episode_history:
            agent = experience['agent']
            base_reward = experience['reward']
            
            if agent.player_id == 1:
                total_reward = base_reward + final_reward_1
            else:
                total_reward = base_reward + final_reward_2
            
            agent.update_q_value(
                experience['state'],
                experience['action'],
                total_reward,
                experience['next_state'],
                experience['next_valid_actions']
            )
        
        # Decay exploration
        if episode % 1000 == 0:
            agent1.epsilon *= 0.95
            agent2.epsilon *= 0.95
        
        # Record results every 500 episodes
        if episode % 500 == 0 and episode > 0:
            recent_episodes = 500
            agent1_recent_wins = sum(1 for i in range(max(0, episode-recent_episodes), episode) 
                                   if i < len(results['agent1_wins']) and results['agent1_wins'][i])
            results['agent1_wins'].append(agent1_recent_wins)
            results['agent2_wins'].append(agent2_recent_wins)
            results['draws'].append(recent_episodes - agent1_recent_wins - agent2_recent_wins)
    
    # Test learned policies
    print(f"\nTraining completed!")
    print(f"Agent 1 (X): {agent1.wins} wins, {agent1.losses} losses, {agent1.draws} draws")
    print(f"Agent 2 (O): {agent2.wins} wins, {agent2.losses} losses, {agent2.draws} draws")
    
    # Demonstrate a game
    demonstrate_game(agent1, agent2)
    
    return agent1, agent2

def demonstrate_game(agent1, agent2):
    """Demonstrate a game between trained agents"""
    print("\n=== DEMONSTRATION GAME ===")
    
    env = TicTacToeEnv()
    state = env.reset()
    
    env.render()
    
    while not env.done:
        current_agent = agent1 if env.current_player == 1 else agent2
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            break
        
        action = current_agent.get_action(state, valid_actions, training=False)
        state, reward, done = env.step(action)
        
        player_symbol = 'X' if current_agent.player_id == 1 else 'O'
        row, col = divmod(action, 3)
        print(f"\nPlayer {player_symbol} plays at ({row}, {col})")
        env.render()
    
    if env.winner == 1:
        print("\nPlayer X wins!")
    elif env.winner == 2:
        print("\nPlayer O wins!")
    else:
        print("\nIt's a draw!")

# Train agents
agent1, agent2 = train_tic_tac_toe_agents()
```

## 2. Robotics and Control Systems {#robotics}

Reinforcement learning enables robots to learn complex behaviors through trial and error, adapting to new environments and tasks.

### Robotic Applications

**Manipulation**: Grasping, assembly, and object manipulation
**Navigation**: Path planning and obstacle avoidance
**Locomotion**: Walking, running, and dynamic movement
**Human-Robot Interaction**: Adaptive and responsive behavior

### Control Challenges

**Continuous Action Spaces**: Real-valued control signals
**High-Dimensional State Spaces**: Sensor data integration
**Safety Constraints**: Preventing damage to robot or environment
**Real-Time Requirements**: Fast decision making for dynamic environments

```python
class RobotArmEnv:
    """Simplified 2D robot arm environment"""
    
    def __init__(self, arm_length=1.0, target_radius=0.1):
        self.arm_length = arm_length
        self.target_radius = target_radius
        self.max_angle_change = 0.1  # Maximum angle change per step
        
        self.reset()
    
    def reset(self):
        """Reset arm to random initial position with random target"""
        self.joint_angles = np.random.uniform(-np.pi, np.pi, 2)  # Two joints
        self.target_pos = np.random.uniform(-1.5, 1.5, 2)  # Random target
        
        self.steps = 0
        self.max_steps = 200
        
        return self.get_state()
    
    def get_state(self):
        """Get current state: joint angles and target position"""
        end_effector_pos = self.forward_kinematics()
        return np.concatenate([
            self.joint_angles,
            self.target_pos,
            end_effector_pos,
            [self.target_pos[0] - end_effector_pos[0],
             self.target_pos[1] - end_effector_pos[1]]  # Position error
        ])
    
    def forward_kinematics(self):
        """Calculate end effector position from joint angles"""
        # First joint
        x1 = self.arm_length * np.cos(self.joint_angles[0])
        y1 = self.arm_length * np.sin(self.joint_angles[0])
        
        # Second joint
        x2 = x1 + self.arm_length * np.cos(self.joint_angles[0] + self.joint_angles[1])
        y2 = y1 + self.arm_length * np.sin(self.joint_angles[0] + self.joint_angles[1])
        
        return np.array([x2, y2])
    
    def step(self, action):
        """Execute action and return (state, reward, done)"""
        # Apply action (change in joint angles)
        action = np.clip(action, -self.max_angle_change, self.max_angle_change)
        self.joint_angles += action
        
        # Keep angles in valid range
        self.joint_angles = np.clip(self.joint_angles, -np.pi, np.pi)
        
        # Calculate reward
        end_effector_pos = self.forward_kinematics()
        distance_to_target = np.linalg.norm(end_effector_pos - self.target_pos)
        
        # Reward function
        if distance_to_target < self.target_radius:
            reward = 100  # Success bonus
            done = True
        else:
            reward = -distance_to_target  # Penalty based on distance
            done = False
        
        # Add penalty for large movements (encourage smooth motion)
        movement_penalty = 0.1 * np.sum(np.abs(action))
        reward -= movement_penalty
        
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        
        return self.get_state(), reward, done
    
    def render(self, ax=None):
        """Visualize the robot arm"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Calculate joint positions
        joint1_pos = np.array([
            self.arm_length * np.cos(self.joint_angles[0]),
            self.arm_length * np.sin(self.joint_angles[0])
        ])
        
        end_effector_pos = self.forward_kinematics()
        
        # Draw arm segments
        ax.plot([0, joint1_pos[0]], [0, joint1_pos[1]], 'b-', linewidth=5, label='Link 1')
        ax.plot([joint1_pos[0], end_effector_pos[0]], 
               [joint1_pos[1], end_effector_pos[1]], 'r-', linewidth=5, label='Link 2')
        
        # Draw joints
        ax.plot(0, 0, 'ko', markersize=10, label='Base')
        ax.plot(joint1_pos[0], joint1_pos[1], 'go', markersize=8, label='Joint 1')
        ax.plot(end_effector_pos[0], end_effector_pos[1], 'ro', markersize=8, label='End Effector')
        
        # Draw target
        circle = plt.Circle(self.target_pos, self.target_radius, 
                          color='yellow', alpha=0.5, label='Target')
        ax.add_patch(circle)
        ax.plot(self.target_pos[0], self.target_pos[1], 'y*', markersize=15)
        
        # Set equal aspect ratio and limits
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('2D Robot Arm Control')
        
        return ax

class DDPGAgent:
    """Simplified DDPG agent for continuous control"""
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        
        # Simple neural network approximation (linear for demo)
        self.policy_weights = np.random.normal(0, 0.1, (action_dim, state_dim))
        self.value_weights = np.random.normal(0, 0.1, state_dim)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.noise_scale = 0.1
        
    def get_action(self, state, add_noise=True):
        """Get action from policy network"""
        action = np.dot(self.policy_weights, state)
        
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, self.action_dim)
            action += noise
        
        return np.clip(action, -0.1, 0.1)  # Clip to valid range
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=32):
        """Train the agent (simplified version)"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        
        # Simple policy gradient update (simplified)
        for state, action, reward, next_state, done in batch:
            # Value update
            if not done:
                target_value = reward + 0.99 * np.dot(self.value_weights, next_state)
            else:
                target_value = reward
            
            current_value = np.dot(self.value_weights, state)
            value_error = target_value - current_value
            
            # Update value weights
            self.value_weights += self.lr * value_error * state
            
            # Policy update (simplified)
            if reward > 0:  # Positive reward
                self.policy_weights += self.lr * 0.1 * np.outer(action, state)

def train_robot_arm():
    """Train robot arm to reach targets"""
    print("=== ROBOT ARM CONTROL TRAINING ===")
    
    env = RobotArmEnv()
    agent = DDPGAgent(state_dim=8, action_dim=2)
    
    n_episodes = 500
    episode_rewards = []
    success_rate = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        success = False
        
        for step in range(200):
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            agent.store_experience(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                if reward > 50:  # Success
                    success = True
                break
        
        # Train agent
        if episode > 50:  # Start training after some exploration
            agent.train()
        
        episode_rewards.append(total_reward)
        
        # Calculate success rate
        if episode >= 50:
            recent_successes = sum(1 for i in range(max(0, episode-49), episode+1) 
                                 if episode_rewards[i] > 50)
            success_rate.append(recent_successes / 50)
        
        # Reduce noise over time
        agent.noise_scale = max(0.01, agent.noise_scale * 0.995)
    
    # Visualize training progress
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Episode rewards
    window = 20
    smoothed_rewards = [np.mean(episode_rewards[max(0, i-window):i+1]) 
                       for i in range(len(episode_rewards))]
    
    axes[0].plot(episode_rewards, alpha=0.3, label='Raw rewards')
    axes[0].plot(smoothed_rewards, label=f'{window}-episode average')
    axes[0].set_title('Robot Arm Training Progress')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Success rate
    if success_rate:
        axes[1].plot(range(50, len(episode_rewards)), success_rate)
        axes[1].set_title('Success Rate')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Success Rate (last 50 episodes)')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate learned behavior
    demonstrate_robot_control(env, agent)
    
    return agent

def demonstrate_robot_control(env, agent):
    """Demonstrate learned robot control"""
    print("\n=== ROBOT ARM DEMONSTRATION ===")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for demo in range(3):
        state = env.reset()
        trajectory = [env.forward_kinematics()]
        
        for step in range(100):
            action = agent.get_action(state, add_noise=False)
            state, reward, done = env.step(action)
            trajectory.append(env.forward_kinematics())
            
            if done:
                break
        
        # Visualize this demonstration
        env.render(axes[demo])
        
        # Draw trajectory
        trajectory = np.array(trajectory)
        axes[demo].plot(trajectory[:, 0], trajectory[:, 1], 
                       'g--', alpha=0.7, linewidth=2, label='Trajectory')
        axes[demo].set_title(f'Demo {demo + 1}')
        axes[demo].legend()
    
    plt.tight_layout()
    plt.show()

# Train robot arm
robot_agent = train_robot_arm()
```

## Summary

This chapter demonstrated practical applications of reinforcement learning:

### Key Applications:
1. **Game Playing**: Strategic decision making in competitive environments
2. **Robotics**: Motor control and manipulation tasks
3. **Finance**: Trading strategies and portfolio optimization
4. **Resource Management**: Allocation and scheduling problems

### Implementation Considerations:
- **Reward Function Design**: Critical for successful learning
- **Safety Constraints**: Preventing harmful actions during exploration
- **Sample Efficiency**: Minimizing required training data
- **Transfer Learning**: Applying knowledge across related tasks

### Best Practices:
- Start with simplified environments for algorithm development
- Gradually increase complexity as agents improve
- Validate performance in realistic test scenarios
- Consider ethical implications of autonomous decision making
- Implement safety mechanisms for real-world deployment

---

## Exercises

1. **Game AI**: Implement RL agent for a different game (Connect Four, Checkers)
2. **Robot Navigation**: Create maze navigation environment with obstacles
3. **Trading Bot**: Develop RL agent for simplified financial trading
4. **Resource Allocation**: Model and solve scheduling optimization problem
5. **Multi-Agent System**: Implement cooperative or competitive multi-agent environment

---

*Reinforcement learning applications span diverse domains, demonstrating the versatility and power of learning through interaction for solving complex real-world problems.* 