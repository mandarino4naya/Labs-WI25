import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

np.random.seed(0)

class WindyCliffWorld(gym.Env):
    def __init__(self):
        super(WindyCliffWorld, self).__init__()
        
        self.grid_size = (7, 10)
        self.start_state = (3, 0)
        self.goal_state = (3, 9)
        self.cliff = [(3, i) for i in range(1, 9)]
        self.obstacles = [(2, 4), (4, 4), (2, 7), (4, 7)]
        
        self.wind_strength = {
            (i, j): np.random.choice([-1, 0, 1]) for i in range(self.grid_size[0]) for j in range(self.grid_size[1])
        }

        self.action_space = spaces.Discrete(4)  # 4 possible actions: up, down, left, right
        self.observation_space = spaces.Discrete(self.grid_size[0] * self.grid_size[1])
        
        self.state = self.start_state
        
        self.action_effects = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }

    def reset(self):
        self.state = self.start_state
        return self.state_to_index(self.state)
    
    def step(self, action):
        new_state = (self.state[0] + self.action_effects[action][0], self.state[1] + self.action_effects[action][1])
        new_state = (max(0, min(new_state[0], self.grid_size[0] - 1)), max(0, min(new_state[1], self.grid_size[1] - 1)))

        # Apply wind effect
        wind = self.wind_strength[new_state]
        new_state = (new_state[0] + wind, new_state[1])
        new_state = (max(0, min(new_state[0], self.grid_size[0] - 1)), max(0, min(new_state[1], self.grid_size[1] - 1)))

        if new_state in self.cliff:
            reward = -100
            done = True
            new_state = self.start_state
        elif new_state == self.goal_state:
            reward = 10
            done = True
        elif new_state in self.obstacles:
            reward = -10
            done = False
        else:
            reward = -1
            done = False

        self.state = new_state
        return self.state_to_index(new_state), reward, done, {}
    
    def state_to_index(self, state):
        return state[0] * self.grid_size[1] + state[1]
    
    def index_to_state(self, index):
        return (index // self.grid_size[1], index % self.grid_size[1])
    
    def render(self):
        grid = np.zeros(self.grid_size)
        grid[self.state] = 1  # Current position
        for c in self.cliff:
            grid[c] = -1  # Cliff positions
        for o in self.obstacles:
            grid[o] = -0.5  # Obstacle positions
        grid[self.goal_state] = 2  # Goal position
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap='viridis')
        plt.axis('off')
        fig.canvas.draw()
        plt.close(fig)
        image = np.array(fig.canvas.get_renderer().buffer_rgba()) # had to edit fig.canvas.get_renderer() for it to work on MacOS
        return image

# Create and register the environment
env = WindyCliffWorld()

def q_learning(env, num_episodes, alpha, gamma, epsilon):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    for _ in range(num_episodes):
        s = env.reset()
        end = False
        while not end:
            # e-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                action = np.argmax(q_table[s])
            next_s, reward, end, _ = env.step(action)
            # off-policy upd
            q_table[s, action] += alpha * (reward + gamma * np.max(q_table[next_s]) - q_table[s, action])
            s = next_s
    return q_table

def sarsa(env, num_episodes, alpha, gamma, epsilon):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    max_steps = 200  # added step-limit to avoid inf loop

    for _ in range(num_episodes):
        s = env.reset()
        end = False
        # e-greedy
        if np.random.rand() < epsilon:
            action = np.random.randint(env.action_space.n)
        else:
            action = np.argmax(q_table[s])
        
        steps = 0
        while not end and steps < max_steps:
            next_s, reward, end, _ = env.step(action)
            
            if not end:
                if np.random.rand() < epsilon:
                    next_action = np.random.randint(env.action_space.n)
                else:
                    next_action = np.argmax(q_table[next_s])
                
                # sarsa upd
                q_table[s, action] += alpha * (
                    reward + gamma * q_table[next_s, next_action] - q_table[s, action]
                )
                
                s, action = next_s, next_action
            else:
                q_table[s, action] += alpha * (reward - q_table[s, action])
            
            steps += 1
    return q_table

def save_gif(frames, path='./', filename='gym_animation.gif'):
    imageio.mimsave(os.path.join(path, filename), frames, duration=0.5)

def visualize_policy(env, q_table, filename='q_learning.gif'):
    state = env.reset()
    frames = []
    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, _, done, _ = env.step(action)
        frames.append(env.render())
    
    save_gif(frames, filename=filename)

# Example usage:

# Testing Q-Learning
env = WindyCliffWorld()
q_table = q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1)
visualize_policy(env, q_table, filename='q_learning_windy_cliff.gif')

# Testing SARSA
env = WindyCliffWorld()
q_table = sarsa(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1)
visualize_policy(env, q_table, filename='sarsa_windy_cliff.gif')

### Helpers for running experiments with different hyperparameters ###

def eval_policy(env, q_table, num_eval_episodes=50, epsilon=0.05):
    rewards = []
    for _ in range(num_eval_episodes):
        s = env.reset()
        end = False
        total_reward = 0
        while not end:
            # e-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                action = np.argmax(q_table[s])
            s, reward, end, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return rewards

def run_exp(alg, num_episodes=500, gamma=0.99, num_eval_episodes=50):
    alphas = [0.1, 0.5]
    epsilons = [0.1, 0.5]
    results = {}
    
    for alpha in alphas:
        for epsilon in epsilons:
            env = WindyCliffWorld()
            if alg == 'q_learning':
                q_table = q_learning(env, num_episodes, alpha, gamma, epsilon)
            elif alg == 'sarsa':
                q_table = sarsa(env, num_episodes, alpha, gamma, epsilon)
            
            eval_rewards = eval_policy(env, q_table, num_eval_episodes)
            results[(alpha, epsilon)] = eval_rewards
    return results

def plot_results(results, alg):
    plt.figure()
    for (alpha, epsilon), rewards in results.items():
        plt.plot(rewards, label=f'α={alpha}, ε={epsilon}')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'{alg} Total Reward over Episodes')
    plt.legend()

    # save png files
    filename = f"{alg.lower()}_windy_cliff_hyperparameters.png"
    plt.savefig(filename)
    plt.show()

# Run experiments with different hyperparameters and visualize the results
# You should generate two plots:
# For each plot, use at least 2 different values for α and 2 different values for ε

# 1. Total reward over episodes for different α and ε values for Q-learning
q_r = run_exp('q_learning', num_episodes=500, gamma=0.99)
plot_results(q_r, 'q_learning')

# 2. Total reward over episodes for different α and ε values for SARSA
sarsa_r = run_exp('sarsa', num_episodes=500, gamma=0.99)
plot_results(sarsa_r, 'sarsa')