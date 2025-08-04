# YAML2Gym 🎮

> **Natural Language → YAML → RL Environment**

> *"I was tired of rewriting environment code for every new RL agent. Inspired by [pyRDDL](https://arxiv.org/abs/2109.14799) and Microsoft's TextWorld, I wanted something simpler - a way to quickly prototype grid-based RL environments from natural language descriptions. The result? YAML2Gym: dead simple, YAML-defined environments that work with any Gymnasium-compatible agent."*

YAML2Gym lets you create custom RL environments using simple YAML files. Describe your grid world in plain text, and get a fully interactive Gymnasium environment with Pygame rendering - no Python coding required!

## 🚀 Why YAML2Gym?

### The Problem
When testing RL agents, I kept hitting the same roadblocks:
1. Writing custom environment code for every new idea
2. Hardcoding agent-specific logic into the environment
3. Wasting time on rendering and visualization
4. Needing to modify code for simple parameter changes

### The Solution
YAML2Gym lets you:
- **Prototype Fast**: Go from idea to environment in minutes
- **Stay Flexible**: Change game mechanics without touching Python
- **Test Thoroughly**: Rapidly iterate on environment design
- **Share Easily**: Just share a YAML file, no code needed

## ✨ Features

- **Natural Language to RL**: Describe your world in simple YAML
- **Zero Python Required**: No more `gym.Env` boilerplate
- **Visual Debugging**: Built-in Pygame renderer
- **RL-Ready**: Works with any Gymnasium-compatible library (Stable Baselines, RLlib, etc.)
- **Tweak On The Fly**: Change rewards, layouts, and rules without restarting

## 🚀 60-Second Quickstart

1. Install (Python 3.8+ required):
   ```bash
   pip install pygame gymnasium pyyaml
   git clone https://github.com/yourusername/yaml2gym.git
   cd yaml2gym
   pip install -e .
   ```

2. Run the example:
   ```bash
   python -m yaml2gym.scripts.play_env examples/collect_coins.yaml
   ```
   - Arrow keys to move
   - 1-6 for special actions
   - R to reset, Q to quit

3. Edit the YAML file and see changes instantly!

## 🎮 Quick Start

1. Create a YAML file defining your environment (see `examples/collect_coins.yaml` for a template)

2. Run the environment with the interactive player:
   ```bash
   python -m yaml2gym.scripts.play_env examples/collect_coins.yaml
   ```

3. Use the arrow keys to move, space to stay, R to reset, and Q to quit

## 🎮 Example: From Text to Environment

### 1. Describe Your World
"I want a 10x10 grid where an agent collects apples while avoiding fire. The agent can move in 4 directions. Each apple gives +1 point, and touching fire ends the episode. The goal is to collect all apples."

### 2. Write the YAML
```yaml
name: "Apple Collector"
grid_size: [10, 10]
max_steps: 200
observation_type: "grid"

agent_start: {x: 0, y: 0}

objects:
  - type: "apple"
    positions: [[2,2], [5,7], [8,3]]
    properties:
      collectable: true
      reward: 1.0
      on_collect: "remove_self"

  - type: "fire"
    positions: [[4,4], [4,5], [5,4]]
    properties:
      damage: 1.0
      terminate_on_contact: true

actions:
  - name: "up"
    effect: {type: "move", params: {dx: 0, dy: -1}}
  - name: "down"
    effect: {type: "move", params: {dx: 0, dy: 1}}
  - name: "left"
    effect: {type: "move", params: {dx: -1, dy: 0}}
  - name: "right"
    effect: {type: "move", params: {dx: 1, dy: 0}}

terminations:
  - condition: "all_collected(apple)"
    done: true
    success: true
```

### 3. Train Your Agent
```python
from stable_baselines3 import PPO
from yaml2gym.env import GenericGridEnv

env = GenericGridEnv("apple_collector.yaml")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

## 📝 YAML Specification

### Basic Structure

```yaml
name: "My Environment"
grid_size: [width, height]  # Grid dimensions
max_steps: 1000            # Maximum number of steps per episode
observation_type: "grid"   # "grid", "dict", or "flat"

# Starting position of the agent
agent_start:
  x: 0
  y: 0

# Objects in the environment
objects:
  - type: "object_type"
    positions:
      - {x: 1, y: 1}
      - {x: 2, y: 3}
    properties:
      key: value

# Available actions
actions:
  - name: "action_name"
    effect: "effect_name"

# Reward rules
rewards:
  - condition: "condition_expression"
    value: 1.0

# Termination conditions
terminations:
  - condition: "condition_expression"
    done: true
    success: true
```

### Example: Coin Collection

See `examples/collect_coins.yaml` for a complete example with coins to collect, lava to avoid, and a goal to reach.

## 🤖 Training Agents

You can use YAML2Gym environments with any reinforcement learning library that supports Gymnasium. Here's a quick example using Stable Baselines3:

```python
from stable_baselines3 import PPO
from yaml2gym.env import GenericGridEnv

# Create the environment
env = GenericGridEnv("path/to/your/env.yaml")

# Create and train a PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Test the trained agent
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
```

## 📚 Examples

Check out the `examples/` directory for more environment definitions:

- `collect_coins.yaml`: A simple coin collection game with obstacles
- (More examples coming soon!)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
