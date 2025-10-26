# melon_game_Auto_1.0


# Fully Automated Melon Game

## Features and Structure of These Scripts

This project provides a physics-based puzzle game, similar to Tetris and Shogi, where you drop fruits and merge identical ones to aim for the highest possible score. A dedicated AI automatically plays, learns, and executes the game.
It includes a "Training Mode" and a "Play Mode".


### Basic Game Rules
- The next fruit is supplied to an arm at the top of the screen and can be dropped into one of 10 divided positions.
- When fruits of the same level come into contact, they merge into a fruit of the next level.
- Points are awarded upon merging.
- The game ends when the fruits stack up to the red game-over line at the top of the screen.

### Fruit Evolution Order
```
Cherry → Strawberry → Grape → Hassaku Orange → Orange
→ Apple → Pear → Peach → Pineapple → Melon → Watermelon
```

### Balance Design
- **Weight Settings**: Smaller fruits have more mass.
- **Spawn Probability**: A cherry is more likely to appear again immediately after a cherry (60% chance for a consecutive cherry).
- **Bonus**: A special bonus is awarded for merging two watermelons.
There are also other secret adjustments.

---

## 🤖 AI Implementation Features

This project utilizes **Deep Neural Network** and is an experimental implementation of an advanced AI agent optimized for MelonGame, combining cutting-edge deep reinforcement learning techniques.
The engineering difficulty is challenging for a hobby-level project, but it is technically feasible with the performance of a personal computer.

### Architecture Overview

**Training Mode**: Uses an AI with a **Dueling Double DQN + Prioritized Experience Replay (D3QN+PER)** architecture.
It captures what is happening on the screen as an image and sends this information to the AI.

```
Input Layer
├─ CNN Path: Board Grid (20×35×2) → 3x Convolutional Layers → Feature Extraction
└─ Dense Path: Metadata Vector (Arm Position, Next Fruit Info)
            ↓
         Feature Integration
            ↓
      Dueling Architecture
      ├─ Value Stream: State Value V(s)
      └─ Advantage Stream: Action Advantage A(s,a)
            ↓
      Q-Value Output: Q(s,a) = V(s) + [A(s,a) - mean(A)]
```

**Play Mode**: Uses the DQN mentioned above as an agent and loads pre-saved weight files if they exist.
When a difficult board state is detected, it initiates a "Monte Carlo Tree Search (MCTS)". The move proposed by the existing AI is sent to MCTS, which then selects a more promising move if one is found, thus avoiding dangerous board states.
This achieves powerful gameplay by combining the "intuition" of deep reinforcement learning with the "reading" of MCTS.

---

## How to Use

### Environment Requirements
Create a venv virtual environment and install the necessary libraries within it.

```bash
pip install pygame Box2D numpy tensorflow
```

### Training Mode

```bash
python melon_game_RL6_beta.py
→ Enter 'train' at the command prompt to start.

# Key Parameters:
# - EPISODES: 1000 (Adjustable, but this is a reasonable initial setting)
# - BATCH_SIZE: 64
# - PHYSICS_ACCELERATION: 24 (Speed of physics simulation)
```

**Training Progress**:
- Runs for 1000 episodes (adjustable).
- The model is saved every 10 episodes (`d3qn_per_model.h5`).

### Play Mode

```bash
python melon_game_RL6_beta.py
# Enter 'play' at the command prompt.

# Adjust MCTS simulation count:
NUM_SIMULATIONS = 30  # Fast (Recommended)
NUM_SIMULATIONS = 200 # Standard
NUM_SIMULATIONS = 500 # Powerful (Long thinking time)
```

Note: A higher simulation count is more powerful but increases the thinking time.

---
Additionally, there is a script prepared to read the numerical logs of the training progress, convert them into graphs, and visualize them.
`visualize_log2.py`
However, executing this script requires the presence of the file "training_log.txt" containing the numerical data series.

## Hyperparameters

### D3QN+PER Settings

| Parameter | Value | Description |
|---|---|---|
| `gamma` | 0.99 | Discount factor (importance of future rewards) |
| `epsilon_decay` | 0.998 | Epsilon decay rate |
| `learning_rate` | 0.00005 | Learning rate (Adam) |
| `alpha` (PER) | 0.6 | Priority intensity |
| `beta` (PER) | 0.4→1.0 | Importance sampling correction |
| `buffer_size` | 50000 | Replay buffer size |

### MCTS Settings

| Parameter | Value | Description |
|---|---|---|
| `num_simulations` | 30-500 | Number of simulations per move |
| `c_puct` | 1.5 | Controls the breadth of the search |
| `temperature` | 0.5 | Sharpness of the policy probability |

---

## Customization List

### Examples of Adjustable Parameters

```python
# Fruit spawn probabilities
normal_spawn_weights = [0.10, 0.25, 0.30, 0.25, 0.10]  # Normal
cherry_followup_weights = [0.60, 0.15, 0.10, 0.10, 0.05]  # After a cherry

# Physics simulation speed
PHYSICS_ACCELERATION = 24  # For faster training
```

### Neural Network Structure

```python
# CNN Layers
# Dense Layers
# This part is complex, so it is omitted.
```

---

## Future Improvements

1. Resolve freezing issues, speed up training mode, and add a conditional branch to select a rendering OFF mode.
2. Lightweight state representation: Speed up by reducing the grid resolution.
3. Advanced mathematics will be applied to the evaluation of the board state.
4. the AI ​​will be retrained to improve the accuracy of the robotic arm's movements and to adapt to the new rules.

---

## License
This project is open source. However, since similar game systems and designs have existed in the past, please use it only as an experimental example of reinforcement learning.

---

## Acknowledgements

- Box2D: Physics engine (C++)
- TensorFlow/Keras: Deep learning framework (Google)
- Pygame: Graphics library

---

## To Developers and Players:
This code is designed as a practical example of reinforcement learning for educational purposes. The implementation of each algorithm prioritizes clarity and is not optimized for production use.
