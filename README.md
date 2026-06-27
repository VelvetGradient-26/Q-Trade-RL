# Q-Trade: Algorithmic Trading with Reinforcement Learning

An interactive Reinforcement Learning (RL) project where an AI agent learns to autonomously swing trade on a scrolling price chart. Built with Python, Gymnasium, and Pygame, this project visualizes the learning process of an algorithmic trader attempting to maximize profit by buying low and selling high.

** New in this version:** A fully revamped, professional-grade Pygame visualization platform with dynamic axes, real-time price tracking, area fills, and actual dates drawn directly from historical stock data via `yfinance`.

##  The Concept

The environment generates a synthetic price curve (a sine wave augmented with Gaussian noise) or fetches real stock market historical data. The RL agent observes this scrolling chart with a starting cash balance and must learn the optimal times to enter and exit the market. 
- **Green Triangles (Buy):** Indicate the agent entering a "Long" position and buying the asset.
- **Red Triangles (Sell):** Indicate the agent exiting the position and converting back to cash.
- **Goal:** Maximize total account balance (P&L) while navigating market volatility.

## Environment Mechanics (Gymnasium)

- **Action Space:** `Discrete(3)`
  - `0`: Hold
  - `1`: Buy
  - `2`: Sell
- **State Space:** Continuous array containing:
  - Current Price
  - Previous Day's Price
  - Current Inventory (Binary: `1` if holding the stock, `0` if holding cash)
- **Reward Function:**
  - Positive reward: Realized profit upon executing a successful "Sell".
  - Negative reward: Realized loss upon a bad "Sell", or a penalty for illegal actions (e.g., selling when you have zero inventory).
  - Step penalty: A small negative penalty for holding cash too long without trading to encourage market participation.

## AI Agents

This repository supports two distinct types of reinforcement learning agents:

1. **Q-Learning Agent (`agent/q_learning.py`):** 
   A tabular Q-Learning agent that discretizes the continuous price space into logical trend "bins" and updates a Q-Table using the Bellman Equation. Great for understanding the fundamentals of RL in straightforward, synthetic environments.
2. **Deep Q-Network - DQN (`agent/dqn.py`):** 
   A deep reinforcement learning agent built with PyTorch using a Neural Network policy, Replay Buffer, and target network. This agent can natively handle continuous observation spaces without manual discretization. It automatically utilizes hardware acceleration (`mps` for Apple Silicon, `cuda` for NVIDIA, or `cpu`).

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Q-Trade-RL.git
   cd Q-Trade-RL
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment or a Conda environment.
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have PyTorch installed appropriately for your system. For Mac M-series, install the ARM64 version of Torch).*

## Comprehensive Instructions

The central hub for operating the AI is `main.py`. The project naturally splits into two phases: **Training** the agent, and **Testing** (visualizing) its performance.

### 1. Training the Agent

Before the agent can successfully trade, it needs to learn the market patterns by running through many back-to-back iterations (episodes).

**Example A: Train a Q-Learning Agent on Synthetic Data**
```bash
python main.py --mode train --agent qlearning --env synthetic --episodes 1000
```
This is the fastest combination. The synthetic environment is a predictable curve, allowing the Q-table to quickly map out the optimal buying and selling points.

**Example B: Train a DQN Agent on Real Stock Data (e.g., Apple)**
```bash
python main.py --mode train --agent dqn --env real --ticker BTC-USD --episodes 500
```
This runs the PyTorch neural network against real historical AAPL stock prices. Watch the console to see the real-time epsilon decay (exploration rate dropping) as the loss stabilizes.

*Note: All trained models are automatically saved into the `models/saved_agents/` directory upon completion.*

### 2. Evaluating and Visualizing

Once your agent is trained, you can evaluate its performance and watch it trade live via the professional Pygame Terminal. 

**Test the Q-Learning Agent (Synthetic Market):**
```bash
python main.py --mode test --agent qlearning --env synthetic
```

**Test the DQN Agent (Real Market):**
```bash
python main.py --mode test --agent dqn --env real --ticker BTC-USD
```

When you launch `test` mode, a **Trading AI Desk** window will appear. It visualizes:
- **X-Axis:** Historical timestamps (when using real data) or Step numbers.
- **Y-Axis:** Dynamic asset price scaling.
- **HUD (Heads Up Display):** Total Profit & Loss and current holding status.
- **Graph:** Blue area-filled chart with an active pricing dot and real-time buy/sell action triangles.

### Command Line Arguments Reference

| Argument | Options | Default | Description |
| :--- | :--- | :--- | :--- |
| `--mode` | `train`, `test` | `train` | Determines whether the agent is learning new policies or visualizing existing ones. |
| `--agent` | `qlearning`, `dqn` | `qlearning` | Selects between Tabular Q-Learning and PyTorch DQN. |
| `--env` | `synthetic`, `real` | `synthetic` | Selects between the generated sine wave and historical stock data from `yfinance`. |
| `--ticker`| Any standard ticker (e.g., `AAPL`, `BTC-USD`) | `BTC-USD` | The asset to fetch from Yahoo Finance. Only applicable if `--env real`. |
| `--episodes` | Integer (e.g., `1000`) | `1000` | The number of full passes through the data. Higher = smarter agent, but longer training time. |
| `--device` | `cpu`, `cuda`, `mps`, `auto` | `auto` | Force PyTorch to use a specific hardware device for tensor operations. |
