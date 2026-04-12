# Q-Trade: Visualizing Algorithmic Swing Trading 📈🤖

An interactive Reinforcement Learning project where an AI agent learns to autonomously swing trade on a scrolling price chart. Built with Python, Gymnasium, and Pygame, this project visualizes the learning process of an algorithmic trader attempting to maximize profit by buying low and selling high.

## 🧠 The Concept

The environment generates a synthetic price curve (a sine wave augmented with Gaussian noise). The RL agent observes this scrolling chart with a starting cash balance and must learn the optimal times to enter and exit the market. 

* **Green Triangles:** Indicate a "Buy" action.
* **Red Triangles:** Indicate a "Sell" action.
* **Goal:** Maximize total account balance while navigating market volatility.

## ⚙️ Environment Details (Gymnasium)

* **Action Space:** `Discrete(3)`
    * `0`: Hold
    * `1`: Buy
    * `2`: Sell
* **State Space:** A continuous array containing:
    * Current Price
    * Previous Day's Price
    * Current Inventory (Binary: `1` if holding the stock, `0` if not)
* **Reward Function:** * Positive reward: Realized profit upon executing a successful "Sell".
    * Negative reward: Loss realized upon a bad "Sell".
    * Step penalty: A small negative penalty for holding cash too long without trading to encourage market participation.

## 🛠️ Tech Stack & Requirements

* **Language:** Python 3.9+
* **RL Environment:** `gymnasium`
* **Visualization:** `pygame`
* **Math/Data:** `numpy`
* **Deep Learning (Optional for DQN):** `torch` (Optimized for Apple Silicon / `mps` acceleration if training on a MacBook Air M1).

## 🚀 Installation & Setup

We recommend using Miniconda to manage dependencies and keep your base system clean.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/Q-Trade-RL.git](https://github.com/yourusername/Q-Trade-RL.git)
   cd Q-Trade-RL