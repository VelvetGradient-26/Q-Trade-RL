import numpy as np
import pickle
import os

class QLearningAgent:
    """
    A Tabular Q-Learning Agent.
    Uses an epsilon-greedy strategy for exploration and updates Q-values using the Bellman equation.
    """
    def __init__(self, action_space_size=3, lr=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.action_space_size = action_space_size
        self.lr = lr                    # Learning rate (alpha)
        self.gamma = gamma              # Discount factor
        self.epsilon = epsilon          # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: Dictionary mapping discrete state tuples to numpy arrays of Q-values
        self.q_table = {}

    def _discretize_state(self, obs):
        """
        Converts continuous observation space into a discrete state for the Q-Table.
        Instead of raw prices, we look at the trend (price difference).
        """
        current_price, prev_price, inventory = obs
        
        # Calculate price change
        price_diff = current_price - prev_price
        
        # Discretize the price difference into 5 logical bins
        if price_diff < -0.5: 
            trend = 0     # Strong Downward Trend
        elif price_diff < 0: 
            trend = 1     # Weak Downward Trend
        elif price_diff == 0: 
            trend = 2     # Flat
        elif price_diff > 0.5: 
            trend = 3     # Strong Upward Trend
        else: 
            trend = 4     # Weak Upward Trend
            
        # Discretize the current price into bins
        if current_price < 46: 
            price_level = 0
        elif current_price < 49: 
            price_level = 1
        elif current_price < 51: 
            price_level = 2
        elif current_price < 54: 
            price_level = 3
        else: 
            price_level = 4
            
        # The state is a tuple: (Market Trend, Price Level, Do I own the stock?)
        return (trend, price_level, int(inventory))

    def get_action(self, obs, training=True):
        """
        Selects an action using the epsilon-greedy policy.
        """
        state = self._discretize_state(obs)
        
        # Initialize state in Q-table if we haven't seen it before
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_size)
            
        # Epsilon-greedy action selection
        if training and np.random.rand() < self.epsilon:
            # Explore: pick a random action
            return np.random.randint(self.action_space_size) 
        else:
            # Exploit: pick the best known action
            return np.argmax(self.q_table[state]) 

    def update(self, obs, action, reward, next_obs):
        """
        Updates the Q-Table using the Bellman Equation.
        """
        state = self._discretize_state(obs)
        next_state = self._discretize_state(next_obs)
        
        # Ensure states exist in Q-table
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_size)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_space_size)
            
        # Bellman Equation
        # Q(s, a) = Q(s, a) + lr * [Reward + gamma * max(Q(s', a')) - Q(s, a)]
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        
        self.q_table[state][action] += self.lr * td_error

    def decay_epsilon(self):
        """
        Reduces the exploration rate over time.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Saves the Q-table to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
            
    def load(self, filepath):
        """Loads a Q-table from a file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.q_table = pickle.load(f)
            # When loading a trained model, minimize exploration
            self.epsilon = self.epsilon_min