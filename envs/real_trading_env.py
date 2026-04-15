import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf

class RealTradingEnv(gym.Env):
    """
    A custom trading environment that uses real-world stock market data via yfinance.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, ticker="BTC-USD", start_date="2018-01-01", end_date="2023-01-01"):
        super().__init__()
        
        self.ticker = ticker
        
        # 1. Fetch Real Data
        print(f"Fetching historical data for {ticker} from {start_date} to {end_date}...")
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=True)
        
        if stock_data.empty:
            raise ValueError(f"No data found for ticker {ticker}.")
            
        # We will use the 'Close' price for trading
        # yfinance sometimes returns a DataFrame with MultiIndex columns (e.g. ('Close', 'AAPL'))
        # This handles both a single level and multi level columns
        if isinstance(stock_data.columns, pd.MultiIndex):
            close_prices = stock_data['Close'][ticker].values
        else:
            close_prices = stock_data['Close'].values

        self.price_data = close_prices.astype(np.float32)
        self.data_length = len(self.price_data)
        
        print(f"Loaded {self.data_length} days of data.")
        
        # 2. Define Action Space
        # 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)
        
        # 3. Define Observation Space
        # State: [Current Price, Previous Price, Inventory (0.0 or 1.0)]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32), 
            high=np.array([np.inf, np.inf, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        # Episode Tracking Variables
        self.current_step = 0
        self.inventory = 0
        self.buy_price = 0.0
        
        # Hyperparameters for reward shaping
        # Since AAPL prices might be $150, a -0.05 penalty is too small. 
        # Using a relative penalty or slightly higher absolute one.
        # But for stability we will keep them as small biases.
        self.inactivity_penalty = -0.05  
        self.invalid_action_penalty = -0.1

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state for a new episode.
        """
        super().reset(seed=seed)
        
        # Start at index 1 so we always have a "previous day" price
        self.current_step = 1 
        self.inventory = 0
        self.buy_price = 0.0
        
        return self._get_obs(), {}

    def _get_obs(self):
        """
        Helper method to construct the state array.
        """
        current_price = self.price_data[self.current_step]
        prev_price = self.price_data[self.current_step - 1]
        return np.array([current_price, prev_price, self.inventory], dtype=np.float32)

    def step(self, action):
        """
        Executes one time step within the environment.
        """
        current_price = self.price_data[self.current_step]
        reward = 0.0
        
        # Process Actions
        if action == 1:  # BUY
            if self.inventory == 0:
                self.inventory = 1
                self.buy_price = current_price
            else:
                reward = self.invalid_action_penalty
                
        elif action == 2:  # SELL
            if self.inventory == 1:
                self.inventory = 0
                profit = current_price - self.buy_price
                reward = profit
            else:
                reward = self.invalid_action_penalty
                
        elif action == 0:  # HOLD
            if self.inventory == 0:
                # Apply penalty to encourage the agent to participate in the market
                reward = self.inactivity_penalty
                
        # Advance the environment
        self.current_step += 1
        
        # Check if we have reached the end of the data
        terminated = self.current_step >= self.data_length - 1
        truncated = False
        
        # Clean up: Force a sell at the end of the episode if still holding to realize final P/L
        if terminated and self.inventory == 1:
            final_profit = self.price_data[self.current_step] - self.buy_price
            reward += final_profit
            self.inventory = 0
            
        return self._get_obs(), reward, terminated, truncated, {}
