import argparse
import gymnasium as gym
import os
import time

# Fix for Intel OMP library collision commonly found in conda envs
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Importing this registers the custom environment
import envs 
from agent import QLearningAgent, DQNAgent
from utils import TradingVisualizer

def train(episodes=500, agent_type='qlearning', env_type='synthetic', ticker='BTC-USD', device=None):
    """
    Trains the agent on the chosen trading environment.
    """
    print(f"Initializing Training Environment: {env_type.upper()}")
    if env_type == 'synthetic':
        env = gym.make('QTrade-v0')
    else:
        env = gym.make('RealQTrade-v0', ticker=ticker)
        
    if agent_type == 'qlearning':
        agent = QLearningAgent(action_space_size=env.action_space.n)
    else:
        agent = DQNAgent(state_dim=env.observation_space.shape[0], action_space_size=env.action_space.n, device=device)
    
    from tqdm import tqdm
    print(f"Starting training for {episodes} episodes using {agent_type.upper()}...")
    
    pbar = tqdm(range(episodes), desc="Training Agent")
    for episode in pbar:
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(obs, training=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update the agent
            if agent_type == 'qlearning':
                agent.update(obs, action, reward, next_obs)
            else:
                agent.update(obs, action, reward, next_obs, done)
            
            obs = next_obs
            total_reward += reward
            
        # Decay exploration rate at the end of each episode
        agent.decay_epsilon()
        
        # Update progress bar dynamically
        if (episode + 1) % 10 == 0:
            pbar.set_postfix({'Reward': f"{total_reward:.2f}", 'Epsilon': f"{agent.epsilon:.3f}"})
            
    # Save the trained agent
    save_dir = 'models/saved_agents'
    os.makedirs(save_dir, exist_ok=True)
    ext = 'pth' if agent_type == 'dqn' else 'pkl'
    save_path = os.path.join(save_dir, f'{agent_type}_{env_type}_model.{ext}')
    agent.save(save_path)
    print(f"\n[DONE] Training complete. Model saved to {save_path}")
    env.close()

def test(agent_type='qlearning', env_type='synthetic', ticker='BTC-USD', device=None):
    """
    Evaluates the trained agent and visualizes its trades using Pygame.
    """
    print(f"Initializing Evaluation Environment: {env_type.upper()}")
    if env_type == 'synthetic':
        env = gym.make('QTrade-v0')
    else:
        env = gym.make('RealQTrade-v0', ticker=ticker)
        
    if agent_type == 'qlearning':
        agent = QLearningAgent(action_space_size=env.action_space.n)
    else:
        agent = DQNAgent(state_dim=env.observation_space.shape[0], action_space_size=env.action_space.n, device=device)
    
    ext = 'pth' if agent_type == 'dqn' else 'pkl'
    model_path = f'models/saved_agents/{agent_type}_{env_type}_model.{ext}'
    
    # Fallback to general q_table.pkl if old setup exists and requested qlearning + synthetic
    if not os.path.exists(model_path) and agent_type == 'qlearning' and env_type == 'synthetic':
        fallback_path = 'models/saved_agents/q_table.pkl'
        if os.path.exists(fallback_path):
            model_path = fallback_path
            
    if not os.path.exists(model_path):
        print(f"[ERROR] Could not find model at {model_path}. Please run training first.")
        return
        
    # Load trained model (automatically minimizes epsilon for exploitation)
    agent.load(model_path)
    print(f"[DONE] Model loaded successfully from {model_path}. Starting live evaluation...")
    
    visualizer = TradingVisualizer()
    
    obs, _ = env.reset()
    done = False
    total_profit = 0.0
    
    # Pad actions history with a 0 so indices match the price data (which starts at step 1)
    actions_history = [0] 
    
    while not done:
        # Get action from the trained policy
        action = agent.get_action(obs, training=False)
        actions_history.append(action)
        
        # Step the environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Calculate visual profit (only realized profits from selling)
        if action == 2 and obs[2] == 1: # Sold while holding inventory
            total_profit += reward
            
        # Extract data for the visualizer using Gymnasium's unwrapped property
        current_step = env.unwrapped.current_step
        price_data = env.unwrapped.price_data
        inventory = next_obs[2]
        
        # Render the Pygame frame
        is_running = visualizer.render(
            price_data=price_data, 
            current_step=current_step, 
            actions_history=actions_history, 
            total_profit=total_profit, 
            inventory=inventory
        )
        
        if not is_running:
            print("\nUser closed the visualization window.")
            break
            
        obs = next_obs
        
        # Small delay so the user can actually watch the trades happen
        time.sleep(0.02)
        
    print(f"\n[FINISHED] Evaluation finished. Final Realized Profit: ${total_profit:.2f}")
    
    # Keep window open for a few seconds at the end before closing
    time.sleep(3) 
    visualizer.close()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-Trade Algorithmic Swing Trading Agent")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help="Mode to run: 'train' to learn, 'test' to visualize")
    parser.add_argument('--episodes', type=int, default=1000,
                        help="Number of episodes to train the agent")
    parser.add_argument('--agent', type=str, choices=['qlearning', 'dqn'], default='qlearning',
                        help="The type of agent to use.")
    parser.add_argument('--env', type=str, choices=['synthetic', 'real'], default='synthetic',
                        help="The environment to use (synthetic sine wave or real yfinance data).")
    parser.add_argument('--ticker', type=str, default='BTC-USD',
                        help="The stock ticker to fetch when using the real environment.")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps', 'auto'], default='auto',
                        help="Force execution on specific hardware (CPU is inherently faster for very small networks).")

    args = parser.parse_args()
    
    device_arg = None if args.device == 'auto' else args.device
    
    if args.mode == 'train':
        train(episodes=args.episodes, agent_type=args.agent, env_type=args.env, ticker=args.ticker, device=device_arg)
    elif args.mode == 'test':
        test(agent_type=args.agent, env_type=args.env, ticker=args.ticker, device=device_arg)