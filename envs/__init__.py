from gymnasium.envs.registration import register

register(
    id='QTrade-v0',
    entry_point='envs.trading_env:TradingEnv',
    max_episode_steps=1000,
)

register(
    id='RealQTrade-v0',
    entry_point='envs.real_trading_env:RealTradingEnv',
    max_episode_steps=1500, # Assuming ~5 years of daily trading data (approx 1250 days)
)