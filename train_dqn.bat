@echo off
echo Starting PyTorch DQN Training...
python main.py --mode train --agent dqn --env synthetic --episodes 1500 --device auto
echo Training finished.
pause
