from agent.ppo import PPOAgent
from stock_market_environment import StockPredict
import sys
import os

if len(sys.argv) != 4:
	print ("Usage: python train.py [stock] [window] [episodes]")
	exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

if __name__ == '__main__':
	agent = PPOAgent(StockPredict(window_size, stock_name, episode_count))
	agent.run(episode_count)
