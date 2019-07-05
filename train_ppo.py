from agent.ppo import PPOAgent
from functions import *
import sys
import os

if len(sys.argv) != 4:
	print ("Usage: python train.py [stock] [window] [episodes]")
	exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

NUM_ACTIONS = 3

class StockPredict:
	def __init__(self, window_size, stock_name):
		self.name = stock_name
		self.episode_length = 1000
		self.window_size = window_size
		self.data = getStockDataVec(stock_name)
		self.pandas_data = getPandasDataVec(stock_name)
		self.batch_size = 32
		self.reset()
    
	def reset(self):
		self.action_space = np.zeros(NUM_ACTIONS)
		self.observation_space = getState(self.data, 0, self.window_size + 1)[0]
		self.current_lesson = 0
		self.l = len(self.data) - 1
		self.total_profit = 0
		self.inventory = []
		self.bought_price = 0
		self.memory = []
		self.t = 0
		return self.observation_space

	def step(self, action):
		#observation, reward, done, info = self.env.step(action)
		# sit
		self.observation_space = getState(self.data,  self.t + 1, self.window_size + 1)[0]
		reward = 0

		if action == 1: # buy
			self.inventory.append(self.data[self.t])
			print("Buy: " + formatPrice(self.data[self.t]))
		elif action == 2 and len(self.inventory) > 0: # sell
			self.bought_price = self.inventory.pop(0)
			reward = max(self.data[self.t] - self.bought_price, 0)
			self.total_profit += self.data[self.t] - self.bought_price
			print("Sell: " + formatPrice(self.data[self.t]) + " | Profit: " + formatPrice(self.data[self.t] - self.bought_price))

		done = True if self.t == self.l - 1 else False
		#self.memory.append((state, action, reward, next_state, done))
		#state = next_state
		self.current_lesson += 1
		self.t += 1
		return self.observation_space, reward, done, 0

agent = PPOAgent(StockPredict(window_size, stock_name))
agent.run()
