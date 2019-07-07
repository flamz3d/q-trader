from functions import *
import sys
import os
from tqdm import tqdm

NUM_ACTIONS = 3

class StockPredict:
    def __init__(self, window_size, stock_name, episode_count):
        self.name = stock_name
        self.episode_count = episode_count
        self.window_size = window_size
        self.data = getStockDataVec(stock_name)
        self.pandas_data = getPandasDataVec(stock_name)
        if (len(self.data) != len(self.pandas_data.count())):
            print("pandas dataframe and data count does not match, something is wrong with the dataset")
            exit();
        
        self.iterations = 0
        self.batch_size = 32
        self.total_trades = 0
        self.total_successfull_trades = 0
        self.pbar = tqdm(total=100)
        self.reset()
    
    def reset(self):
        self.action_space = np.zeros(NUM_ACTIONS)
        #self.observation_space = getState(self.data, 0, self.window_size + 1)[0]
        self.observation_space = getIndicators(self.pandas_data, 0, self.window_size + 1)
        self.current_lesson = 0
        self.l = len(self.data) - 1
        self.total_profit = 0
        self.inventory = []
        self.bought_price = 0
        self.memory = []
        self.t = 0
        self.pbar.close()
        self.pbar = tqdm(total=100, desc = "episode " + str(self.iterations) + "/" + str(self.episode_count))
        self.iterations += 1
        return self.observation_space

    def stats(self):
        accuracy = 0
        if (self.total_trades>0):
            accuracy = self.total_successfull_trades / self.total_trades
        stat = {"total_trades": self.total_trades, "trade_accuracy": accuracy}
        print(stat)
        self.total_trades = 0
        self.total_successfull_trades = 0
        return stat

    def step(self, action):
        #observation, reward, done, info = self.env.step(action)
        # sit
        #self.observation_space = getState(self.data,  self.t + 1, self.window_size + 1)[0]
        self.observation_space = getIndicators(self.pandas_data, self.t + 1, self.window_size + 1)
        
        reward = 0

        if action == 1: # buy
            self.inventory.append(self.data[self.t])
            #print("Buy: " + formatPrice(self.data[self.t]))
        elif action == 2 and len(self.inventory) > 0: # sell
            self.bought_price = self.inventory.pop(0)
            reward = max(self.data[self.t] - self.bought_price, 0)
            self.total_profit += self.data[self.t] - self.bought_price
            self.total_trades += 1
            if (reward>0):
                self.total_successfull_trades += 1
            #print("Sell: " + formatPrice(self.data[self.t]) + " | Profit: " + formatPrice(self.data[self.t] - self.bought_price))

        done = True if self.t == self.l - 1 else False
        #self.memory.append((state, action, reward, next_state, done))
        #state = next_state
        self.current_lesson += 1
        self.t += 1
        #print("batch:", "% {:.2f}".format((self.t / len(self.data))*100))
        progress = (int)((self.t / len(self.data))*100)
        
        self.pbar.n = progress
        self.pbar.last_print_n = progress
        self.pbar.update()
        return self.observation_space, reward, done, 0