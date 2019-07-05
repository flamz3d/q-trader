from agent.agent import Agent
from agent.ppo import PPOAgent
from functions import *
import sys
import os
from tensorboardX import SummaryWriter

if len(sys.argv) < 4:
	print ("Usage: python train.py [stock] [window] [episodes] ([load_model])")
	exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
model_to_load = ""
if len(sys.argv)>4:
	model_to_load = sys.argv[4]

agent = Agent(window_size, model_name = model_to_load)

data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32
total_overall_trades = 0
writer = SummaryWriter('tensorboard/' + stock_name)

for e in range(episode_count + 1):
	state = getState(data, 0, window_size + 1)
	if (e>0):
		print("Trades per episodes:", total_overall_trades / e)

	total_profit = 0
	agent.inventory = []
	bah_buy_price = data[0]
	total_trades = 0
	total_successfull_trades = 0

	for t in range(l):
		action = agent.act(state)

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0

		bah_profit = data[t] - bah_buy_price
		episode_title = "episode " + str(e) + "/" + str(episode_count)
		episode_progress = "%{:.2f}".format((t / l) * 100.0)
		if action == 1: # buy
			agent.inventory.append(data[t])
			#print ("Buy: " + formatPrice(data[t]))
			total_trades += 1
			if (total_trades % 10 == 0):
				print(episode_title, episode_progress, "Trades: ", total_trades, "Successful:", total_successfull_trades, "Ratio:", total_successfull_trades / total_trades, " | Profit: RL(" + formatPrice(total_profit) + ") BAH(" + formatPrice(bah_profit) + ")")
		
		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			if (reward>0):
				total_successfull_trades += 1
			total_profit += data[t] - bought_price
			#print ("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price) + " | Profit: RL(" + formatPrice(total_profit) + ") BAH(" + formatPrice(bah_profit) + ")")

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print ("--------------------------------")
			print ("Total Profit: " + formatPrice(total_profit))
			print ("--------------------------------")

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	total_overall_trades += total_trades
	writer.add_scalar("trade accuracy", total_successfull_trades / total_trades, e)
	writer.add_scalar("total trades", total_trades, e)

	if e % 10 == 0:
		if (not os.path.isdir("models")):
			try:  
				os.mkdir("models")
			except:
				pass
		agent.model.save("models/model_ep" + str(e))
