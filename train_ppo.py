from agent.ppo import PPOAgent
from stock_market_environment import StockPredict
import sys
import os

if len(sys.argv) < 4:
	print ("Usage: python train.py [stock] [window] [episodes]")
	exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
model_file = None

if len(sys.argv) == 5:
    model_file = sys.argv[4]

if __name__ == '__main__':
    if model_file is not None:
        model_file_path = os.path.dirname(os.path.abspath(model_file))
        base_filename = os.path.basename(os.path.abspath(model_file))
        base_filename = os.path.splitext(base_filename)[0]
        index = base_filename.rfind("_")
        episode_to_load = base_filename[index+1:]
        critic_model_file = os.path.join(model_file_path, "model_critic_" + stock_name + "_" + episode_to_load + ".h5")
        actor_model_file = os.path.join(model_file_path, "model_actor_" + stock_name + "_" + episode_to_load + ".h5")
    
    agent = PPOAgent(StockPredict(window_size, stock_name, episode_count), actor_model_file, critic_model_file)
    agent.run(episode_count)
