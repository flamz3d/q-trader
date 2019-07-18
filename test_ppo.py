from agent.ppo import PPOAgent
from test_environment import TestEnvironment
import sys
import os
import argparse

parser = argparse.ArgumentParser(description='test with ppo.')
parser.add_argument('--stock_name', dest='stock_name', required=True, help='name of stock')
parser.add_argument('--window', dest='window', default=10, help='number of candles in window')
parser.add_argument('--model_file', dest='model_file', required=True, help='checkpoint file relative to models directory')
args = parser.parse_args()

agent = PPOAgent(TestEnvironment(args.window, args.stock_name, 1000), args.model_file)
agent.test()
