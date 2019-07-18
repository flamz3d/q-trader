# Initial framework taken from https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py

#things to try
# - 1 LSTM sequence layers instead of 2
# - relu instead of tan

import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Flatten, Dropout
from keras import backend as K
from keras.optimizers import Adam
from tensorboardX import SummaryWriter
import keras.losses

ENV = 'STOCK'
CONTINUOUS = False

EPISODES = 100000

LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS = 10
NOISE = 1.0 # Exploration noise

GAMMA = 0.99

BUFFER_SIZE = 256
BATCH_SIZE = 64
NUM_STATE = 8
HIDDEN_SIZE = 128
NUM_LAYERS = 2
ENTROPY_LOSS = 1e-3
LR = 1e-4 # Lower lr stabilises training greatly
LSTM_SIZE = 32

def exponential_average(old, new, b1):
	return old * b1 + (1-b1) * new


def proximal_policy_optimization_loss(advantage, old_prediction):
	def loss(y_true, y_pred):
		prob = y_true * y_pred
		old_prob = y_true * old_prediction
		r = prob/(old_prob + 1e-10)
		return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
	return loss


def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
	def loss(y_true, y_pred):
		var = K.square(NOISE)
		pi = 3.1415926
		denom = K.sqrt(2 * pi * var)
		prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
		old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

		prob = prob_num/denom
		old_prob = old_prob_num/denom
		r = prob/(old_prob + 1e-10)

		return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage))
	return loss


class PPOAgent:
	def __init__(self, env, actor_model_file = None, critic_model_file = None):
		self.env = env 
		self.episode = 0
		self.observation = self.env.reset()
		self.num_states = len(self.observation)
		self.num_actions = self.env.action_space.size
		self.num_timesteps = len(self.observation)
		self.state_dimensions = len(self.observation[0])
		self.dummy_action, self.dummy_value = np.zeros((1, self.num_actions)), np.zeros((1, 1))

		self.critic = self.build_critic()
		if CONTINUOUS is False:
			self.actor = self.build_actor()
		else:
			self.actor = self.build_actor_continuous()

		if actor_model_file is not None:
			print("loading actor weights from", actor_model_file)
			self.actor.load_weights(actor_model_file)

		if critic_model_file is not None:
			print("loading critic weights from", critic_model_file)
			self.critic.load_weights(critic_model_file)

		self.val = False
		self.reward = []
		self.reward_over_time = []
		self.name = self.get_name()
		self.writer = SummaryWriter(self.name)
		self.gradient_steps = 0

	def get_name(self):
		name = 'AllRuns/'
		if CONTINUOUS is True:
			name += 'continous/'
		else:
			name += 'discrete/'
		name += self.env.name
		return name


	def build_actor(self):
		#if (model_file != None):
			#advantage = Input(shape=(1,))
			#old_prediction = Input(shape=(self.num_actions,))
			#loss = proximal_policy_optimization_loss(
			#			  advantage=advantage,
			#			  old_prediction=old_prediction)
			#model = keras.models.load_model(model_file, custom_objects={'loss':loss})
			#print("LOADED ACTOR MODEL")
			#model.summary()
			#return model

		state_input = Input(shape=(self.num_timesteps, self.state_dimensions))
		advantage = Input(shape=(1,))
		old_prediction = Input(shape=(self.num_actions,))

		lstm_0 = LSTM(LSTM_SIZE, return_sequences=True)(state_input)
		lstm_1 = LSTM(LSTM_SIZE, return_sequences=True)(lstm_0)
		lstm_2 = LSTM(LSTM_SIZE)(lstm_1)
		dropout = Dropout(0.25)(lstm_2)
		x = Dense(HIDDEN_SIZE, activation='tanh')(dropout)
		for _ in range(NUM_LAYERS - 1):
			x = Dense(HIDDEN_SIZE, activation='tanh')(x)

		out_actions = Dense(self.num_actions, activation='softmax', name='output')(x)

		model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
		model.compile(optimizer=Adam(lr=LR),
					  loss=[proximal_policy_optimization_loss(
						  advantage=advantage,
						  old_prediction=old_prediction)])
		print("ACTOR MODEL")
		model.summary()

		return model

	def build_actor_continuous(self):
		state_input = Input(shape=(self.num_states,))
		advantage = Input(shape=(1,))
		old_prediction = Input(shape=(self.num_actions,))

		x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
		for _ in range(NUM_LAYERS - 1):
			x = Dense(HIDDEN_SIZE, activation='tanh')(x)

		out_actions = Dense(self.num_actions, name='output', activation='tanh')(x)

		model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
		model.compile(optimizer=Adam(lr=LR),
					  loss=[proximal_policy_optimization_loss_continuous(
						  advantage=advantage,
						  old_prediction=old_prediction)])
		model.summary()

		return model

	def build_critic(self):

		state_input = Input(shape=(self.num_timesteps, self.state_dimensions))

		lstm_0 = LSTM(LSTM_SIZE, return_sequences=True)(state_input)
		lstm_1 = LSTM(LSTM_SIZE, return_sequences=True)(lstm_0)
		lstm_2 = LSTM(LSTM_SIZE)(lstm_1)
		dropout = Dropout(0.25)(lstm_2)
		x = Dense(HIDDEN_SIZE, activation='tanh')(dropout)
		for _ in range(NUM_LAYERS - 1):
			x = Dense(HIDDEN_SIZE, activation='tanh')(x)

		out_value = Dense(1)(x)

		model = Model(inputs=[state_input], outputs=[out_value])
		model.compile(optimizer=Adam(lr=LR), loss='mse')
		print("CRITIC MODEL")
		model.summary()

		return model

	def reset_env(self):
		self.episode += 1
		if self.episode % 100 == 0:
			self.val = True
		else:
			self.val = False
		self.observation = self.env.reset()
		self.reward = []

	def get_action(self):
		p = self.actor.predict([[self.observation], self.dummy_value, self.dummy_action])
		#p = self.actor.predict(self.observation)
		if self.val is False:
			action = np.random.choice(self.num_actions, p=np.nan_to_num(p[0]))
		else:
			action = np.argmax(p[0])
		action_matrix = np.zeros(self.num_actions)
		action_matrix[action] = 1
		return action, action_matrix, p

	def get_action_continuous(self):
		p = self.actor.predict([self.observation, self.dummy_value, self.dummy_action])
		if self.val is False:
			action = action_matrix = p[0] + np.random.normal(loc=0, scale=NOISE, size=p[0].shape)
		else:
			action = action_matrix = p[0]
		return action, action_matrix, p

	def transform_reward(self):
		if self.val is True:
			self.writer.add_scalar('Val episode reward', np.array(self.reward).sum(), self.episode)
		else:
			self.writer.add_scalar('Episode reward', np.array(self.reward).sum(), self.episode)
		for j in range(len(self.reward) - 2, -1, -1):
			self.reward[j] += self.reward[j + 1] * GAMMA

	def get_batch(self):
		batch = [[], [], [], []]
		tmp_batch = [[], [], []]

		while len(batch[0]) < BUFFER_SIZE:
			if CONTINUOUS is False:
				action, action_matrix, predicted_action = self.get_action()
			else:
				action, action_matrix, predicted_action = self.get_action_continuous()
			observation, reward, done, info = self.env.step(action)
			self.reward.append(reward)

			tmp_batch[0].append(self.observation)
			tmp_batch[1].append(action_matrix)
			tmp_batch[2].append(predicted_action)
			self.observation = observation

			if done:
				self.transform_reward()
				if self.val is False:
					for i in range(len(tmp_batch[0])):
						obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
						r = self.reward[i]
						batch[0].append(obs)
						batch[1].append(action)
						batch[2].append(pred)
						batch[3].append(r)
				tmp_batch = [[], [], []]
				self.reset_env()

		obs, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]), (len(batch[3]), 1))
		pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
		return obs, action, pred, reward

	def test(self):
		print("allo")

	def run(self, episodeCount):
		EPISODES = episodeCount
		while self.episode < EPISODES:
			obs, action, pred, reward = self.get_batch()
			obs, action, pred, reward = obs[:BUFFER_SIZE], action[:BUFFER_SIZE], pred[:BUFFER_SIZE], reward[:BUFFER_SIZE]
			old_prediction = pred
			pred_values = self.critic.predict(obs)

			advantage = reward - pred_values
			# advantage = (advantage - advantage.mean()) / advantage.std()
			actor_loss = self.actor.fit([obs, advantage, old_prediction], [action], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
			critic_loss = self.critic.fit([obs], [reward], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
			self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
			self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)
			data_dict = self.env.stats()
			for key,val in data_dict.items():
				self.writer.add_scalar(key, val, self.gradient_steps)
			self.gradient_steps += 1
			if (self.gradient_steps % 2 == 0):
				self.actor.save_weights('./models/model_actor_' + self.env.name + '_' + str(self.gradient_steps) + '.h5')
				self.critic.save_weights('./models/model_critic_' + self.env.name + '_' + str(self.gradient_steps) + '.h5')

class Env:
	def __init__(self):
		self.episode_length = 1000
		self.reset()
	
	def reset(self):
		self.action_space = np.zeros(self.num_actions)
		self.observation_space = np.zeros(self.num_states)
		self.current_lesson = 0
		return self.observation_space

	def step(self, action):
		#observation, reward, done, info = self.env.step(action)
		self.current_lesson += 1
		return self.observation_space, 0, self.current_lesson >= self.episode_length, 0

if __name__ == '__main__':
	ag = PPOAgent(Env())
	ag.run()