import numpy as np
import math
import pandas as pd
from ta import *

# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(float(line.split(",")[4]))

	return vec

def getPandasDataVec(key):
	df = pd.read_csv("data/" + key + ".csv", sep=',')
	df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume", fillna=True)

	indicators = ['Close', 'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em',
	   'volume_vpt', 'volume_nvi', 'volatility_atr', 'volatility_bbh',
	   'volatility_bbl', 'volatility_bbm', 'volatility_bbhi',
	   'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
	   'volatility_kchi', 'volatility_kcli', 'volatility_dch',
	   'volatility_dcl', 'volatility_dchi', 'volatility_dcli', 'trend_macd',
	   'trend_macd_signal', 'trend_macd_diff', 'trend_ema_fast',
	   'trend_ema_slow', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg',
	   'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_diff',
	   'trend_trix', 'trend_mass_index', 'trend_cci', 'trend_dpo', 'trend_kst',
	   'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_a',
	   'trend_ichimoku_b', 'trend_visual_ichimoku_a',
	   'trend_visual_ichimoku_b', 'trend_aroon_up', 'trend_aroon_down',
	   'trend_aroon_ind', 'momentum_rsi', 'momentum_mfi', 'momentum_tsi',
	   'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
	   'momentum_ao', 'others_dr', 'others_dlr', 'others_cr']

	zcores = []
	for i in indicators:
		# z-score
		zscore = (df[i] - df[i].mean())/df[i].std(ddof=0)
		zcores.append(zscore)
	return zcores

# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# return indicators for n-day ending at time t
def getIndicators(dataFrame, t, n):

	state_vector = list()

	for timestep in range(n-1):
		current_vector = list()
		for i in dataFrame:
			# z-score
			#zscore = (dataFrame[i] - dataFrame[i].mean())/dataFrame[i].std(ddof=0)
			window = list(i)[t:t+n]
			val = window[0]
			if timestep<len(window):
				val = window[timestep]
			current_vector.append(val)
		state_vector.append(current_vector)
	
	return state_vector

# returns an an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])
