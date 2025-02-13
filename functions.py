import numpy as np
import math
import pandas as pd
import warnings
from tqdm import tqdm
from ta import *
from dateutil.parser import parse
import datetime

cache = {}
dataframes = {}
warnings.simplefilter(action='ignore', category=FutureWarning)

def to_date(str):
	dt = None
	try:
		dt = parse(str)
	except:
		dt = datetime.datetime.strptime(str, '%Y-%m-%d %I-%p')
	if dt is None:
		print("date format not recognized", str)
		exit()
	return dt

# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	df = pd.read_csv("data/" + key + ".csv", sep=',')
	return list(df['Close'])
	# vec = []
	# lines = open("data/" + key + ".csv", "r").read().splitlines()

	# for line in lines[1:]:
	# 	vec.append(float(line.split(",")[4]))

	# return vec

def partOfWeek(p):
	# returns 0 for week, 1 for weekend
	if (p.weekday()<5):
		return 0.0
	return 1.0

def partOfDay(p):
    return (
    	# morning
        -1.0 if 5 <= p.hour <= 11
        else
        # day
        0 if 12 <= p.hour <= 22
        else
        # night
        1.0
    )

def timeOfDay(p):
	print(partOfDay(p))
	return p

def prepareDataFrames(key, window):
	df = pd.read_csv("data/" + key + ".csv", sep=',')

	# check if dataset needs to be reversed chronologically
	last_date = to_date(df['Date'].get_value(len(df)-1, 'datetime'));
	first_date = to_date(df['Date'].get_value(0, 'datetime'));
	if ((last_date - first_date).total_seconds() < 0):
		print("data is in reversed chronological order, reversing dataframe")
		df = df.reindex(index=df.index[::-1])
		df.reset_index(inplace=True, drop=True) 
		last_date = to_date(df['Date'].get_value(len(df)-1, 'datetime'));
		first_date = to_date(df['Date'].get_value(0, 'datetime'));
		if ((last_date - first_date).total_seconds() < 0):
			print("unable to reverse dataset")
			exit()

	df['open_close_diff'] = df['Close'] - df['Open']
	df['part_of_day'] = df.apply(lambda row: partOfDay(to_date(row['Date'])), axis=1)
	df['part_of_week'] = df.apply(lambda row: partOfWeek(to_date(row['Date'])), axis=1)
	
	print("computing all stock signals...")
	with np.errstate(divide='ignore', invalid='ignore'):
		df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume", fillna=True)
	
	indicators = ['open_close_diff', 'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em',
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
	
	extra_features = ['part_of_day', 'part_of_week']
	
	indicators_dataframe = df [indicators]
	extra_feature_dataframe = df[extra_features]

	rows = df['open_close_diff'].count()
	for i in tqdm(range(rows), desc="caching"):
			large_window = indicators_dataframe[0:i+1]
			large_window_ex = extra_feature_dataframe[0:i+1]

			zscore = ((large_window - large_window.mean(axis=0)) / large_window.std(ddof=0)).fillna	(value=0)
			missing_rows = window - large_window['open_close_diff'].count()
			duplicate_row = zscore.loc[0]
			duplicate_ex = large_window_ex.loc[0]

			if missing_rows>0:
				zscore = zscore.append([duplicate_row]*missing_rows,ignore_index=True)
				large_window_ex = large_window_ex.append([duplicate_ex]*missing_rows,ignore_index=True)

			merged = pd.concat([zscore, large_window_ex], axis=1, sort=False)
			#print(merged.tail(window))
			dataframes[i] = merged.values.tolist();

	return dataframes

# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))


# return debug info
def getDebugObservations(closeData, dataFrame, t, n):
	d = t - (n )
	if d < 0:
		d = 0
	
	slice = dataFrame.iloc[:, d:d+n]
	return { "time" : t, "observed": (d, d+n), "window_open_close_zscore" : slice.loc['open_close_diff'].tolist()  }

# return indicators for n-day ending at time t
def getIndicators(dataFrame, t, n):
	return dataframes[t]

	# d = t - (n )
	# if d < 0:
	# 	d = 0
	# if (t in cache):
	# 	return cache[t]

	# state_vector = list()
	# slice = dataFrame.iloc[:, d:d+n]

	# #print(slice)
	# #print(slice)
	# #print("---------------------")
	# for column in slice:
	# 	state_vector.append(list(slice[column]))
	
	# missing_cols = n - len(state_vector)
	# for m in range(missing_cols):
	# 	state_vector.append(state_vector[0])

	# # for timestep in range(n-1):
	# # 	current_vector = list()
	# # 	for i in dataFrame:
	# # 		window = list(i)[t:t+n]
	# # 		val = window[0]
	# # 		if timestep<len(window):
	# # 			val = window[timestep]
	# # 		current_vector.append(val)
	# # 	state_vector.append(current_vector)
	
	# cache[t] = state_vector
	# return state_vector

# returns an an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])
