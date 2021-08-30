import pickle

def save_history(history, fname):
	'''
	fname format:
	[modelname]_[classes]
	e.g.
	cnn_1d_10

	data_top10.csv -> 10 classes
	data_1000_max
	'''
	f = open('{}.pickle'.format(fname), 'wb')
	pickle.dump(history, f)
	f.close()