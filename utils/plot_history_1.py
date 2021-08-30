import pickle
import matplotlib.pyplot as plt
import numpy as np

def plotFigure(df, name):
    plt.title("model " + name)
    acc_ts = df[1]
    plt.plot(np.arange(len(acc_ts)), acc_ts, label = name)    

fname = "1DCNN_34"
df = pickle.load(open('{}.pickle'.format(fname), 'rb'))
plotFigure(df, "1D-CNN")

fname = "1DCNN_undersample_34"
df = pickle.load(open('{}.pickle'.format(fname), 'rb'))
plotFigure(df, "1D-CNN, 4-gram")


fname = "GRU_ngram_34"
df = pickle.load(open('{}.pickle'.format(fname), 'rb'))
plotFigure(df, "GRU, 3-gram")

fname = "cnn_gru_ngram_34"
df = pickle.load(open('{}.pickle'.format(fname), 'rb'))
plotFigure(df, "CNN + GRU, 3-gram")

plt.legend()
plt.xlabel("epoch")
plt.ylabel("acc")
plt.title("learning curve (34 classes)")
plt.savefig("learning_curve_34")