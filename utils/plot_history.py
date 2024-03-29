import pickle
import matplotlib.pyplot as plt
import numpy as np

def plotFigure(df, name):
    plt.title("model " + name)
    acc_ts = df[1]
    plt.plot(np.arange(len(acc_ts)), acc_ts, label = name)    

fname = "1DCNN_10"
df = pickle.load(open('{}.pickle'.format(fname), 'rb'))
plotFigure(df, "1D-CNN")

fname = "1DCNN_10_undersample"
df = pickle.load(open('{}.pickle'.format(fname), 'rb'))
plotFigure(df, "1D-CNN, 4-gram")

fname = "GRU_ngram_10_undersample"
df = pickle.load(open('{}.pickle'.format(fname), 'rb'))
plotFigure(df, "GRU, 3-gram")

fname = "cnn_gru_ngram_10"
df = pickle.load(open('{}.pickle'.format(fname), 'rb'))
plotFigure(df, "CNN + GRU, 3-gram")

plt.legend()
plt.xlabel("epoch")
plt.ylabel("acc")
plt.title("learning curve (top 10)")
plt.savefig("learning_curve")
