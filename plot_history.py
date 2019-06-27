import pickle
import matplotlib.pyplot as plt
import numpy as np

def plotFigure(df, name):
    plt.title("model " + name)
    acc_ts = df[1]
    plt.plot(np.arange(len(acc_ts)), acc_ts, label = name)    



fname = "1DCNN_10"
df = pickle.load(open('{}.pickle'.format(fname), 'rb'))
plotFigure(df, "1D-CNN, top 10")

fname = "1DCNN_10_undersample"
df = pickle.load(open('{}.pickle'.format(fname), 'rb'))
plotFigure(df, "1D-CNN, top 10, 4-gram")

fname = "GRU_ngram_10_undersample"
df = pickle.load(open('{}.pickle'.format(fname), 'rb'))
plotFigure(df, "GRU, top 10, 3-gram")

fname = "cnn_gru_ngram_10"
df = pickle.load(open('{}.pickle'.format(fname), 'rb'))
plotFigure(df, "CNN + GRU, top 10, 3-gram")

plt.legend()
plt.xlabel("epoch")
plt.ylabel("acc")
plt.title("learning curve")
plt.savefig("learning_curve")


