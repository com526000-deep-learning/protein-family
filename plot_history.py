import pickle
import matplotlib.pyplot as plt
import numpy as np

def plotFigure(y_train, y_test, fname, name):
    plt.title("model " + name)
    plt.plot(np.arange(len(y_train)), y_train, label = 'train')    
    plt.plot(np.arange(len(y_test)), y_test, label = 'validate')    
    plt.xlabel("epoch")
    plt.ylabel(name)
    plt.legend(loc = 'lower right')
    plt.savefig(fname + "_" + name)
    plt.close("all")

fname = "cnn_gru_ngram_10"

df = pickle.load(open('{}.pickle'.format(fname), 'rb'))
acc_tr = df[0]
acc_ts = df[1]
loss_tr = df[2]
loss_ts = df[3]

plotFigure(acc_tr, acc_ts, fname, "acc")
plotFigure(loss_tr, loss_ts, fname, "loss")
