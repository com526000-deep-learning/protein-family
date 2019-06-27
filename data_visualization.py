import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import pickle

fname = 'data_1000_max2000.csv'
# fname = 'data_1000_max1200.csv'
# fname = 'data_top10.csv'
# fname = 'data_top10_undersample.csv'

data = pd.read_csv(fname)
seqs = data['sequence']
labels = data['classification']
lengths = [len(s) for s in seqs]
cnt = Counter(labels)
classes = [c[:6] for c in cnt]
counts = [cnt[c] for c in cnt]


# visualize
fig, axarr = plt.subplots(1,2, figsize=(20,8))
axarr[0].bar(range(len(classes)), counts)
plt.sca(axarr[0])
plt.xticks(range(len(classes)), classes, rotation='vertical')
axarr[0].set_ylabel('frequency')
axarr[0].set_title('class count={}'.format(len(classes)))

axarr[1].hist(lengths, bins=50, density=False)
axarr[1].set_xlim([0, 2000])
axarr[1].set_xlabel('sequence length')
axarr[1].set_ylabel('# sequences')
plt.show()