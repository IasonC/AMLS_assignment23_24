import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import json
from tqdm import tqdm

C = []
gamma = []
val_acc = []

with open('./val_acc.json', 'r') as fp:
    hparams = json.load(fp)

for k in tqdm(hparams):
    s = k.split('(')[1].split(',')
    c = float(s[0])
    g = float(s[1].split(')')[0])

    C.append(c)
    gamma.append(g)
    val_acc.append(hparams[k])


fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection='3d')

ax.scatter3D(np.log10(np.array(gamma)), np.array(C), np.array(val_acc), color='green')
# Set logarithmic scale on the x variable

plt.show()