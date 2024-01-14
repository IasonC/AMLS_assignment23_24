import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import json
from tqdm import tqdm
from typing import Literal

def plot_vals(file_prepend: Literal['bagged_', '']):

    C = []
    gamma = []
    val_acc = []

    with open(f'./{file_prepend}val_acc.json', 'r') as fp:
        hparams = json.load(fp)

    for k in tqdm(hparams):
        s = k.split('(')[1].split(',')
        c = float(s[0])
        g = float(s[1].split(')')[0])

        C.append(c)
        gamma.append(g)
        val_acc.append(hparams[k])

    return (C, gamma, val_acc)

(C,gamma,val_acc) = plot_vals('')
(C_b,gamma_b,val_acc_b) = plot_vals('bagged_')

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection='3d')


ax.scatter3D(np.array(C), np.log10(np.array(gamma)), np.array(val_acc), color='green')
ax.scatter3D(np.array(C_b), np.log10(np.array(gamma_b)), np.array(val_acc_b), color='red')


# Set logarithmic scale on the x variable
ax.set_xlabel('C', fontsize=20, labelpad=20)
ax.set_ylabel('Gamma (γ)', fontsize=20, labelpad=20)
ax.set_zlabel('Validation Accuracy', fontsize=20, labelpad=20)
plt.title('Validation Accuracy for Random Search {C,γ}', fontsize=20)
plt.legend(['SVM', 'SVM-B'])

ax.tick_params(axis='both', which='major', labelsize=15)
ax.tick_params(axis='both', which='minor', labelsize=15)

#plt.show()
plt.savefig('both_hparam_svm.pdf')