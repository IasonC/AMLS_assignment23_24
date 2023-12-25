from tqdm import tqdm
import numpy as np


DATA_FILE = '../Datasets/PneumoniaMNIST/pneumoniamnist.npz'
data = np.load(DATA_FILE)

# Train-Val-Test Split extract
train = data['train_images']
val = data['val_images']
test = data['test_images']

# Labels extract
train_labels = np.ravel(data['train_labels'])
val_labels = np.ravel(data['val_labels'])
test_labels = np.ravel(data['test_labels'])

svm_tr, svm_v, svm_te = [], [], []
for i in tqdm(range(len(train))):
    svm_tr.append(list(train[i].flatten()))
for i in tqdm(range(len(val))):
    svm_v.append(list(val[i].flatten()))
for i in tqdm(range(len(test))):
    svm_te.append(list(test[i].flatten()))
svm_tr = np.array(svm_tr)
svm_v = np.array(svm_v)
svm_te = np.array(svm_te)

print(svm_tr[0:10])
print('\n')
print(train_labels[0:10])

print('Mean-Variance for Train-Val-Test:')
print(f'TRAIN: μ = {np.mean(svm_tr)}, σ = {np.var(svm_tr)}, MIN = {np.min(svm_tr)}, MAX = {np.max(svm_tr)}')
print(f'VAL: μ = {np.mean(svm_v)}, σ = {np.var(svm_v)}, MIN = {np.min(svm_v)}, MAX = {np.max(svm_v)}')
print(f'TEST: μ = {np.mean(svm_te)}, σ = {np.var(svm_te)}, MIN = {np.min(svm_te)}, MAX = {np.max(svm_te)}')

print('Class Split')
print(f'TRAIN: {sum(train_labels)} pneumonia, {len(train_labels)-sum(train_labels)} normal')
