from medmnist import PneumoniaMNIST
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


class Data_Pneumonia(object):

    def __init__(self):

        self.DATA_ROOT = '../Datasets/PneumoniaMNIST'
        self.DATA_FILE = self.DATA_ROOT + '/pneumoniamnist.npz' # file after download

        if not os.path.exists(self.DATA_FILE):
            self._downloader()
        
        self._dataloader()
        print('Data loaded successfully...')

    def _downloader(self):
        """ Download PneumoniaMNIST data if it does not exist """

        PneumoniaMNIST(split="val", download=True, root=self.DATA_ROOT)
        #dataset = PneumoniaMNIST(split="train", download=True, root=self.DATA_ROOT)
        #dataset = PneumoniaMNIST(split="test", download=True, root=self.DATA_ROOT)

    def _dataloader(self):
        """ Loads downloaded PneumoniaMNIST data and 
            extracts image & labels from train-val-test split """

        self.data = np.load(self.DATA_FILE)

        # Train-Val-Test Split extract
        self.train = self.data['train_images']
        self.val = self.data['val_images']
        self.test = self.data['test_images']

        # Labels extract
        self.train_labels = np.ravel(self.data['train_labels'])
        self.val_labels = np.ravel(self.data['val_labels'])
        self.test_labels = np.ravel(self.data['test_labels'])


class SVM_Pneumonia(Data_Pneumonia):

    def __init__(self):
        
        super().__init__()
        self._preprocessing() # preprocess to flatten images to 1-dimension feature vector
        self.classifier = svm.SVC(C=1, gamma='scale', kernel='linear')

    def _preprocessing(self):
        """ Data preprocess by flattening to match SVM input type """

        svm_tr, svm_v, svm_te = [], [], []
        for i in tqdm(range(len(self.train))):
            svm_tr.append(list(self.train[i].flatten()))
        for i in tqdm(range(len(self.val))):
            svm_v.append(list(self.val[i].flatten()))
        for i in tqdm(range(len(self.test))):
            svm_te.append(list(self.test[i].flatten()))
        self.svm_tr = np.array(svm_tr)
        self.svm_v = np.array(svm_v)
        self.svm_te = np.array(svm_te)

    def fit(self):
        """ Fit SVM classifier to flattened data """

        self.classifier.fit(self.svm_tr, self.train_labels)

    def predict_validation(self):
        
        pred_labels = self.classifier.predict(self.svm_v)
        accuracy = accuracy_score(self.val_labels, pred_labels)
        print(f'Val Accuracy = {accuracy}')
        cm = confusion_matrix(self.val_labels, pred_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig('A_svm_cm_val.pdf')

    def predict_test(self):
        
        pred_labels = self.classifier.predict(self.svm_te)
        accuracy = accuracy_score(self.test_labels, pred_labels)
        print(f'Test Accuracy = {accuracy}')
        cm = confusion_matrix(self.test_labels, pred_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig('A_svm_cm_test.pdf')

if __name__ == '__main__':
    svm = SVM_Pneumonia()
    svm.fit()
    svm.predict_validation()
    svm.predict_test()