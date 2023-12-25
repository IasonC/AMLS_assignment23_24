from medmnist import PneumoniaMNIST
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import KFold, RandomizedSearchCV
from scipy.stats import uniform, loguniform
from typing import Literal, Tuple
import json


############################################################################
#######################                           ##########################
#######################        DATALOADING        ##########################
#######################                           ##########################
############################################################################

class Data_Pneumonia(object):

    def __init__(self) -> None:

        self.DATA_ROOT = '../Datasets/PneumoniaMNIST'
        self.DATA_FILE = self.DATA_ROOT + '/pneumoniamnist.npz' # file after download

        if not os.path.exists(self.DATA_FILE):
            self._downloader()
        
        self._dataloader()
        print('Data loaded successfully...')

    def _downloader(self) -> None:
        """ Download PneumoniaMNIST data if it does not exist """

        PneumoniaMNIST(split="val", download=True, root=self.DATA_ROOT)
        #dataset = PneumoniaMNIST(split="train", download=True, root=self.DATA_ROOT)
        #dataset = PneumoniaMNIST(split="test", download=True, root=self.DATA_ROOT)

    def _dataloader(self) -> None:
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


############################################################################
#######################                           ##########################
#######################         SVM MODEL         ##########################
#######################                           ##########################
############################################################################

class SVM_Pneumonia(Data_Pneumonia):

    def __init__(self) -> None:
        
        super().__init__()
        self._preprocessing() # preprocess to flatten images to 1-dimension feature vector
        self.set_classifier(C=1, gamma='scale')

    def _preprocessing(self) -> None:
        """ Data preprocess by flattening to match SVM input type """

        svm_tr, svm_v, svm_te = [], [], []
        for i in tqdm(range(len(self.train))):
            svm_tr.append(list(self.train[i].flatten()))
        for i in tqdm(range(len(self.val))):
            svm_v.append(list(self.val[i].flatten()))
        for i in tqdm(range(len(self.test))):
            svm_te.append(list(self.test[i].flatten()))
        
        # normalise
        self.svm_tr = np.array(svm_tr) / 255
        self.svm_v = np.array(svm_v) / 255
        self.svm_te = np.array(svm_te) / 255

    def set_classifier(self, C, gamma) -> None:
        """ Set SVM hyperparameters """

        self.classifier = svm.SVC(C=C, gamma=gamma, kernel='rbf')

    def get_classifier(self) -> svm._classes.SVC:
        """ Print current SVC configuration (C, gamma) """

        print(f'C = {self.classifier.C}')
        print(f'gamma = {self.classifier.gamma}')

        return self.classifier

    def fit_classifier(self) -> None:
        """ Fit SVM classifier to flattened data """

        self.classifier.fit(self.svm_tr, self.train_labels)

    def predict_classifier(self, mode: Literal['train','val','test'], 
                           save: bool = False, params: None | Tuple = None) -> int:
        
        if mode == 'train': (x,y) = (self.svm_tr, self.train_labels)
        elif mode == 'val': (x,y) = (self.svm_v, self.val_labels)
        elif mode == 'test': (x,y) = (self.svm_te, self.test_labels)

        if params:
            (C, gamma) = params
            self.set_classifier(C, gamma)
            self.fit_classifier()

        pred_labels = self.classifier.predict(x)
        accuracy = accuracy_score(y, pred_labels)
        print(f'{mode} accuracy = {accuracy}')

        classes = ['normal','pneumonia']
        print(classification_report(y, pred_labels, target_names=classes))
        
        if save:
            cm = confusion_matrix(y, pred_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig(f'A_svm_cm_{mode}.pdf')

        return accuracy


############################################################################
######################      BAGGED SVM MODEL      ##########################
############################################################################

class BaggedSVM_Pneumonia(SVM_Pneumonia):

    """
    Bagged SVM with subsampling uniformly from entire training set
    (contains unequal Normal-Pneumonia subsets). Uses standard sklearn bagger.

    """

    def __init__(self, C=1, gamma='scale'):

        super().__init__()
        self.set_classifier(C=C, gamma=gamma)

    def set_baggedSVM(self, n_estimators):

        self.baggedSVM = BaggingClassifier(base_estimator=self.classifier,
                                           max_samples=0.4, # rand sel samples for each bagged SVM
                                           bootstrap=True,
                                           n_estimators=n_estimators,
                                           random_state=123,
                                           verbose=1,
                                           )
        
    def fit(self):

        self.baggedSVM.fit(self.svm_tr, self.train_labels)

    def predict_classifier(self, mode: Literal['train','val','test'], save: bool = False) -> int:
        
        if mode == 'train': (x,y) = (self.svm_tr, self.train_labels)
        elif mode == 'val': (x,y) = (self.svm_v, self.val_labels)
        elif mode == 'test': (x,y) = (self.svm_te, self.test_labels)

        pred_labels = self.baggedSVM.predict(x)
        accuracy = accuracy_score(y, pred_labels)
        print(f'{mode} accuracy = {accuracy}')

        classes = ['normal','pneumonia']
        print(classification_report(y, pred_labels, target_names=classes))
        
        if save:
            cm = confusion_matrix(y, pred_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig(f'A_svm_cm_{mode}.pdf')

        return accuracy
    

class custom_BaggedSVM_Pneumonia(SVM_Pneumonia):

    """
    Bagged SVM with custom implementation to sample equally from
    Normal & Pneumonia categories in Training (fitting).
    
    """

    def __init__(self, C=1, gamma='scale', n_estimators=10, max_samples=1.0):

        super().__init__()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.C = C
        self.gamma = gamma

    def fit(self):
        """ Equal class-based sampling for Bagging """
        
        self.classifiers = []
        for _ in tqdm(range(self.n_estimators), desc='Bagging'):
            idx_pneumonia = np.where(self.train_labels==1)[0]
            idx_normal = np.where(self.train_labels==0)[0]
            assert(self.max_samples <= len(idx_normal)/len(self.train_labels))

            # choose class-balanced sample
            sel_pneumonia = np.random.choice(idx_pneumonia, size=int(self.max_samples/2 * len(self.svm_tr)), replace=False)
            sel_normal = np.random.choice(idx_normal, size=int(self.max_samples/2 * len(self.svm_tr)), replace=False)
            
            sel = np.concatenate((sel_pneumonia, sel_normal))

            # random sample for fitting this bag
            x = self.svm_tr[sel]
            y = self.train_labels[sel]

            # create SVM classifier and fit to (x,y)
            svm_n = svm.SVC(C=self.C, gamma=self.gamma, kernel='rbf')
            svm_n.fit(x, y)
            self.classifiers.append(svm_n)
        
    def predict_classifier(self, mode: Literal['train', 'val', 'test'], save: bool = False) -> int:
        
        if mode == 'train': (x,y) = (self.svm_tr, self.train_labels)
        elif mode == 'val': (x,y) = (self.svm_v, self.val_labels)
        elif mode == 'test': (x,y) = (self.svm_te, self.test_labels)

        # aggregate predictions for bagged SVMs
        total_pred = np.zeros(y.shape) # initialise zeros for all test images
        for svm in self.classifiers:
            y_pred = svm.predict(x)
            total_pred += y_pred # add up elementwise

        # majority vote among bagged SVMs
        # if an element value is above n_estimators//2, this is a positive pneumonia prediction
        # because this is majority vote of n_estimators summed up (1 or 0 pred)
        pred_labels = np.array(total_pred > self.n_estimators//2).astype(int)

        accuracy = accuracy_score(y, pred_labels)
        print(f'{mode} accuracy = {accuracy}')

        classes = ['normal','pneumonia']
        print(classification_report(y, pred_labels, target_names=classes))
        
        if save:
            cm = confusion_matrix(y, pred_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig(f'A_custombag_svm_cm_{mode}.pdf')

        return accuracy
    

class custom_BaggedSVM_Pneumonia_CV(SVM_Pneumonia, BaseEstimator):

    """
    Modification of `custom_BaggedSVM_Pneumonia'
    so that it works with the sklearn.base.BaseEstimator
    (
        reformatting fit() and predict() with correct args:
            fit(x,y)->None
            predict(x)->y_pred
    )

    """

    def __init__(self, C=1, gamma='scale', n_estimators=10, max_samples=1.0):

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.C = C
        self.gamma = gamma

    def fit(self, x, y):
        """ Equal class-based sampling for Bagging """
        
        self.classifiers = []
        for _ in tqdm(range(self.n_estimators), desc='Bagging'):
            idx_pneumonia = np.where(y==1)[0]
            idx_normal = np.where(y==0)[0]
            assert(self.max_samples <= len(idx_normal)/len(y))

            # choose class-balanced sample
            sel_pneumonia = np.random.choice(idx_pneumonia, size=int(self.max_samples/2 * len(x)), replace=False)
            sel_normal = np.random.choice(idx_normal, size=int(self.max_samples/2 * len(x)), replace=False)
            
            sel = np.concatenate((sel_pneumonia, sel_normal))

            # random sample for fitting this bag
            xx = x[sel]
            yy = y[sel]

            # create SVM classifier and fit to (X,y)
            svm_n = svm.SVC(C=self.C, gamma=self.gamma, kernel='rbf')
            svm_n.fit(xx, yy)
            self.classifiers.append(svm_n)
        
    def predict(self, x):
        
        # aggregate predictions for bagged SVMs
        total_pred = np.zeros(x.shape[0]) # initialise zeros for all test images
        for svm in self.classifiers:
            y_pred = svm.predict(x)
            total_pred += y_pred # add up elementwise

        # majority vote among bagged SVMs
        # if an element value is above n_estimators//2, this is a positive pneumonia prediction
        # because this is majority vote of n_estimators summed up (1 or 0 pred)
        pred_labels = np.array(total_pred > self.n_estimators//2).astype(int)

        return pred_labels
    

############################################################################
#######################                           ##########################
#######################      MODEL SELECTION      ##########################
#######################                           ##########################
############################################################################
    
class hparam_RandomSearch(object):

    """
    Model Selection:
    -   Random Search of hand-designed hyperparameter ranges
    -   Train-Validation-Test split
    -   Hyperparameters chosen to maximise Validation Accuracy

    Models: SVM_Pneumonia & BaggedSVM_Pneumonia

    """

    def __init__(self, model_class: SVM_Pneumonia | BaggedSVM_Pneumonia):
        
        self.model_class = model_class
        self.svm_p = self.model_class()

    def find_hparams(self):

        #SVM_Pneumonia()
        gamma_mean = 1/(self.svm_p.svm_tr.shape[1] * self.svm_p.svm_tr.var())
            # 1 / (n_features * train_images.var())
            # 'scale' setting as suggested gamma setting
        gamma_e7 = np.random.randn(1,20) * 3e-7 + gamma_mean
            # random hyperparams around 'scale' setting with standard dev 3e-7
        gamma_e7[gamma_e7 < 0] = gamma_mean # mask for non-negative restriction
        gamma_e6 = np.random.rand(1,5) * 1e-6
        gamma_e8 = np.random.rand(1,5) * 1e-8

        gamma = np.concatenate((gamma_e8,gamma_e7,gamma_e6), axis=1)    
        C = np.random.rand(1,20)
        n_estimators = np.array([10,50,100])
        
        val_acc = {}
        best = (0,0,0) # (C, gamma, val acc)
        
        # hyperparameter tuning on held-out validation
        for ci in tqdm(range(C.shape[1]), desc='Hparam tuning'):
            for gi in range(gamma.shape[1]):
                for n in n_estimators:
                    c = C[0,ci]
                    g = gamma[0,gi]

                    self.svm_p.set_classifier(C=c, gamma=g)
                    self.svm_p.set_baggedSVM(n_estimators=n)
                    self.svm_p.fit()
                    acc = self.svm_p.predict_classifier('val')
                    val_acc[f'({c},{g})'] = acc

                    with open('./svm/randsearch_hparams_val_acc.json', 'w') as fp:
                        json.dump(val_acc, fp)

                    if acc > best[2]: best = (c,g,acc)

        self.best_C = best[0]
        self.best_gamma = best[1]
    
    def predict_on_best_hparams(self):

        svm_p = self.model_class()
        # test predictions with best hyperparams
        print('Final Prediction with best hyperparams...')
        svm_p.set_classifier(C=self.best_C,gamma=self.best_gamma)
        svm_p.get_classifier()
        svm_p.fit_classifier()
        svm_p.predict_classifier('train')
        svm_p.predict_classifier('val')
        svm_p.predict_classifier('test')
    

class cross_validation(SVM_Pneumonia):

    """
    Model Selection:
    -   Random search over hyperparameter ranges with given uniform/log distribution
    -   K-Fold Cross-Validation on combined Train-Val set
    -   Hyperparams selected to maximise average accuracy over K held-out folds

    Models: custom_BaggedSVM_Pneumonia_CV
    
    """

    def __init__(self, n=10, iter=10):

        super().__init__() # preprocessing
        self.n = n
        self.iter = iter

        # combine train & validation sets to do K-Fold CV
        self.x = np.concatenate((self.svm_tr, self.svm_v))
        self.y = np.concatenate((self.train_labels, self.val_labels))
    
    def cv(self):
        """ Model Selection with K-Fold Cross-Validation """
        
        kf = KFold(n_splits=self.n, shuffle=True, random_state=123)
        
        distr = dict(
            C = uniform(loc=0, scale=1), # samples in uniform [0,1]
            gamma = loguniform(a=1e-7, b=1), # loguniform in [1e-7,1]
            n_estimators = [10,50],
            max_samples = [0.1, 0.2],
        )
        hparams = RandomizedSearchCV(
            estimator=custom_BaggedSVM_Pneumonia_CV(),
            param_distributions=distr,
            n_iter=self.iter,
            scoring=make_scorer(accuracy_score),
            cv=kf,
            verbose=5,
            random_state=123,
            return_train_score=True,
        )

        hparams.fit(self.x, self.y)

        self.best_params = hparams.best_params_
        print('Best Params: ', self.best_params)

        np.savez_compressed('./svm/cv_results.npz', hparams.cv_results_)

    def predict_on_best_hparams(self, mode: Literal['train','val','test'] = 'test', save: bool = False):
        
        # hard-code
        #self.best_params = {'C': 0.5513147690828912, 
        #                    'gamma': 0.010871332741421732, 
        #                    'max_samples': 0.2, 
        #                    'n_estimators': 10,
        #                    }
        
        if mode == 'train': (testX,trueY) = (self.svm_tr, self.train_labels)
        elif mode == 'val': (testX,trueY) = (self.svm_v, self.val_labels)
        elif mode == 'test': (testX,trueY) = (self.svm_te, self.test_labels)
        
        s = custom_BaggedSVM_Pneumonia_CV(
                C = self.best_params['C'],
                gamma = self.best_params['gamma'],
                max_samples = self.best_params['max_samples'],
                n_estimators = self.best_params['n_estimators'],
            )
        s.fit(self.x, self.y) # fit on concatenated train-val data -- all K folds
        
        pred_labels = s.predict(testX) # predict on test set

        accuracy = accuracy_score(trueY, pred_labels)
        print(f'{mode} accuracy = {accuracy}')

        classes = ['normal','pneumonia']
        print(classification_report(trueY, pred_labels, target_names=classes))
        
        if save:
            cm = confusion_matrix(trueY, pred_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig(f'A_custombag_bestHParam_svm_cm_{mode}.pdf')

        return accuracy


if __name__ == '__main__':

    '''
    s = BaggedSVM_Pneumonia(C=0.7858110973815596,
                            gamma=0.04500387079787024)
    s.set_baggedSVM(n_estimators=100)
    s.fit()
    s.predict_classifier('train')
    s.predict_classifier('val')
    s.predict_classifier('test')
    '''

    '''
    svm_p = SVM_Pneumonia()
    svm_p.set_classifier(C=1,gamma='scale')
    svm_p.fit_classifier()
    svm_p.predict_classifier('test', save=True)
    '''
    '''
    s = custom_BaggedSVM_Pneumonia(n_estimators=10, max_samples=0.2)
    s.fit()
    s.predict_classifier('train', save=True)
    s.predict_classifier('val', save=True)
    s.predict_classifier('test', save=True)
    ''' 

    modelsel = cross_validation(n=7, iter=3)
    modelsel.cv()
    modelsel.predict_on_best_hparams('train')
    modelsel.predict_on_best_hparams('val')
    modelsel.predict_on_best_hparams('test')
    