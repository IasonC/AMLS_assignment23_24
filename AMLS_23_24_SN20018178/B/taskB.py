import os
import time
from tqdm import tqdm
from typing import Literal, Tuple
from medmnist import PathMNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


############################################################################
#######################                           ##########################
#######################        DATALOADING        ##########################
#######################                           ##########################
############################################################################


class Data_Path(object):
    def __init__(self) -> None:
        self.DATA_ROOT = "../Datasets/PathMNIST"
        self.DATA_FILE = self.DATA_ROOT + "/pathmnist.npz"  # file after download

        if not os.path.exists(self.DATA_FILE):
            self._downloader()

        self._dataloader()
        print("Data loaded successfully...")

    def _downloader(self) -> None:
        """Download PneumoniaMNIST data if it does not exist"""

        PathMNIST(split="val", download=True, root=self.DATA_ROOT)
        # dataset = PneumoniaMNIST(split="train", download=True, root=self.DATA_ROOT)
        # dataset = PneumoniaMNIST(split="test", download=True, root=self.DATA_ROOT)

    def _dataloader(self) -> None:
        """Loads downloaded PneumoniaMNIST data and
        extracts image & labels from train-val-test split"""

        self.data = np.load(self.DATA_FILE)

        # Train-Val-Test Split extract
        self.train = self.data["train_images"]
        self.val = self.data["val_images"]
        self.test = self.data["test_images"]

        # Labels extract
        self.train_labels = np.ravel(self.data["train_labels"])
        self.val_labels = np.ravel(self.data["val_labels"])
        self.test_labels = np.ravel(self.data["test_labels"])


############################################################################
#######################                           ##########################
#######################         SVM MODEL         ##########################
#######################                           ##########################
############################################################################


class SVM_Path(Data_Path):
    def __init__(
        self,
        C: int = 1,
        gamma: float | Literal["scale", "auto"] = "scale",
        multiclass: Literal["ovr", "ovo"] = "ovr",
        break_ties: bool = False,
    ) -> None:
        super().__init__()

        self.C = C
        self.gamma = gamma
        self.multiclass = multiclass
        self.break_ties = break_ties

        self._preprocessing()
        print("Preprocessing done...\n")

    def _preprocessing(self) -> None:
        """Data preprocess by flattening to match SVM input type"""

        # reshape to flatten and normalise to [0,1] range (max 255 originally)
        self.svm_tr = np.reshape(self.train, (self.train.shape[0], -1)) / 255
        self.svm_v = np.reshape(self.val, (self.val.shape[0], -1)) / 255
        self.svm_te = np.reshape(self.test, (self.test.shape[0], -1)) / 255

    def set_classifier(self, ret: bool = False) -> None:
        """Set SVM hyperparameters"""

        #classifier = svm.SVC(
        #    C=self.C,
        #    gamma=self.gamma,
        #    kernel="linear",
        #    class_weight=None,
        #    decision_function_shape=self.multiclass,
        #    break_ties=self.break_ties,
        #)

        classifier = svm.LinearSVC(
            penalty='l2',
            loss='squared_hinge',
            dual=False,
            C=self.C,
            multi_class='ovr',
            verbose=1,
        )

        if ret:
            return classifier
        else:
            self.classifier = classifier

    def fit_classifier(self) -> None:
        """Fit SVM classifier to flattened data"""
        
        '''self.n_classes = np.unique(self.train_labels)

        self.ovr_classifiers = []
        for c in tqdm(self.n_classes, desc='Fit OvR SVMs'):
            # transform labels to be binary [Class | Other], Other = -1
            labels = np.zeros_like(self.train_labels)-1
            labels[self.train_labels == c] = c

            # fit binary
            classifier = self.set_classifier(ret=True)
            classifier.fit(self.svm_tr, labels)

            self.ovr_classifiers.append(classifier)'''
        

        self.classifier.fit(self.svm_tr, self.train_labels)

    def predict_classifier(
        self,
        mode: Literal["train", "val", "test"] = "test",
        save: bool = False,
    ) -> int:
        if mode == "train":
            (x, y) = (self.svm_tr, self.train_labels)
        elif mode == "val":
            (x, y) = (self.svm_v, self.val_labels)
        elif mode == "test":
            (x, y) = (self.svm_te, self.test_labels)

        '''preds = np.zeros((len(y), self.n_classes))
        for c in tqdm(range(len(self.ovr_classifiers)), desc='Predict from OvR SVMs'):
            classifier = self.ovr_classifiers[c]
            pred_c = classifier.predict(x)
            preds[pred_c == -1] += 1 #####################
            preds[pred_c == c] += 1 #####################'''

        pred_labels = self.classifier.predict(x)
        accuracy = accuracy_score(y, pred_labels)
        print(f"{mode} accuracy = {accuracy}")

        print(classification_report(y, pred_labels))

        if save:
            cm = confusion_matrix(y, pred_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig(f"B_svm_cm_{mode}.pdf")

        return accuracy
    

############################################################################
############                                                    ############
############       SQUEEZE-AND-EXCITATION RESNET MODEL          ############
############                                                    ############
############################################################################

import torch
import torchvision
from torchvision.models.resnet import ResNet18_Weights
from torch.utils.data import TensorDataset, DataLoader 
from torchmetrics import Precision, Recall
#from senet.se_resnet import resnet_model
import copy

class SqueezeExcitationResNet(Data_Path):
    def __init__(self, 
                 num_classes: int = 9, 
                 batch_size: int = 64, 
                 feature_extract: bool = True, 
                 use_pretrained: bool = True, 
                 device: str = 'cuda'
                 ) -> None:
        super().__init__()

        self.device = device
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.dataloading()

        #self.model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) #resnet_model()
            # resnet pretrained on ImageNet (transfer learning)
        self.model_ft = torchvision.models.resnet18(pretrained=use_pretrained)
        self.set_parameter_requires_grad(self.model_ft, feature_extract)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
        self.input_size = 28

        self.model_ft.to(device)
        
        print('\nResNet18 Model:\n---------------------------------')
        print(self.model_ft)

        # Gather parameters for learning (based on pretraining & freezing)
        if feature_extract:
            params_to_update = []
            for name,param in self.model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
        else:
            params_to_update = self.model_ft.parameters()
        
        self.optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()
        #self.scheduler = lr_scheduler.StepLR(80, 0.1)

        # metrics Precision & Recall
        self.P = Precision(task="multiclass", average='macro', num_classes=num_classes).to(device)
        self.R = Recall(task="multiclass", average='macro', num_classes=num_classes).to(device)
            # macro averaging to emphasize class performance, not instance (avoid class imbalance)

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def dataloading(self) -> None:
        print('Dataloading starting...')

        tr = torch.Tensor(self.train).to(self.device)
        tr_label = torch.tensor(self.train_labels, dtype=torch.long).to(self.device)
        v = torch.Tensor(self.val).to(self.device)
        v_label = torch.tensor(self.val_labels, dtype=torch.long).to(self.device)
        te = torch.Tensor(self.test).to(self.device)
        te_label = torch.tensor(self.test_labels, dtype=torch.long).to(self.device)

        train_dataset = TensorDataset(tr, tr_label)
        val_dataset = TensorDataset(v, v_label)
        test_dataset = TensorDataset(te, te_label)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)

        self.dataloaders_dict = dict(
            train=train_dataloader,
            val=val_dataloader,
            test=test_dataloader,
        )

        print('Dataloading done...')

    def train_val_epochs(self, epochs):
        
        '''tqdm_rep = reporters.TQDMReporter(range(epochs))
        tensorboard_rep = reporters.TensorboardReporter('./senet/logdir/')
        _callbacks = [tqdm_rep, tensorboard_rep, callbacks.AccuracyCallback()]
        with Trainer(self.model, self.optimizer, F.cross_entropy, scheduler=self.scheduler, callbacks=_callbacks) as trainer:
            for _ in tqdm_rep:
                trainer.train(self.train_dataloader)
                trainer.test(self.val_dataloader)'''
        

        val_acc = [] # track val accuracy
        best_model = copy.deepcopy(self.model_ft.state_dict())
        best_acc = 0.0
        count_not_better = 0


        for e in range(epochs):

            print(f'\nEPOCH {e} of {epochs}\n')

            # handle training and validation modes
            for mode in ['train','val']:
                if mode=='train':
                    self.model_ft.train()
                elif mode=='val':
                    self.model_ft.eval() # set mode

                current_loss = 0.0
                current_corrects = 0
                current_precision = 0
                current_recall = 0
                batch_count = 0

                for x, y in self.dataloaders_dict[mode]:
                    #x = x.to(self.device)

                    if x.shape[0] < self.batch_size:
                        continue # last batch, uneven
                    
                    #print(f'SHAPE: x = {x.shape}, y = {y.shape}')
                    x = torch.reshape(x, (self.batch_size, 3, 28, 28))
                    #y = y.to(self.device)
                
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(mode=='train'):
                        y_pred = self.model_ft(x)
                        loss = self.criterion(y_pred, y)
            
                        _, preds = torch.max(y_pred, 1)

                        
                        (precision, recall) = (self.P(preds, y), self.R(preds, y))
                            # macro average for emphasis on classes, not instances (counter class imbalance)
                        precision, recall = precision.item(), recall.item() # extract value from 1-d torch.tensor output


                        # backward + optimize only if in training phase
                        if mode == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    current_loss += loss.item() * x.size(0)
                    current_corrects += torch.sum(preds == y)
                    current_precision += precision
                    current_recall += recall

                    batch_count += 1

                epoch_loss = current_loss / len(self.dataloaders_dict[mode].dataset)
                epoch_acc = current_corrects.double() / len(self.dataloaders_dict[mode].dataset)
                epoch_precision = current_precision / batch_count
                epoch_recall = current_recall / batch_count

                print('{} Loss: {:.4f} Acc: {:.4f} Prec: {:.4f} Rec: {:.4f}'.format(mode, epoch_loss, epoch_acc, epoch_precision, epoch_recall))
                print('\n========================================')

                # log val_acc and deep copy the model again if it is an improvement
                if mode == 'val':
                    val_acc.append(epoch_acc)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model = copy.deepcopy(self.model_ft.state_dict())
                    elif epoch_acc < 1.01 * best_acc: # next epoch less than 1 percent val-acc improvement
                        count_not_better += 1
                
                # early stopping
                if count_not_better >= 5:
                    print('Early Stopping: 5 epochs with no val-acc improvement...')
                    break

        self.model_ft.load_state_dict(best_model)
        return self.model_ft, val_acc

        '''
        # Training
        for batch_idx, (x,y) in enumerate(self.train_dataloader):
            x = torch.reshape(x, (64,3,28,28))
            
            self.optimizer.zero_grad() # clear gradients
            ypred = self.model(x)
            loss = self.criterion(ypred,y)
            loss.backward() # calculate gradients
            self.optimizer.step() # update weights
            
        print(f'Train Loss: {loss.item()}')

        # Validation
        with torch.no_grad():
            for batch_idx, (x,y) in enumerate(self.val_dataloader):
                ypred = self.model(x)
                loss = self.criterion(ypred,y)
            
        print(f'Val Loss: {loss.item()}')
        '''

        print('\n\n')


if __name__ == "__main__":
    '''s = SVM_Path()
    s.set_classifier()
    tf_i = time.time()
    s.fit_classifier()
    tf_f = time.time()
    s.predict_classifier("train", save=True)
    tp_tr = time.time()
    s.predict_classifier("val", save=True)
    tp_v = time.time()
    s.predict_classifier("test", save=True)
    tp_te = time.time()

    print(f'\nTime to fit: {tf_f - tf_i}')
    print(f'Time to pred: {tp_tr-tf_f} Train | {tp_v-tp_tr} Val | {tp_te-tp_v} Test')'''

    se = SqueezeExcitationResNet(feature_extract=False)
    se.train_val_epochs(epochs=100)