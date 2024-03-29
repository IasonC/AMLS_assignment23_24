import os, sys
import time, json
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
        self.DATA_ROOT = "Datasets/"
        self.DATA_FILE = self.DATA_ROOT + "pathmnist.npz"  # file after download

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

        # classifier = svm.SVC(
        #    C=self.C,
        #    gamma=self.gamma,
        #    kernel="linear",
        #    class_weight=None,
        #    decision_function_shape=self.multiclass,
        #    break_ties=self.break_ties,
        # )

        classifier = svm.LinearSVC(
            penalty="l2",
            loss="squared_hinge",
            dual=False,
            C=self.C,
            multi_class="ovr",
            verbose=1,
        )

        if ret:
            return classifier
        else:
            self.classifier = classifier

    def fit_classifier(self) -> None:
        """Fit SVM classifier to flattened data"""

        """self.n_classes = np.unique(self.train_labels)

        self.ovr_classifiers = []
        for c in tqdm(self.n_classes, desc='Fit OvR SVMs'):
            # transform labels to be binary [Class | Other], Other = -1
            labels = np.zeros_like(self.train_labels)-1
            labels[self.train_labels == c] = c

            # fit binary
            classifier = self.set_classifier(ret=True)
            classifier.fit(self.svm_tr, labels)

            self.ovr_classifiers.append(classifier)"""

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

        """preds = np.zeros((len(y), self.n_classes))
        for c in tqdm(range(len(self.ovr_classifiers)), desc='Predict from OvR SVMs'):
            classifier = self.ovr_classifiers[c]
            pred_c = classifier.predict(x)
            preds[pred_c == -1] += 1 #####################
            preds[pred_c == c] += 1 #####################"""

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
############      SQUEEZE-AND-EXCITATION & RESNET MODELS        ############
############                                                    ############
############################################################################

import torch
import torchvision
from torchvision.models.resnet import BasicBlock
from torchvision.transforms import v2
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Precision, Recall
import copy
from collections import OrderedDict

class PathResNet18(Data_Path):
    def __init__(
        self,
        num_classes: int = 9,
        batch_size: int = 64,
        feature_extract: bool = True,
        up_to_layer: int = 0,
        save_name: str = '',
        use_pretrained: bool = True,
        post_layers: bool = True,
        dr: float = 0.1,
        lr: float = 0.001,
        rho: float = 0.9
    ) -> None:
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.save_name = save_name
        self.dataloading()

        # self.model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) #resnet_model()
        # resnet pretrained on ImageNet (transfer learning)
        self.model_ft = torchvision.models.resnet18(pretrained=use_pretrained)
        self.set_parameter_requires_grad(self.model_ft, feature_extract, up_to_layer)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
        self.input_size = 28

        if post_layers:
            self.model_ft = torch.nn.Sequential(
                self.model_ft,
                torch.nn.Dropout1d(p=dr, inplace=True),
                torch.nn.Softmax()
            )

        self.model_ft.to(self.device)

        print("\nResNet18 Model:\n---------------------------------")
        print(self.model_ft)

        # Gather parameters for learning (based on pretraining & freezing)
        if feature_extract:
            params_to_update = []
            for name, param in self.model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
        else:
            params_to_update = self.model_ft.parameters()

        self.optimizer = torch.optim.SGD(params_to_update, lr=lr, momentum=rho)#, weight_decay=0.01)
        self.criterion = torch.nn.CrossEntropyLoss()
        # self.scheduler = lr_scheduler.StepLR(80, 0.1)

        # metrics Precision & Recall
        self.P = Precision(
            task="multiclass", average="macro", num_classes=num_classes
        ).to(self.device)
        self.R = Recall(task="multiclass", average="macro", num_classes=num_classes).to(
            self.device
        )
        # macro averaging to emphasize class performance, not instance (avoid class imbalance)

    def set_parameter_requires_grad(self, model, feature_extracting, up_to_layer):
        if feature_extracting:
            cnt = 0
            for child in self.model_ft.children():
                if cnt >= up_to_layer:
                    break # freeze up to given layers
                print(f'Freezing child layer: {child}')
                for param in child.parameters():
                    param.requires_grad = False
                cnt += 1
                

    def dataloading(self) -> None:
        print("Dataloading starting...")

        tr = torch.Tensor(self.train).to(self.device)
        tr_label = torch.tensor(self.train_labels, dtype=torch.long).to(self.device)
        v = torch.Tensor(self.val).to(self.device)
        v_label = torch.tensor(self.val_labels, dtype=torch.long).to(self.device)
        te = torch.Tensor(self.test).to(self.device)
        te_label = torch.tensor(self.test_labels, dtype=torch.long).to(self.device)

        train_dataset = TensorDataset(tr, tr_label)
        val_dataset = TensorDataset(v, v_label)
        test_dataset = TensorDataset(te, te_label)

        #self.transforms = v2.Compose([
        #    #v2.RandomHorizontalFlip(p=0.5),
        #    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # policy learnt on ImageNet (generalisable ok)
        #])
        #self.cutmix = v2.CutMix(num_classes=self.num_classes)

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)

        self.dataloaders_dict = dict(
            train=train_dataloader,
            val=val_dataloader,
            test=test_dataloader,
        )

        print("Dataloading done...")

    def train_model(self, epochs: int = 10, early_stopping: bool = False, save_cm: bool = True):
        """tqdm_rep = reporters.TQDMReporter(range(epochs))
        tensorboard_rep = reporters.TensorboardReporter('./senet/logdir/')
        _callbacks = [tqdm_rep, tensorboard_rep, callbacks.AccuracyCallback()]
        with Trainer(self.model, self.optimizer, F.cross_entropy, scheduler=self.scheduler, callbacks=_callbacks) as trainer:
            for _ in tqdm_rep:
                trainer.train(self.train_dataloader)
                trainer.test(self.val_dataloader)"""

        val_acc = []  # track val accuracy
        self.best_model = copy.deepcopy(self.model_ft.state_dict())
        self.BEST_MODEL_PATH = f'best_model_resnet_{self.save_name}.pty'
        best_acc = 0.0
        count_not_better = 0

        log = {
            "train-loss": [],
            "train-acc": [],
            "train-precision": [],
            "train-recall": [],
            "val-loss": [],
            "val-acc": [],
            "val-precision": [],
            "val-recall": [],
        }

        all_preds, all_labels = torch.tensor([]), torch.tensor([])
        flag_cm = True

        for e in range(epochs):
            print(f"\nEPOCH {e} of {epochs}\n")

            # handle training and validation modes
            for mode in ["train", "val"]:
                if mode == "train":
                    self.model_ft.train()
                elif mode == "val":
                    self.model_ft.eval()  # set mode

                current_loss = 0.0
                current_corrects = 0
                current_precision = 0
                current_recall = 0
                batch_count = 0

                for x, y in self.dataloaders_dict[mode]:
                    # x = x.to(self.device)

                    if x.shape[0] < self.batch_size or y.shape[0] < self.batch_size:
                        #print(f'{mode} batch undersized --> skipped...')
                        continue # last batch, uneven

                    # print(f'SHAPE: x = {x.shape}, y = {y.shape}')
                    x = torch.reshape(x, (self.batch_size, 3, 28, 28))
                    #x = self.transforms(x) # apply transforms

                    #x, y = self.cutmix(x, y)
                    # y = y.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(mode == "train"):
                        y_pred = self.model_ft(x)
                        loss = self.criterion(y_pred, y)

                        _, preds = torch.max(y_pred, 1)

                        (precision, recall) = (self.P(preds, y), self.R(preds, y))
                        # macro average for emphasis on classes, not instances (counter class imbalance)
                        precision, recall = (
                            precision.item(),
                            recall.item(),
                        )  # extract value from 1-d torch.tensor output

                        # backward + optimize only if in training phase
                        if mode == "train":
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    current_loss += loss.item() * x.size(0)
                    current_corrects += torch.sum(preds == y)
                    current_precision += precision
                    current_recall += recall

                    batch_count += 1

                    # confusion matrix -> aggregate valacc
                    if mode=='val' and e==epochs-1 and flag_cm: # last epoch, first batch
                        all_preds = preds
                        all_labels = y
                        flag_cm = False
                    elif mode=='val' and e==epochs-1:
                        all_preds = torch.cat((all_preds, preds), dim=-1)
                        all_labels = torch.cat((all_labels, y), dim=-1)

                epoch_loss = current_loss / len(self.dataloaders_dict[mode].dataset)
                epoch_acc = current_corrects.double() / len(
                    self.dataloaders_dict[mode].dataset
                )
                epoch_precision = current_precision / batch_count
                epoch_recall = current_recall / batch_count

                # add to log
                log[f"{mode}-loss"].append(epoch_loss)
                log[f"{mode}-acc"].append(epoch_acc.item())
                log[f"{mode}-precision"].append(epoch_precision)
                log[f"{mode}-recall"].append(epoch_recall)

                print(
                    "{} Loss: {:.4f} Acc: {:.4f} Prec: {:.4f} Rec: {:.4f}".format(
                        mode, epoch_loss, epoch_acc, epoch_precision, epoch_recall
                    )
                )

                # log val_acc and deep copy the model again if it is an improvement
                if mode == "val":
                    val_acc.append(epoch_acc)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        self.best_model = copy.deepcopy(self.model_ft.state_dict())
                    elif (
                        epoch_acc < 1.01 * best_acc
                    ):  # next epoch less than 10 percent val-acc improvement
                        count_not_better += 1

            print("=========================================")

            # early stopping
            if early_stopping and count_not_better >= 10:
                print("Early Stopping: 10 epochs with no val-acc improvement...")
                break

        # save json log
        with open(f"./logs/log_resnet_{self.save_name}.json", "w") as fp:
            json.dump(log, fp)

        # save best model state dict
        self.model_ft.load_state_dict(self.best_model)
        torch.save(self.model_ft.state_dict(), self.BEST_MODEL_PATH)

        if save_cm: # confusion matrix of validation results on last epoch
            cm = confusion_matrix(all_labels.cpu(), all_preds.cpu())
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig(f"B_resnet_cm_val_{self.save_name}.pdf")


    def test_model(self, load_model: bool = False, save_cm: bool = True):
        # load best model

        if load_model:
            self.model_ft.load_state_dict(torch.load(self.BEST_MODEL_PATH))
            self.model_ft.to(self.device)
        else:
            try: self.model_ft.load_state_dict(self.best_model)
            except: raise Exception('No state dict cached. Re-run training or load from disk.')

        L, acc, prec, rec, batch_count = 0.0, 0, 0.0, 0.0, 0

        all_preds, all_labels = torch.tensor([]), torch.tensor([])
        flag_cm = True

        for x, y in self.dataloaders_dict["test"]:
            if x.shape[0] < self.batch_size:
                continue
            else:
                s = self.batch_size

            # print(f'SHAPE: x = {x.shape}, y = {y.shape}')
            x = torch.reshape(x, (s, 3, 28, 28))
            # y = y.to(self.device)

            self.optimizer.zero_grad()

            with torch.no_grad():
                y_pred = self.model_ft(x)
                loss = self.criterion(y_pred, y)

                _, preds = torch.max(y_pred, 1)

                (precision, recall) = (self.P(preds, y), self.R(preds, y))
                # macro average for emphasis on classes, not instances (counter class imbalance)
                precision, recall = (
                    precision.item(),
                    recall.item(),
                )  # extract value from 1-d torch.tensor output

            # statistics
            L += loss.item() * x.size(0)
            acc += torch.sum(preds == y)
            prec += precision
            rec += recall
            batch_count += 1

            # confusion matrix aggregation
            if flag_cm: # first batch
                all_preds = preds
                all_labels = y
                flag_cm = False
            else:
                all_preds = torch.cat((all_preds, preds), dim=-1)
                all_labels = torch.cat((all_labels, y), dim=-1)

        L /= len(self.dataloaders_dict["test"].dataset)
        acc = acc.double() / len(self.dataloaders_dict["test"].dataset)
        prec /= batch_count
        rec /= batch_count

        print(
            "TEST DATA results:\nLoss {:.4f} || Acc {:.4f} || Precision {:.4f} || Recall {:.4f}".format(
                L, acc, prec, rec
            )
        )

        if save_cm:
            cm = confusion_matrix(all_labels.cpu(), all_preds.cpu())
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig(f"B_resnet_cm_test_{self.save_name}.pdf")


class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        C: int,
        r = 16,
        activation = torch.nn.ReLU,
        scale_activation = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = torch.nn.Linear(C, int(C/r)) #torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Linear(int(C/r), C)
        self.activation = activation()
        self.scale_activation = scale_activation()
        self.C = C

    def _scale(self, input):
        scale = self.avgpool(input)
        scale = torch.reshape(scale, (-1, 1, 1, self.C))
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input):
        scale = self._scale(input)
        
        repeats = input.shape[-1]
        scale = torch.reshape(scale, (input.shape[0],self.C,1,1))
        scale = scale.repeat(1,1,repeats,repeats)
        
        se = scale * input
        return se
    
class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, input):
        return torch.reshape(input, self.shape)
    

class SqueezeExcitationResNet(PathResNet18):
    
    def __init__(
        self,
        num_classes: int = 9,
        batch_size: int = 64,
        feature_extract: bool = True,
        up_to_layer: int = 0,
        save_name: str = '',
        use_pretrained: bool = True,
        dr: float = 0,
        lr: float = 0.001,
        rho: float = 0.9
    ) -> None:
        super().__init__(num_classes=num_classes,
                         batch_size=batch_size,
                         feature_extract=feature_extract,
                         up_to_layer=up_to_layer,
                         save_name=save_name,
                         dr=dr,
                         lr=lr,
                         rho=rho,
                         )

        """layers = [] #OrderedDict([])
        for child in self.model_ft.children():           
            if child.children():
                for subchild in child.children():
                    if isinstance(subchild, torch.nn.Sequential):
                        for m in list(subchild.modules())[1:]:
                            layers.append(m)
                            if isinstance(m, BasicBlock):
                                in_channels = m.conv1.in_channels
                                out_channels = m.conv1.out_channels

                                '''SEBlock = torchvision.ops.SqueezeExcitation(
                                    input_channels=out_channels,
                                    squeeze_channels=out_channels,
                                    # ReLU activation
                                )
                                SE_list = list(SEBlock.modules())
                                SE_list[1].output_size = (7, 7)
                                SE_list[2].in_channels = in_channels
                                SE_list[2].kernel_size = (3,3)
                                SE_list[3].kernel_size = (3,3)
                                SEBlock_final = SE_list[0]
                                
                                layers.append(SEBlock_final)'''

                                layers.append(SqueezeExcitation(C = out_channels))
                    else:
                        layers.append(subchild)
            else:
                layers.append(child)"""
        
        layers = list(self.model_ft.children())
        senet_layers = list(layers[0].children())
        #print({k:v for v,k in enumerate(senet_layers)})
        senet_layers.insert(5, SqueezeExcitation(C=64))
        senet_layers.insert(7, SqueezeExcitation(C=128))
        senet_layers.insert(9, SqueezeExcitation(C=256))
        senet_layers.insert(11, SqueezeExcitation(C=512))
        senet_layers.insert(13, Reshape((self.batch_size,512)))
        
        senet_layers.append(layers[1])
        senet_layers.append(layers[2])
        
        self.model_ft = torch.nn.Sequential(*senet_layers)

        self.model_ft.to(self.device)
        
        print('\n\nMODEL with Squeeze-Excitation.......................\n\n')
        print(self.model_ft)
        

if __name__ == "__main__":
    try:
        model_class = sys.argv[1] # svm or resnet
        if model_class not in ['svm', 'resnet', 'squeeze-excitation']:
            raise Exception(f'model_class is "svm" or "resnet" or "squeeze-excitation, got {model_class}')
        mode = sys.argv[2]
        if mode not in ['train-val','test','all']:
            raise Exception(f'mode is "train-val","test","all", got {mode}')
        up_to_layer = int(sys.argv[3]) # for resnet
        save_name = sys.argv[4] # save logs and paths
        if model_class != 'svm':
            dropout = float(sys.argv[5])
            lr = float(sys.argv[6])
            rho = float(sys.argv[7])
            epoch_num = int(sys.argv[8])
    except:
        model_class = 'squeeze-excitation'
        mode = 'all'
        up_to_layer = 0
        save_name = ''
        dropout = 0.1
        lr = 0.001
        rho = 0.9
        epoch_num = 50

    if model_class == 'svm':
        s = SVM_Path()
        s.set_classifier()
        s.fit_classifier()

        if mode == 'train-val':
            s.predict_classifier("train", save=True)
            s.predict_classifier("val", save=True)
        elif mode == 'test':
            s.predict_classifier("test", save=True)
        else:
            s.predict_classifier("train", save=True)
            s.predict_classifier("val", save=True)
            s.predict_classifier("test", save=True)

    elif model_class == 'resnet':
        rn = PathResNet18(feature_extract=True, up_to_layer=up_to_layer, save_name=save_name, post_layers=True, dr=dropout, lr=lr, rho=rho)
        
        if mode == 'train_val':
            rn.train_model(epochs=epoch_num)
        elif mode == 'test':
            rn.BEST_MODEL_PATH = './best_model/best_model_resnet_lr01rho9resnet.pty'
            rn.test_model(load_model=True)
        else:
            rn.train_model(epochs=epoch_num)
            rn.test_model()

    elif model_class == 'squeeze-excitation':
        se = SqueezeExcitationResNet(feature_extract=False, up_to_layer=up_to_layer, save_name=save_name, dr=dropout, lr=lr, rho=rho)
        
        if mode == 'train_val':
            se.train_model(epochs=epoch_num)
        elif mode == 'test':
            se.BEST_MODEL_PATH = './best_model/best_model_resnet_lr001rho99SENet_fr0.pty'
            se.test_model()
        else:
            se.train_model(epochs=epoch_num)
            se.test_model()