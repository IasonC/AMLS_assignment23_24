## Project File Organisation

- A: includes Task A code ```taskA.py```, confusion matrices included in Report
    - svm: includes results for hyperparameter tuning (Rand Search or Cross-Validation) for svm, svm-b, svm-be
- B: includes Task B code ```taskB.py```, confusion matrices included in Report
    - logs: log of training and validation metrics (Acc, P, TPR) per epoch of ResNet and SENet training/validation, plus a plotting script ```plot_logs.py``` to generate the hyperparameter Figures for ResNet and SENet lr and rho (0.01-0.9, 0.001-0.9, 0.01-0.99, 0.001-0.99)
    - best_model: includes saved models .pty which can be loaded for inference (done in ```taskB.py``` for ```-m test```)
- Datasets: empty; ```pneumoniamnist.npz``` and ```pathmnist.npz``` are placed here
- ```main.py```: used for running ```taskA.py``` and ```taskB.py```

## Run Instructions

To run Task A, run ```main.py -t A <...>``` with task argument \{-t A\}, required arguments \{-M, -m\}, and optional arguments \{-save, -C, -gamma, -p, -CV\} (as given in ```A/taskA.py```)

To run Task B, run ```main.py -t B <...>``` with task argument \{-t B\}, required arguments \{-M, -m\}, and optional arguments \{-L, -saveN, -dr, -lr, -rho, -epoch\} (as given ```in B/taskB.py```)

* Required Packages *
```
torch // I use 2.1.0+cu121 (CUDA-enabled for GPU)
torchvision
torchmetrics
medmnist
sklearn
scipy
numpy
matplotlib
tqdm
copy
```

## Important note on Paths
In ```taskA.py``` and ```taskB.py```, the ```Data_Pneumonia``` and ```Data_Path``` classes access Datasets/ and download/read the data. The *path* to access the data should be ```../Datasets/<data.npz>``` from the perspective of the Task A and B files.

However, for the ```main.py``` file here to call ```taskA.py``` and ```taskB.py```, which are located in different directories, I have set the *path with respect to main here*, such that the path in the dataloader classes of ```taskA.py``` and ```taskB.py``` is set as ```Datasets/<data.npz>```. This is ok to use Task A and Task B scripts from ```main.py```.

To run Task A and B correctly _from within the ./A or ./B directories_, set the following lines:
- ```taskA.py``` Line 26 --> ```self.DATA_ROOT = '../Datasets/'```
- ```taskB.py``` Line 26 --> ```self.DATA_ROOT = '../Datasets/'```