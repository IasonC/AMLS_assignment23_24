import argparse, os

parser = argparse.ArgumentParser(prog='main.py')
parser.add_argument('-t', '--task', choices=['A','B'], required=True)
parser.add_argument('-m', '--model_class', default='svm', choices = ['svm', 'svm-b', 'svm-be', 'resnet', 'squeeze-excitation'])
parser.add_argument('-M', '--mode', default='train-val', choices=['train-val','test','all'], required=True)

# Optional args for Task A
parser.add_argument('-save', '--save_plts', required=False)
parser.add_argument('-C', '--SVM_C', type=float, required=False)
parser.add_argument('-gamma', '--SVM_gamma', required=False)
parser.add_argument('-B', '--SVM_B', required=False)
parser.add_argument('-p', '--SVM_p', required=False)
parser.add_argument('-CV', '--SVM_cross_val', choices=['True', 'False'], required=False)

# Optional args for Task B
parser.add_argument('-L', '--up_to_layer', required=False)
parser.add_argument('-saveN', '--save_name', required=False)
parser.add_argument('-dr', '--dropout', required=False)
parser.add_argument('-lr', '--learning_rate', required=False)
parser.add_argument('-rho', '--momentum', required=False)
parser.add_argument('-epoch', '--epoch_num', required=False)


args = parser.parse_args()

if args.task == 'A': # task A
    model = args.model_class
    assert(model in ['svm','svm-b','svm-be']), Exception('In Task A, model_class is ["svm","svm-b","svm-be"]')
    
    mode = args.mode
    save = args.save_plts
    
    C = args.SVM_C
    gamma = args.SVM_gamma

    if model in ['svm-b','svm-be']:
        B = args.SVM_B
        p = args.SVM_p

        if model == 'svm-b':
            os.system(f'python ./A/taskA.py {model} {mode} {save} {C} {gamma} {B} {p}')
        else:
            CV = args.SVM_cross_val
            os.system(f'python ./A/taskA.py {model} {mode} {save} {C} {gamma} {B} {p} {CV}')
    else:
        os.system(f'python ./A/taskA.py {model} {mode} {save} {C} {gamma}')

else: # task B
    model = args.model_class
    assert(model in ['svm','resnet','squeeze-excitation']), Exception('In Task B, model_class is ["svm","resnet","squeeze-excitation"]')

    mode = args.mode
    up_to_layer = args.up_to_layer
    saveN = args.save_name

    if model == 'svm':
        os.system(f'python ./A/taskA.py {model} {mode} {up_to_layer} {saveN}')
    else:   
        dr = args.dropout
        lr = args.learning_rate
        rho = args.momentum
        epoch = args.epoch_num

        os.system(f'python ./A/taskA.py {model} {mode} {up_to_layer} {saveN} {dr} {lr} {rho} {epoch}')