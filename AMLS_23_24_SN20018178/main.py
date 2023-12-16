import argparse

parser = argparse.ArgumentParser(prog='main.py')
parser.add_argument('-t', '--task', choices=['A','B'])
parser.add_argument('-m', '--mode', default='train-val', choices=['train-val','test'])
args = parser.parse_args()

# implement TASK
# call args.task dir ./A/ or ./B/

# implement MODE
# args.mode