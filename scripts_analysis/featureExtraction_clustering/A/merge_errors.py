import os
import glob
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np


def read_file(file_path):

    with open(file_path, 'r') as file:
        losses = [float(line.strip()) for line in file]
    return losses


def together_part1_part2(ae_name):

    folder = '.../AE/data/AE_trained'

    files = glob.glob(os.path.join(folder, f'{ae_name}*.txt'))

    train_file = [f for f in files if 'train_loss_part1' in f]
    train_file2 = [f for f in files if 'train_loss_part2' in f]
    train_file3 = [f for f in files if 'train_loss_part3' in f]
    val_file = [f for f in files if 'val_loss_part1' in f]
    val_file2 = [f for f in files if 'val_loss_part2' in f]
    val_file3 = [f for f in files if 'val_loss_part3' in f]

    train_file_r = read_file(train_file[0])
    train_file_r2 = read_file(train_file2[0])
    train_file_r3 = read_file(train_file3[0])
    val_file_r = read_file(val_file[0])
    val_file_r2 = read_file(val_file2[0])
    val_file_r3 = read_file(val_file3[0])

    train_file = train_file_r + train_file_r2 + train_file_r3
    val_file = val_file_r + val_file_r2 + val_file_r3

    path_save_train = f'{folder}/{ae_name}_train_loss.txt'
    path_save_val =f'{folder}/{ae_name}_val_loss.txt'

    with open(path_save_train, 'w') as file:
        for item in train_file:
            file.write(f"{item}\n")

    with open(path_save_val, 'w') as file:
        for item in val_file:
            file.write(f"{item}\n")

together_part1_part2('AE_model128')

