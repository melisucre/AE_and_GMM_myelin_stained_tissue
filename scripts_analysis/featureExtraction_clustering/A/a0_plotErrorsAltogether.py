import os
import glob
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def read_file(file_path):

    with open(file_path, 'r') as file:
        losses = [float(line.strip()) for line in file]
    return losses


ae_names = ['AE_model128', 'AE_model256']
CB_colors = ['#ff7f00', '#4daf4a']

def plot_errors_batches(ae_names):

    train_file_r = []
    val_file_r = []

    ae_labels = ['model128', 'model256']
    colors = CB_colors

    for ae_name in ae_names:
        folder = '.../AE/data/AE_trained/'
        folder_save = '.../AE/data/AE_usage/'

        files = glob.glob(os.path.join(folder, f'{ae_name}*.txt'))
        train_file = [f for f in files if 'train_loss.txt' in f]
        val_file = [f for f in files if 'val_loss.txt' in f]

        train_file_r.append(read_file(train_file[0]))
        val_file_r.append(read_file(val_file[0]))

    n_values_per_epoch = 8  # 8 values per epoch
    num_epochs = 20
    epoch_boundaries = [i * n_values_per_epoch for i in range(1, num_epochs)]
    epoch_ticks = [i * n_values_per_epoch for i in range(num_epochs)]
    epoch_labels = [f"{i + 1}" for i in range(num_epochs)]
    plt.figure(figsize=(10, 6))

    for i, (train_loss, val_loss) in enumerate(zip(train_file_r, val_file_r)):
        plt.plot(train_loss, label=f'{ae_labels[i]} training loss', color=colors[i], alpha=0.7)
        plt.plot(val_loss, label=f'{ae_labels[i]} validation loss', color=colors[i], linestyle='dashed', alpha=0.7)

    plt.ylim(0, 0.01)
    plt.ylabel('loss', fontsize=14, rotation=90)#, labelpad=10)
    plt.xlabel('epochs', fontsize=14)
    plt.title('training and validation losses of AE-models over epochs', fontsize=16)
    plt.legend(fontsize=12)
    custom_lines = [
        Line2D([0], [0], color=CB_colors[0], linestyle='-', lw=1.5, label='model128 training loss'),
        Line2D([0], [0], color=CB_colors[0], linestyle='--', lw=1.5, label='model128 validation loss'),
        Line2D([0], [0], color=CB_colors[1], linestyle='-', lw=1.5, label='model256 training loss'),
        Line2D([0], [0], color=CB_colors[1], linestyle='--', lw=1.5, label='model256 validation loss')
    ]
    plt.legend(handles=custom_lines, fontsize=12, loc='upper right')#, bbox_to_anchor=(1, 1.01))

    plt.grid(True)

    # Add vertical lines for each epoch boundary
    for boundary in epoch_boundaries:
        plt.axvline(x=boundary, color='grey', linestyle='-', linewidth=0.003)

    plt.xticks(epoch_ticks, epoch_labels, fontsize=10, rotation=0)
    plt.xlim(0, 143)

    plt.savefig(os.path.join(folder_save, 'errors_model128_model256.png'), bbox_inches='tight', dpi=300)
    plt.close()


plot_errors_batches(ae_names)
