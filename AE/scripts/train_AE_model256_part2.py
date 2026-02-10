import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, default_collate
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from patchify import patchify
import numpy as np
import sys
from architectures import AE_model256 as ae
import time
import random

stain = 'mye'
patch_size = 256
batch_size = 64 # 512
num_epochs = 6

num_patches_per_section = 25600 #102400 #400384  #500224

ae_name = 'AE_model256'
# ae_name = 'AE_n9_'+stain+'_w'+str(patch_size)+'_LS256_batch'+str(batch_size)+'_batchnorm'
folder_save = f'/scratch/project_2008349/AE/data/ae_trained/{stain}/'

# -------------------------------------------------
# load those png_files for continuing training:
save_path = "/.../AE/data/datasets_training/"

with open(os.path.join(save_path, f"{ae_name}_png_files_train.txt"), "r") as f:
    png_files_train = f.read().splitlines()

with open(os.path.join(save_path, f"{ae_name}_png_files_val.txt"), "r") as f:
    png_files_val = f.read().splitlines()

# function to patch the sections at the desired window size (non overlapping)
def patchifying(file_path, patch_size, num_patches=num_patches_per_section):
    image = Image.open(file_path).convert('L')
    image = np.array(image)
    patches = patchify(image, (patch_size, patch_size), step=patch_size)
    patches_flatten = patches.reshape(-1, patch_size, patch_size)
    np.random.shuffle(patches_flatten)
    if len(patches_flatten) > num_patches:
        patches_flatten = patches_flatten[:num_patches]
    else:
        raise ValueError(f"Number of patches {len(patches_flatten)} is less than {num_patches}.")
    return patches_flatten

# to deal iteratively with opening the images, patchifying, and training
class my_dataset(Dataset):
    def __init__(self, file_paths, patch_size):
        self.file_paths = file_paths
        print(f'paths are: {file_paths}')
        self.patch_size = patch_size
        # this is just a decision, for not counting
        self.total_patches = num_patches_per_section*len(file_paths)

        self.current_patches = None
        self.current_patch_count = 0
        self.current_image_idx = -1

    def __len__(self):
        return self.total_patches

    def load_one_image(self):
        self.current_patches = patchifying(self.file_paths[self.current_image_idx], self.patch_size)
        self.current_patch_count = len(self.current_patches)

    def __getitem__(self, idx):
        if self.current_patch_count == 0:
            self.current_image_idx += 1
            if self.current_image_idx >= len(self.file_paths): # it shouldnt enter here ideally but just in case
                self.current_image_idx=0
                print('it went over the file path')
            self.load_one_image()

        if self.current_patch_count > 0:
            patch = self.current_patches[self.current_patch_count - 1]
            self.current_patch_count -= 1
            self.current_patches = self.current_patches[:self.current_patch_count]
            return torch.tensor(patch / 255, dtype=torch.float32).unsqueeze(0)


# set up DataLoader for dataset
dataset_training = my_dataset(png_files_train, patch_size)
dataset_val = my_dataset(png_files_val, patch_size)
# testing now shuffleing totta
data_loader_training = DataLoader(dataset_training, batch_size=batch_size, shuffle=True, num_workers=1)
data_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=1)

# sizes of the datasets (for the errors)
num_training_samples = len(dataset_training)
num_val_samples = len(dataset_val)

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

# ---------------------------------
def save_checkpoint(epoch, model, optimizer, best_loss, patience, train_loss_1_8, val_loss_1_8):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'patience': patience,
        'train_loss_1_8': train_loss_1_8,
        'val_loss_1_8': val_loss_1_8
    }
    checkpoint_path = os.path.join(folder_save, ae_name + "_checkpoint_part2.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")
# ---------------------------------
# model, loss function, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AE = ae.instance().to(device)
loss_funct = nn.MSELoss()
optimizer = optim.Adam(AE.parameters(), lr=1e-3)  # lr is the default, not sure if i should play with it
# -------------------------------------------------
def load_checkpoint(model, optimizer):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Continue from the next epoch
        best_loss = checkpoint['best_loss']
        patience = checkpoint['patience']
        train_loss_1_8 = checkpoint['train_loss_1_8']
        val_loss_1_8 = checkpoint['val_loss_1_8']

        print(f"Checkpoint loaded. Resuming from epoch {start_epoch} with best loss {best_loss:.4f}")
        return model, optimizer, start_epoch, best_loss, patience, train_loss_1_8, val_loss_1_8
    else:
        print("No checkpoint found, starting fresh training.")
        return model, optimizer, 0, float('inf'), 10, [], []


checkpoint_path = os.path.join(folder_save, ae_name + "_checkpoint.pth")
AE, optimizer, start_epoch, best_loss, patience, train_loss_1_8, val_loss_1_8 = load_checkpoint(AE, optimizer)
# -------------------------------------------------

save_path = folder_save + ae_name + '.pth'

start_time = time.time()
train_loss_per_epoch = []
val_loss_per_epoch = []

# best_loss = float('inf')
# best_model = None
# patience = 10
stop_all = False

# lets do during 1 epoch the following:
# train in 1/8 of data, then validate, then train in 1/8 of data, and validate, etc

total_batches = len(data_loader_training)
validation_interval = total_batches // 8


# for some reason it didnt work, so I'll save new train and val_loss:
train_loss_1_8 = []
val_loss_1_8 = []

for epoch in range(num_epochs):

    AE.train()
    train_loss = 0.0
    val_loss = 0.0

    train_loss_per_batch = []
    val_loss_per_batch = []

    for batch_idx, batch in enumerate(data_loader_training):

        batch = batch.to(device)
        optimizer.zero_grad()

        # forward pass
        outputs = AE(batch)
        loss = loss_funct(outputs, batch)

        # acumulate training loss
        train_loss_per_batch.append(loss.item())

        # backward pass
        loss.backward()
        optimizer.step()

        # run evaluation every 1/8 training data:
        if (batch_idx + 1) % validation_interval == 0:

            # append mean training loss for this interval
            mean_train_loss = np.mean(train_loss_per_batch)
            train_loss_1_8.append(mean_train_loss)
            train_loss_per_batch = []

            AE.eval()
            temp_val_loss = 0.0

            with torch.no_grad():
                for val_batch in data_loader_val:
                    val_batch = val_batch.to(device)

                    # forward pass
                    val_outputs = AE(val_batch)
                    val_loss = loss_funct(val_outputs, val_batch)

                    temp_val_loss += val_loss.item()
                    val_loss_per_batch.append(val_loss.item())

            mean_val_loss = np.mean(val_loss_per_batch)
            val_loss_1_8.append(mean_val_loss)
            val_loss_per_batch = []

            if temp_val_loss < best_loss:
                best_loss = temp_val_loss
                # save the model's state:
                best_model = AE.state_dict()
                patience = 10
            else:
                patience -= 1
                if patience == 0:
                    stop_all = True
                    break

        if stop_all:
            break

        # set the model back to train:
        AE.train()


    # checkpoint:
    save_checkpoint(epoch, AE, optimizer, best_loss, patience, train_loss_1_8, val_loss_1_8)
    torch.cuda.empty_cache()

    if stop_all:
        print('validation loss not improving, so we stop training')
        torch.save(best_model, save_path)
        print(f'model saved to {save_path}')

        # train_loss_per_epoch.extend(train_loss_per_batch)
        # val_loss_per_epoch.extend(val_loss_per_batch)

        np.savetxt(folder_save + ae_name + '_train_loss_part2.txt', np.array(train_loss_1_8))
        np.savetxt(folder_save + ae_name + '_val_loss_part2.txt', np.array(val_loss_1_8))

        partial_time = time.time()
        total_time = partial_time - start_time
        total_time_h = total_time / 3600
        print('hours needed', total_time_h, '\n')
        break


    print("one epoch.")
    torch.save(best_model, save_path)
    print(f"best model saved to {save_path}")

    partial_time = time.time()
    total_time = partial_time - start_time
    total_time_h = total_time / 3600
    print('hours needed', total_time_h, '\n')

    # Print the training and validation loss for the epoch
    # print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")
    np.savetxt(folder_save + ae_name + '_train_loss_part2.txt', np.array(train_loss_1_8))
    np.savetxt(folder_save + ae_name + '_val_loss_part2.txt', np.array(val_loss_1_8))


if not stop_all:
    print(f"Training complete, {num_epochs} epochs reached, not best model achieved")
    torch.save(best_model, save_path)
    print(f"best model saved to {save_path}")

    save_checkpoint(epoch, AE, optimizer, best_loss, patience, train_loss_1_8, val_loss_1_8)
    print('checkpoint saved\n')

    partial_time = time.time()
    total_time = partial_time - start_time
    total_time_h = total_time / 3600
    print('hours needed', total_time_h, '\n')

    np.savetxt(folder_save + ae_name + '_train_loss_part2.txt', np.array(train_loss_1_8))
    np.savetxt(folder_save + ae_name + '_val_loss_part2.txt', np.array(val_loss_1_8))
