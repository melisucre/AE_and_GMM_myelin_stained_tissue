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

num_patches_per_section = 25600 #400384  #500224

ae_name = 'AE_model256'
# ae_name = 'AE_n9_'+stain+'_w'+str(patch_size)+'_LS256_batch'+str(batch_size)+'_batchnorm'
folder_save = f'/.../AE/data/ae_trained/{stain}/'

# paths to the images
data_path = '/.../AE/data/'
# animal_ids = ['ISM_24', 'ISM_27', 'ISM_29', 'ISM_32', 'ISM_33', 'ISM_34', 'ISM_35', 'ISM_36', 'ISM_37', 'ISM_38']
animal_ids_sham = ['ISM_27', 'ISM_29', 'ISM_35', 'ISM_38']
animal_ids_mtbi = ['ISM_24', 'ISM_32', 'ISM_33', 'ISM_34', 'ISM_36']

# lets randomize the mtbi ids and pick one for validation:
random.shuffle(animal_ids_mtbi)
val_animals = [animal_ids_mtbi[0]]

# lets define the animal ids for training:
train_animals = animal_ids_mtbi[1:] + animal_ids_sham
random.shuffle(train_animals)

# lets get the image file paths given the animal ids
def get_image_paths(animal_list):
    image_paths = []
    for animal_id in animal_list:
        animal_dir = os.path.join(data_path, f'{animal_id}/{stain}_renorm/')
        png_files = glob.glob(os.path.join(animal_dir, '*.png'))
        image_paths.extend(png_files)
    return image_paths

png_files_train = get_image_paths(train_animals)
png_files_val = get_image_paths(val_animals)

random.shuffle(png_files_train)
random.shuffle(png_files_val)

# save those png_files for later continuing training:
save_path = "/.../AE/data/datasets_training/"
os.makedirs(save_path, exist_ok=True)

with open(os.path.join(save_path, f"{ae_name}_png_files_train.txt"), "w") as f:
    f.write("\n".join(png_files_train))

with open(os.path.join(save_path, f"{ae_name}_png_files_val.txt"), "w") as f:
    f.write("\n".join(png_files_val))


# function to patch the sections at the desired window size (non overlapping)
def patchifying(file_path, patch_size, num_patches=num_patches_per_section):
    image = Image.open(file_path).convert('L')
    image = np.array(image)

    mask_path = file_path.replace("mye_renorm", "mye_masks").replace(".png", "_mask.png")
    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask)

    # Patchify image and mask
    image_patches = patchify(image, (patch_size, patch_size), step=patch_size)
    mask_patches = patchify(mask, (patch_size, patch_size), step=patch_size)
    # Flatten both sets of patches
    image_patches = image_patches.reshape(-1, patch_size, patch_size)
    mask_patches = mask_patches.reshape(-1, patch_size, patch_size)

    # Classify patches
    foreground_indices = [i for i, m in enumerate(mask_patches) if np.any(m > 0)]
    background_indices = [i for i, m in enumerate(mask_patches) if not np.any(m > 0)]

    # Shuffle both for randomness
    np.random.shuffle(foreground_indices)
    np.random.shuffle(background_indices)

    # Select from both sets
    selected_indices = foreground_indices[:num_patches]
    if len(selected_indices) < num_patches:
        n_needed = num_patches - len(selected_indices)
        if len(background_indices) < n_needed:
            raise ValueError(f"Not enough total patches to reach {num_patches}")
        selected_indices += background_indices[:n_needed]

    selected_patches = image_patches[selected_indices]

    return selected_patches

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
# here I change to false the shuffle because later we'll need to continue training, and those have been shuffled already
data_loader_training = DataLoader(dataset_training, batch_size=batch_size, shuffle=True, num_workers=1)
data_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=1)

# sizes of the datasets (for the errors)
num_training_samples = len(dataset_training)
num_val_samples = len(dataset_val)

sizes_path = folder_save + ae_name + '_sizesDatasets.txt'
with open(sizes_path, 'w') as f:
    f.write(f"number of training samples: {num_training_samples}\n")
    f.write(f"number of validation samples: {num_val_samples}\n")


# ---------------------------------
checkpoint_path = os.path.join(folder_save, ae_name + "_checkpoint.pth")
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
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

# ---------------------------------

# model, loss function, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AE = ae.instance().to(device)
loss_funct = nn.MSELoss()
optimizer = optim.Adam(AE.parameters(), lr=1e-3)  # lr is the default, not sure if i should play with it

save_path = folder_save + ae_name + '.pth'

start_time = time.time()
train_loss_per_epoch = []
val_loss_per_epoch = []

best_loss = float('inf')
best_model = None
patience = 10
stop_all = False

# lets do during 1 epoch the following:
# train in 1/8 of data, then validate, then train in 1/8 of data, and validate, etc

total_batches = len(data_loader_training)
validation_interval = total_batches // 8

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

        np.savetxt(folder_save + ae_name + '_train_loss.txt', np.array(train_loss_1_8))
        np.savetxt(folder_save + ae_name + '_val_loss.txt', np.array(val_loss_1_8))

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
    np.savetxt(folder_save + ae_name + '_train_loss.txt', np.array(train_loss_1_8))
    np.savetxt(folder_save + ae_name + '_val_loss.txt', np.array(val_loss_1_8))


if not stop_all:
    print(f"Training complete, {num_epochs} epochs reached")
    torch.save(best_model, save_path)
    print(f"best model saved to {save_path}")

    save_checkpoint(epoch, AE, optimizer, best_loss, patience, train_loss_1_8, val_loss_1_8)
    print('checkpoint saved\n')

    partial_time = time.time()
    total_time = partial_time - start_time
    total_time_h = total_time / 3600
    print('hours needed', total_time_h, '\n')

    np.savetxt(folder_save + ae_name + '_train_loss_p1.txt', np.array(train_loss_1_8))
    np.savetxt(folder_save + ae_name + '_val_loss_p1.txt', np.array(val_loss_1_8))
