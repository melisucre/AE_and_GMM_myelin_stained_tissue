import numpy as np
# import time
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

path = '.../AE/image_preprocessing/data/mtbi/ISM_XX/tiff_files/mye/glass_6/downsampled_by_16.tif'
image = Image.open(path).convert('L')
image = np.array(image)


# # let's cut the slices:
# # first y coordinates, then x coord

2, 5, 44, 66
3, 71, 45, 131
8, 134, 52, 195
50, 5, 92, 65
48, 72, 92, 134
51, 137, 94, 198


tile = image[50:92, 5:65]
tile_nb = tile.astype('uint8')  # / 255.
plt.imshow(tile_nb, cmap='gray', vmin=0, vmax=255)


# plt.savefig('.../AE/image_preprocessing/trials/yes.png')
# plt.close()



