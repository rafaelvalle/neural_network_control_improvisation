from scipy.ndimage import imread

BATCH_SIZE = 64
IMG_SHAPE = (64, 64)
img = imread(img_path, flatten=True)
img = img.reshape((BATCH_SIZE,)+IMG_SHAPE)
