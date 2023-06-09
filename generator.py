from keras.utils import Sequence, to_categorical
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import glob

def get_files(pattern):
    return glob.glob(pattern)

class Img2ImgGenerator(Sequence):
    def __init__(self, x_files, y_files, batch_size, x_shape=None, y_shape=None,real_world = False):
        self.x_files, self.y_files, self.x_shape, self.y_shape = x_files, y_files, x_shape, y_shape
        self.real_world = real_world
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x_files) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_files[idx * self.batch_size:(idx + 1) * self.batch_size]

        X = [imread(file_name) for file_name in batch_x]
        Y = [imread(file_name) for file_name in batch_y]
        if self.real_world:
            pass
        else:
            Y = to_categorical(np.array(rgb2gray(Y)), 6)

        if self.x_shape is not None:
            X = [resize(img, self.x_shape) for img in X]
        if self.y_shape is not None:
            Y = [resize(img, (288, 512,6)) for img in Y]
        # print(np.shape(X))
        # print(np.shape(Y))

        return np.array(X), np.array(Y)

    @staticmethod
    def _resize(img, shape):
        if shape is None:
            return img
        else:
            return resize(img, shape)

