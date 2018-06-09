import torch
from torch.autograd import Variable
import numpy as np


def ones_tensor(size):
    t = Variable(torch.ones(size, 1))
    if torch.cuda.is_available():
        return t.cuda()
    return t


def zeros_tensor(size):
    t = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available():
        return t.cuda()
    return t


def image_normalize(img):
    img_min = img.min()
    img_max = img.max()
    if img_min < img_max:
        img = (img - img_min) / (img_max - img_min)
    return img


def to_image_tensor(input_tensor, n_rows=9, n_cols=9, width=64, height=64):
    images_to_plot = n_rows * n_cols
    tensor_to_plot = input_tensor[:images_to_plot, :]
    image_grid = np.zeros((3, n_rows * height, n_cols*width))
    for i in range(images_to_plot):
        r = i//n_rows
        c = i%n_cols
        image_grid[:, r * height:(r + 1) * height, c * width:(c + 1) * width] = tensor_to_plot[i, :].reshape(3, width, height)
    return torch.Tensor(image_normalize(image_grid))