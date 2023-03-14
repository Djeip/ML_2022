from zz.model import ConvLayerDownsampleLayer
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import torch


def load_image_rgb(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (256, 256))
    img = img.astype('float')
    img = img / 256
    return img


color_image = load_image_rgb(r'C:\Users\Lenovo\Pictures\UbisoftConnect\astronaut.jpg')
red, blue, green = color_image[:, :, 0], color_image[:, :, 1], color_image[:, :, 2]
color_tensors_1 = {
    'red': torch.Tensor(red.reshape(1, 1, 256, 256)),
    'blue': torch.Tensor(blue.reshape(1, 1, 256, 256)),
    'green': torch.Tensor(green.reshape(1, 1, 256, 256))
}
plt.axis('off')
plt.imshow(color_image)
plt.show()

if __name__ == '__main__':
