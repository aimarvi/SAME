import os
import random

import torch

from PIL import Image

from torchvision import datasets, transforms

'''
TODO:

finish this script

copy and adapt code in the .ipynb stimuli files
'''

# basic stimulus parameters
canvas_size = 400
canvas_color = (0, 0, 0) # black background
image_size = 100  # resize 28x28 to 100x100
num_imgs = 10
pad = 20
save_root = './object_stim'

# 4 corner positions
positions = {
    'top_left': (pad, pad),
    'top_right': (canvas_size - image_size - pad, pad),
    'bottom_left': (pad, canvas_size - image_size - pad),
    'bottom_right': (canvas_size - image_size - pad, canvas_size - image_size - pad)
}

def create_stimulus(img_paths, mode='same_image'):
'''
generates stimuli with two images on a black background. images can either be:
    1. the same image
    2. two exemplars of the same label
    3. two exemplars of different labels

args:
    img_paths (dict): keys are each label/category. values are all image paths of that label
    mode (str); one of {same_image, same_label, different_label}
'''
