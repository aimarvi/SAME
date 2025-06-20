import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import svm
from tqdm import tqdm

'''
TODO:

finish function for running svm
decide on id: 
    same_img vs same_label vs diff_label?
    one vs two?
    none vs one vs two?
'''
# set up svm
num_ids = 3
num_reps = 100
num_samples = num_reps * num_ids

indTest = np.arange(0,num_samples,num_reps)
indAll = np.arange(0,num_samples)

perf_fold = np.zeros(shape=(num_reps,))

x = np.arange(0,num_ids)
trainCat = np.repeat(x,num_reps-1)
