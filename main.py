import numpy as np
import cv2
import time
from utilities import *
from moviepy.editor import VideoFileClip
from functools import reduce
import pickle
from scipy.ndimage.measurements import label
import sys


dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
print(dist_pickle)
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
