from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import random
import glob
import time

from utilities import *


#Read cars and not-cars images

#Data folders
vehicles_dir =     'D:/badours/Data/vehicles/'
non_vehicles_dir = 'D:/badours/Data/non-vehicles/'

# images are divided up into vehicles and non-vehicles
cars = []
notcars = []

print("Start Loading")

# Read vehicle images
images = glob.iglob(vehicles_dir + '/**/*.png', recursive=True)

print("End Loading")


for image in images:
        cars.append(image)
        
# Read non-vehicle images
images = glob.iglob(non_vehicles_dir + '/**/*.png', recursive=True)

for image in images:
        notcars.append(image)
    
data_info = data_look(cars, notcars)

