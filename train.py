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

#Visualize some input images


num_images = 10

# Just for fun choose random car / not-car indices and plot example images   
cars_samples = random.sample(list(cars), num_images)
notcar_samples = random.sample(list(notcars), num_images)
    
# Read in car / not-car images
car_images = []
notcar_images = []
for sample in cars_samples:
    car_images.append(mpimg.imread(sample))
    
for sample in notcar_samples:
    notcar_images.append(mpimg.imread(sample))


orient = 9
pix_per_cell = 8
cell_per_block = 2

print("start hog")

car_features, hog_image = get_hog_features(cv2.cvtColor(car_images[1], cv2.COLOR_RGB2GRAY), orient, pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=True)

notcar_features, notcar_hog_image = get_hog_features(cv2.cvtColor(notcar_images[2], cv2.COLOR_RGB2GRAY), orient, pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=True)

print("end hog")