import os
import csv
import cv2
import numpy as np
import random
import sklearn
import matplotlib.image as mpimg
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Lambda, Cropping2D, Dropout, Activation


### Read data from csv file
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split the data
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


### Augmented images and steering angles generator
def generator(samples, batch_size=16):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            augmented_images = []
            augmented_angles = []
            for batch_sample in batch_samples:
                # Images and angles from center camera
                name_center = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = mpimg.imread(name_center)
                center_angle = float(batch_sample[3])
                augmented_images.append(center_image)
                augmented_angles.append(center_angle)
                               
                # From left camera
                name_left = './data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = mpimg.imread(name_left)
                left_angle = center_angle + 0.2
                augmented_images.append(left_image)
                augmented_angles.append(left_angle)
                           
                # From right camera
                name_right = './data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = mpimg.imread(name_right)
                right_angle = center_angle - 0.2
                augmented_images.append(right_image)
                augmented_angles.append(right_angle)
                
                
                # Flipped images from center camera
                flip_image_center = cv2.flip(center_image,1)
                augmented_images.append(flip_image_center)
                augmented_angles.append(center_angle*-1.0)
                
                # From left camera
                flip_image_left = cv2.flip(left_image,1)
                augmented_images.append(flip_image_left)
                augmented_angles.append(left_angle*-1)
                
                # From right camera
                flip_image_right = cv2.flip(right_image,1)
                augmented_images.append(flip_image_right)
                augmented_angles.append(right_angle*-1)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=16

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


### Convolutional Neural Network architecture with Keras
# Initialization
model = Sequential()

# Normalization of data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# Cropping unnecessary parts of image
model.add(Cropping2D(cropping=((70,25), (0,0))))

# First convolution layer
model.add(Conv2D(24,(5,5),strides=(2,2),activation="elu"))

# Second convolution layer
model.add(Conv2D(36,(5,5),strides=(2,2),activation="elu"))

# Third convolution layer
model.add(Conv2D(48,(5,5),strides=(2,2),activation="elu"))

# Fourth convolution layer, strides 1x1
model.add(Conv2D(64,(3,3),activation="elu"))

# Fifth convolution layer, strides 1x1
model.add(Conv2D(64,(3,3),activation="elu"))

# Flatten layer
model.add(Flatten())

#model.add(Dropout(0.8))

# First fully connected layer
model.add(Dense(100, activation='elu'))

# Second fully connected layer
model.add(Dense(50, activation='elu'))

# Third fully connected layer
model.add(Dense(10, activation='elu'))

# Final fully connected layer
model.add(Dense(1))

# Compile, optimization and generator
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

# Save the model
model.save('model.h5')