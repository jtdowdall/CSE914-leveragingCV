import cv2
import glob
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
import skimage
from matplotlib.colors import hsv_to_rgb
from keras.models import Sequential, load_model
from keras.layers import Convolution2D
from keras.layers import Dense, Dropout,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers import ELU
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split
import sys
import os

IMG_HEIGHT = 66
IMG_WIDTH = 200

y = pd.read_csv('data/info.csv')
y = y.sample(frac=1, random_state=42).reset_index(drop=True)
X = []
print('Loading data...')
total = len(y)
for index, row in y.iterrows():
    print('{}/{}'.format(index, total))
    try:
        img = cv2.imread(row['path'])
        X.append(img)
    except:
        y = y.drop(index)
y = y.drop('path', axis=1)


# MODEL
inputShape = X[0].shape

model = Sequential()
try:
    print(sys.argv[1])
except:
    print('Please specify model as command line arg (cnn or nvidia)')
    exit(0)
# create models
if sys.argv[1] == 'cnn':
    """ REGULAR CNN """
    output_dir = 'output/cnn/'
    os.makedirs(output_dir, exist_ok=True)
    model.add(Lambda(lambda x: x/ 127.5 - 1, input_shape = inputShape))
    model.add(Conv2D(32, kernel_size=5, padding="same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=3, padding="same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, kernel_size=3, padding="same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=10, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, name = 'output'))
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer = adam, loss = 'mse')
elif sys.argv[1] == 'nvidia':
    """ NVIDIA CNN """
    output_dir = 'output/cnn/'
    os.makedirs(output_dir, exist_ok=True)
    # normalization    
    # perform custom normalization before lambda layer in network
    model.add(Lambda(lambda x: x/ 127.5 - 1, input_shape = inputShape))
    model.add(Convolution2D(24, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv1'))
    model.add(ELU())    
    model.add(Convolution2D(36, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv2'))
    model.add(ELU())    
    model.add(Convolution2D(48, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv3'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, (3, 3), 
                            strides = (1,1), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv4'))
    model.add(ELU())              
    model.add(Convolution2D(64, (3, 3), 
                            strides= (1,1), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv5'))
    model.add(Flatten(name = 'flatten'))
    model.add(ELU())
    model.add(Dense(100, kernel_initializer = 'he_normal', name = 'fc1'))
    model.add(ELU())
    model.add(Dense(50, kernel_initializer = 'he_normal', name = 'fc2'))
    model.add(ELU())
    model.add(Dense(10, kernel_initializer = 'he_normal', name = 'fc3'))
    model.add(ELU())
    # do not put activation at the end because we want to exact output, not a class identifier
    model.add(Dense(3, name = 'output', kernel_initializer = 'he_normal'))
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer = adam, loss = 'mse')

else:
    print('Please specify model as command line arg (cnn or nvidia)')
    exit(0)
    
# TRAIN

# model = load_model('output/weights-500-46.47.hdf5')
filepath= output_dir + "weights-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
csv_logger = CSVLogger('{}training.log'.format(output_dir),append=False)

callbacks = [checkpoint, csv_logger]

history = model.fit(np.array(X), np.array(y), validation_split=0.3, epochs=1000, verbose = 1, batch_size=1, callbacks=callbacks, shuffle = False)

loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

## As loss always exists
epochs = range(1,len(history.history[loss_list[0]]) + 1)

## Loss
plt.figure(1)
for l in loss_list:
    plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
for l in val_loss_list:
    plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

plt.title('Mean Squared Error')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.savefig('{}loss.png'.format(output_dir))