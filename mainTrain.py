import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense


image_directory = 'image/'

dataset = []
label = []
Size = 64

no_tumor_images = os.listdir(image_directory + '0/')

for i, image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + '0/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((Size, Size))
        dataset.append(np.array(image))
        label.append(0)


gleoma_tumor_images = os.listdir(image_directory + '1/')

for i, image_name in enumerate(gleoma_tumor_images):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + '1/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((Size, Size))
        dataset.append(np.array(image))
        label.append(1)

meningeoma_tumor_images = os.listdir(image_directory + '2/')

for i, image_name in enumerate(meningeoma_tumor_images):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + '2/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((Size, Size))
        dataset.append(np.array(image))
        label.append(2)

putuitary_tumor_images = os.listdir(image_directory + '3/')

for i, image_name in enumerate(putuitary_tumor_images):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + '3/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((Size, Size))
        dataset.append(np.array(image))
        label.append(3)

# print(dataset)
# print(len(dataset))   # 4237
# print(label)
# print(len(label))     # 4237

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# print(x_train.shape)
# number, img-width, img-height, number-channels (RGB)
# 3389  ,    64    ,     64    ,       3

# print(y_train.shape)
# 3389

# print(x_test.shape)
# 848, 64, 64, 3
# print(y_test.shape)
# 848


x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)


#Mode Build

# model = Sequential()

# model.add(Conv2D(32, (3, 3), input_shape=(Size, Size, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))


# model.compile(loss='binary_crossentropy', optimizer='adam', 
#               metrics=['accuracy'])

# model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, 
#           validation_data=(x_test, y_test), shuffle=False)

# model.save('BrainTumor10Epochs.h5')


#########################3

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Hyperparameters
input_shape = (Size, Size, 3)  # Ensure Size is defined appropriately
batch_size = 32
epochs = 50
learning_rate = 0.001

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Ensure data normalization
x_train = x_train / 255.0
x_test = x_test / 255.0

# Model Definition
model = Sequential([
    Conv2D(32, (3, 3), input_shape=input_shape, kernel_initializer='he_uniform'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, (3, 3), kernel_initializer='he_uniform'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), kernel_initializer='he_uniform'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), kernel_initializer='he_uniform'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, kernel_initializer='he_uniform'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(1),
    Activation('sigmoid')
])

# Compile Model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Training with Data Augmentation
train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)

model.fit(
    train_generator,
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    verbose=1
)

# Save the Model
model.save('BrainTumor50Epochs.h5')

# 43.75%