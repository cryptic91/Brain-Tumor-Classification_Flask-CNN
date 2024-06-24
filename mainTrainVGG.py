import cv2
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Hyperparameters
Size = 64
batch_size = 32
epochs = 100
learning_rate = 0.001
input_shape = (Size, Size, 3)

# Load images
image_directory = 'image/'
dataset = []
label = []

def load_images_from_folder(folder, label_value):
    images = os.listdir(folder)
    for image_name in images:
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(folder + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((Size, Size))
            dataset.append(np.array(image))
            label.append(label_value)

load_images_from_folder(image_directory + '0/', 0)
load_images_from_folder(image_directory + '1/', 1)
load_images_from_folder(image_directory + '2/', 2)
load_images_from_folder(image_directory + '3/', 3)

# Convert lists to arrays
dataset = np.array(dataset)
label = np.array(label)

# Normalize images
dataset = dataset.astype('float32') / 255.0

# Convert labels to categorical (one-hot encoding)
label = to_categorical(label, num_classes=4)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load the VGG16 model, excluding the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Model Definition
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, kernel_initializer='he_uniform'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(4),  # Number of classes
    Activation('softmax')
])

# Compile Model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Training with Data Augmentation
train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)

model.fit(
    train_generator,
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Save the Model
model.save('BrainTumorVGG16.h5')

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 86.79%