from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential


from tensorflow.keras.optimizers import SGD

import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# tf.config.experimental.set_memory_growth(gpus[0], True)

# Image size for the median image
IMAGE_SIZE = [74, 200]

# Relative path to training and testing folders.
train_Path = "/kaggle/input/min-data/min_data/train"
test_Path = "/kaggle/input/min-data/min_data/test"

resnet = ResNet50(
    input_shape = IMAGE_SIZE + [3], # Making the image into 3 Channel, so concating 3.
    weights = 'imagenet', # Default weights.
    include_top = False,   # 
    classes = 5
)

# resnet.summary()

for layer in resnet.layers:
    layer.trainable = False

folders = glob(train_Path + '/*')
folders

vehicle_label = ['car', 'minivan', 'suv', 'truck', 'van']

# Set the flatten layer.
x = Flatten() (resnet.output)

# Changed to relu, added more dense layers
prediction = Dense(100, activation = 'relu')(x)
prediction = Dense(100, activation='relu')(x)
prediction = Dense(50, activation = 'relu')(x)
prediction = Dense(10, activation='relu')(x)
prediction = Dense(5, activation='softmax')(x)

# Create a model Object
model = Model(inputs = resnet.input, outputs = prediction)

print(model.summary())

model.compile (
    loss = 'categorical_crossentropy',
    optimizer = SGD(lr=0.1, momentum=0.9, weight_decay=0.01),
    metrics = ['accuracy']
)

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(
    rescale = 1./255
)

training_set = train_datagen.flow_from_directory(
    train_Path,
    target_size = IMAGE_SIZE,
    batch_size = 64,
    class_mode = 'categorical' # As we have more than 2 so using categorical.. for 2 we might have used binary.
)

test_set = train_datagen.flow_from_directory(
    test_Path,
    target_size = IMAGE_SIZE,
    batch_size = 64,
    class_mode = 'categorical'
)

# Fit the model.
history = model.fit(
    training_set,
    validation_data = test_set,
    epochs = 100,
    steps_per_epoch = len(training_set),
    validation_steps = len(test_set)
)

# Plot the Loss
plt.plot(history.history['loss'], label = 'train_loss')
plt.plot(history.history['val_loss'], label ='val loss')
plt.legend()
plt.show()
# plt.savefig('LossVal_loss')

# Plot the Accuracy
plt.plot(history.history['accuracy'], label = 'train accuracy')
plt.plot(history.history['val_accuracy'], label ='val accuracy')
plt.legend()
plt.show()
# plt.savefig('valAccuracy')