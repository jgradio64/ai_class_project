from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# tf.config.experimental.set_memory_growth(gpus[0], True)

# Image size for the median image
IMAGE_SIZE = [640, 426]

# Relative path to training and testing folders.
train_Path = "/kaggle/input/median-data/median_data/train"
test_Path = '/kaggle/input/median-data/median_data/test'

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
x = Flatten()(resnet.output)
prediction = Dense(5, activation = 'softmax')(x)
# Create a model Object
model = Model(inputs = resnet.input, outputs = prediction)
model.compile (
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)
filepath = "/kaggle/working/weights_og.cp.ckpt"
checkpoint = ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1,save_best_only=True, mode='max')
callbacks_list = [checkpoint]
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
# As we have more than 2 so using categorical.. for 2 we might have used binary.
    class_mode = 'categorical'

)
test_set = train_datagen.flow_from_directory(
    test_Path,
    target_size = IMAGE_SIZE,
    batch_size = 64,
    class_mode = 'categorical'
)
# Fit the model.
#len(training_set)
# len(testing_set)//2
history = model.fit(
    training_set,
    validation_data = test_set,
    epochs = 25,
    callbacks=callbacks_list,
    verbose=1,
    steps_per_epoch = len(training_set) ,
    validation_steps = len(test_set)//2
)
# Plot the Loss
plt.plot(history.history['loss'], label = 'train_loss')
plt.plot(history.history['val_loss'], label ='val loss')
plt.legend()
plt.show()
plt.savefig('/kaggle/working/LossVal_loss')
# Plot the Accuracy
plt.plot(history.history['accuracy'], label = 'train accuracy')
plt.plot(history.history['val_accuracy'], label ='val accuracy')
plt.legend()
plt.show()
plt.savefig('/kaggle/working/valAccuracy')