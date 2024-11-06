############################################################
###################### PROJECT 1 ###########################
####### Image Classification Using Pre-trained Models ######
####### AHMED REFAAY #######################################
####### 5th of Nov., 2024 ##################################
############################################################

################ IMPORT LIBRARIES ##########################
import numpy as np
import random
random.seed(10)
import sys
import datetime
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2 as cv
import keras
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.python.platform.build_info as build
from tensorflow.keras.utils import to_categorical
from keras.preprocessing import image

################ GLOBAL VARS ############################
DIMX = 96
DIMY = 96
NUM_TRANSF_EPOCHS=20 # Number of training epochs for transferleaning
NUM_FINETUNE_EPOCHS=10 # Number of training epochs for finetuning steps
HIDDEN=512 # Number of nodes in hidden layer
DROPOUT=0.25 # optional dropout rate
BATCH_SIZE = 32 # batch size of training data
NUM_CLASSES = 10
LEARNING_RATE = 0.0025
TIME_NOW = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Set the log directory to store the logs for tensorboard
log_dir = os.path.join("logs", "fit", TIME_NOW)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

################ PRINT VERSIONS #########################
#print("Python version: ", sys.version)
#print("Tensorflow version:",tf.__version__)
#print("Keras version:",keras.__version__)
#print(build.build_info)

def _augment_images(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    # Apply random horizontal flip
    image = tf.image.random_flip_left_right(image)
    # Apply random brightness adjustment
    image = tf.image.random_brightness(image, max_delta=0.1)
    # Similarly, apply other augmentations
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    # Ensure pixel values remain between 0 and 1
    image = tf.clip_by_value(image, 0, 1)
    return image, label

############## LOAD CIFAR10 #############################
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)
"""
print("Before augmentation")
print(x_train.shape)
print(x_train[0][0][0])
print(y_train.shape)
print(y_train[0])
"""
#cv.imshow("Train before augmentation", x_train[0])
#k = cv.waitKey(0) # Wait for a keystroke in the window

x_train = tf.image.resize(x_train, [DIMX, DIMY], method='bicubic')
x_test  = tf.image.resize(x_test , [DIMX, DIMY], method='bicubic')
y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test  = to_categorical(y_test, num_classes=NUM_CLASSES)
#x_train = x_train.numpy().astype(np.uint8)
#x_test = x_test.numpy().astype(np.uint8)
#y_train = y_train.astype(np.uint8)
#y_test = y_test.astype(np.uint8)
"""
print("After resize")
print(x_train.shape)
print(x_train[0][0][0])
print(y_train.shape)
print(y_train[0])
"""
x_train = x_train.numpy().astype("float16") / 255.0
x_test = x_test.numpy().astype("float16") / 255.0
"""
print("After normalization")
print(x_train.shape)
print(x_train[0][0][0])
"""
#cv.imshow("Train after resize", x_train[0])
#k = cv.waitKey(0) # Wait for a keystroke in the window

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
"""
print("Datasets info")
print(len(train_dataset))
print(train_dataset)
"""
validation_dataset = train_dataset.take(10000)
train_dataset = train_dataset.skip(10000)
"""
print("Datasets info after take validation")
print(len(train_dataset))
print(train_dataset)
"""
# Sample a subset for augmentation
augmented = train_dataset.take(20000)
# Apply augmentation
augmented = augmented.map(_augment_images)
rotation = tf.keras.layers.RandomRotation(0.15, dtype = "float16")
augmented = augmented.map(lambda x, y: (rotation(x), y))
train_dataset = train_dataset.concatenate(augmented)
train_dataset = train_dataset.shuffle(buffer_size=train_dataset.cardinality())
"""
print("Datasets info after take augmentation")
print(len(train_dataset))
print(train_dataset)
for images, labels in train_dataset.take(1):  # only take first element of dataset
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    print(numpy_images.shape)
    print(numpy_images[0][0])
    print(numpy_labels)
    numpy_images = numpy_images * 255
    numpy_images = numpy_images.astype(np.uint8)
    cv.imshow("Augmented image", numpy_images)
    k = cv.waitKey(0) # Wait for a keystroke in the window

print("Total items in train dataset:", train_dataset.cardinality())
print("Total items in validation dataset:",
        validation_dataset.cardinality())
print("Total items in test dataset:", test_dataset.cardinality())
"""
train_dataset = train_dataset.shuffle(
    buffer_size=train_dataset.cardinality(), reshuffle_each_iteration=True)
train_dataset = train_dataset.prefetch(
    buffer_size=tf.data.AUTOTUNE)

# Batch the datasets
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(BATCH_SIZE)  # cache()??? prefetch()???
test_dataset = test_dataset.batch(BATCH_SIZE)

print("Data loaded")

############## LOAD MIBILENETV2 #########################
# Load the base model with imagenet weights without the top layer
base_model = tf.keras.applications.MobileNetV2(include_top=False, 
    weights='imagenet', classes=NUM_CLASSES, input_shape=(DIMX, DIMY, 3))
base_model.trainable = False  # Freeze the base model initially
# base_model.summary(show_trainable=True, expand_nested=True)
print("Model loaded")

############ ADD INPUT, FULLY-CONNECTED, & OUTPUT LAYERS ######
# Define input
input = tf.keras.Input(shape=(DIMX, DIMY, 3), name="input")
# Add new layers on top of the model
x = base_model(input, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# optional for your own experiments
x = tf.keras.layers.Dropout(DROPOUT)(x) 
x = tf.keras.layers.Dense(HIDDEN, activation='relu')(x)
predictions = tf.keras.layers.Dense(NUM_CLASSES, 
    activation='softmax', name="output")(x)
model = tf.keras.models.Model(inputs=input, 
    outputs=predictions, name="ft_net")
#model.summary(show_trainable=True, expand_nested=True)
print("Layers added")

############ TRAIN NEW ADDED LAYERS #########################
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
      loss='categorical_crossentropy', metrics=['accuracy'])
"""
print("Training starts")
fit_history_t = model.fit(train_dataset, epochs=NUM_TRANSF_EPOCHS, validation_data=validation_dataset, callbacks=[tensorboard_callback], shuffle=True, batch_size=BATCH_SIZE)
print("Training ends")
loss, accuracy = model.evaluate(x=x_test, y=y_test, batch_size=BATCH_SIZE)
print("TRANSFER LEARN: test accuracy: ", accuracy, "test loss: ", loss)
MODEL_FILE="./models/CNN_Net_"+TIME_NOW+"_"+str(accuracy)+".keras"
model.save(MODEL_FILE)
print("Model saved")
"""

############ FINE TUNING ####################################
"""
def unfreeze_base_layers(model: tf.keras.Model, 
    layers: int, learning_rate: float) -> tf.keras.Model:
    base_model = model.layers[1]
    # Unfreeze the last layers
    for layer in base_model.layers[-min(len(base_model.layers), layers):]:
        layer.trainable = True
    # Compile the model again to apply the changes
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
      loss='categorical_crossentropy', metrics=['accuracy'])
    #model.summary(show_trainable=True, expand_nested=True)
    return model

LEARNING_RATE = 0.0001
unfreeze=92
model = unfreeze_base_layers(model, layers=unfreeze, 
    learning_rate=LEARNING_RATE)
fit_history_ft1 = model.fit(train_dataset, epochs=NUM_FINETUNE_EPOCHS,
    validation_data=validation_dataset, 
    callbacks=[tensorboard_callback])
loss, accuracy = model.evaluate(x=x_test, y=y_test, batch_size=BATCH_SIZE)

unfreeze+=9
model = unfreeze_base_layers(model, layers=unfreeze,
        learning_rate=LEARNING_RATE/2.0)
fit_history_ft2 = model.fit(train_dataset, epochs=NUM_FINETUNE_EPOCHS,
        validation_data=validation_dataset, 
        callbacks=[tensorboard_callback])
loss, accuracy = model.evaluate(x=x_test, y=y_test, batch_size=BATCH_SIZE)

unfreeze += 9
model = unfreeze_base_layers(model, layers=unfreeze, 
        learning_rate=LEARNING_RATE/4.0)
fit_history_ft3 = model.fit(train_dataset, epochs=NUM_FINETUNE_EPOCHS,
        validation_data=validation_dataset,
        callbacks=[tensorboard_callback])
loss, accuracy = model.evaluate(x=x_test, y=y_test, batch_size=BATCH_SIZE)
MODEL_FILE="./models/CNN_Net_Finetuned_"+TIME_NOW+"_"+str(accuracy)+".keras"
model.save(MODEL_FILE)
"""
################ LOAD & PREDICT ############################

model.load_weights('models/CNN_Net_Finetuned_20241105-134056_0.9193000197410583.keras')
print("Saved model loaded")
"""
for images, labels in test_dataset.take(1):  # only take first element of dataset
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    print(numpy_images.shape)
    print(numpy_images[63].shape)
    print(numpy_images[63][0][0])
    print(numpy_labels[0])
    max_predictions = tf.argmax(numpy_labels, axis=1)
    print(max_predictions)
    numpy_image = numpy_images[63] * 255
    numpy_image = numpy_image.astype(np.uint8)
    cv.imshow("Test image", numpy_image)
    k = cv.waitKey(0) # Wait for a keystroke in the window

pred_dataset = test_dataset.take(1).cache()
# Make Prediction with test_dataset
predictions = model.predict(pred_dataset)
print(predictions[0])
max_predictions = tf.argmax(predictions, axis=1)
print(max_predictions)
"""
# Load external image & predict
img = cv.imread('Images/deer.jpg')
cv.imshow("External test original image", img)
k = cv.waitKey(0) # Wait for a keystroke in the window
img = cv.resize(img, (DIMX, DIMY), interpolation= cv.INTER_LINEAR)
cv.imshow("External test resized image", img)
k = cv.waitKey(0) # Wait for a keystroke in the window

img = image.load_img('Images/deer.jpg', target_size=(DIMX, DIMY))
xx = image.img_to_array(img)
xx = xx / 255.0
xx = np.expand_dims(xx, axis=0)
images = np.vstack([xx])
classes = model.predict(images, batch_size=10)
# print(classes)
max_predictions = tf.argmax(classes, axis=1)
# print(max_predictions[0].numpy())
cifar_class_names = ["plane", "car", "bird", "cat",
                         "deer", "dog", "frog", "horse", "ship", "truck"]
print('This is a ' + cifar_class_names[max_predictions[0].numpy()])
