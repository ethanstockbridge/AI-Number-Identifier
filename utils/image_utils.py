import numpy as np
import tensorflow as tf
import os
import cv2
from PIL import Image

def fit_image(img):
    """Modifies an image by cropping the whitespace out and then resizing to 
    TRAINED_PIXELS size. 

    Args:
        img (numpy.ndarray): Input image

    Returns:
        numpy.ndarray: Resulting image
    """
    #set threshhold for image cropping
    points = np.argwhere(img>0.1)
    #change from row/col to x,y
    points = np.fliplr(points)
    x, y, w, h = cv2.boundingRect(points)
    crop_img = img[y:y+h, x:x+w]
    resized_img = cv2.resize(crop_img, (28,28))
    return resized_img


def train_model(x_train, y_train):
    """Train the model from custom x and y dataset of numbers

    Args:
        x_train (numpy.ndarray): Images of numbers drawn
        y_train (numpy.ndarray): Resulting numeric number

    Returns:
        Tensorflow model: Trained model
    """
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])

    predictions = model(x_train[:1]).numpy()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    loss_fn(y_train[:1], predictions).numpy()

    model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10)

    return model


def get_data(path):
    """Grabs the data from a path where the folder represents the 
    numeric answer of the images inside

    Args:
        path (str): Input path where folders "0" through "9" live

    Returns:
        (numpy.ndarray, numpy.ndarray): training data
    """
    folders = os.listdir(path)
    x_train = []
    y_train = np.array([])
    for folder in folders:
        for file in os.listdir(os.path.join(path, folder)):
            y_train=np.append(y_train,int(folder))
            img = Image.open(os.path.join(path,folder,file))
            img=np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img=img/255
            img = abs(img-1)
            x_train.append(img)
    x_train=np.array(x_train)

    for i in range(len(x_train)):
        x_train[i] = fit_image(x_train[i])
    
    return x_train, y_train