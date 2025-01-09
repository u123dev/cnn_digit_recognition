#  Test Model
import os

import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt


# image size
img_height = 28
img_width = 28

# load model
model = tf.keras.models.load_model("cnn_digit01.keras")

model.summary()

def load_image(filepath):
    """ 1. Load an image from a filepath, convert to grayscale, invert, resize, save (as BW)
        2. convert to np """
    img = Image.open(filepath).convert('L')
    img = ImageOps.invert(img)
    img = img.resize((img_height, img_width))

    # save BW file
    newfilepath = filepath.replace("data\\", "data_bw\\bw-")
    img.save(newfilepath)
    print(f"{filepath} saved as {newfilepath}")

    img = np.array(img)
    img = img / 255.0
    img = img.reshape(-1, img_height, img_width, 1)
    return img

def predict_digit(test_img):
    """ load image & model predict"""
    img = load_image(test_img)
    prediction = model.predict(img)
    return np.argmax(prediction)


# for all images in directory
path = "data"
images = os.listdir(path)
for i, image in enumerate(images):
    # test_image = "data/003.jpg"
    test_image = os.path.join(path, image)
    predicted_digit = predict_digit(test_image)

    plt.subplot(len(images) // 3 + 1, 3, i + 1)
    img = Image.open(test_image).convert('L')
    img = img.resize((img_height, img_width))
    img = np.array(img)
    img = img / 255.0

    plt.title(predicted_digit, fontsize="xx-large", fontweight="bold", color="red")
    plt.imshow(img, cmap=plt.get_cmap('gray'))

    print(f"{test_image} : Digit:= {predicted_digit}")

plt.show()
