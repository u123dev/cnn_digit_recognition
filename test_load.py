#  Test Model
import glob
import os
from datetime import datetime

import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt


def load_image(filepath):
    """ 1. Load an image from a filepath, convert to grayscale, invert, resize, save (as BW)
        2. convert to np """
    img = Image.open(filepath).convert('L')
    img = ImageOps.invert(img)
    img = img.resize((img_height, img_width))

    # save BW file
    newfilepath = os.path.join("data_bw", "bw-" + image)
    if not os.path.isfile(newfilepath):
        img.save(newfilepath)
        print(f"{filepath} saved as {newfilepath}")

    img = np.array(img)
    img = img / 255.0
    img = img.reshape(-1, img_height, img_width, 1)
    return img

def predict_digit(test_img):
    """ load image & model predict"""
    img = load_image(test_img)
    prediction = model.predict(img, verbose=0)
    return np.argmax(prediction)


# path to test data
path = "data"

# image size
img_height = 28
img_width = 28

# models to load
model_files = glob.glob("*.keras")
if len(model_files) == 0:
    print("No models found")
    exit(0)
print(f"Found {len(model_files)} models: {model_files}")

for n, model_file in enumerate(model_files):
    print(f"Model {n + 1}: {model_file}")
    model = tf.keras.models.load_model(model_file)
    model.summary()

    # for all images in directory
    images = os.listdir(path)
    for i, image in enumerate(images):
        # test_image = "data/003.jpg"
        test_image = os.path.join(path, image)
        predicted_digit = predict_digit(test_image)

        plt.subplot(len(images) // 5 + 1, 5, i + 1)
        img = Image.open(test_image).convert('L')
        img = img.resize((img_height, img_width))
        img = np.array(img)
        img = img / 255.0

        plt.text(30,20, predicted_digit, fontsize="xx-large", fontweight="bold", color="red", )
        plt.imshow(img, cmap=plt.get_cmap('gray'))

        print(f"{test_image} : Digit:= {predicted_digit}")

    # save to file
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # YYYY-MM-DD_HH-MM-SS
    filename = f"test_{model_file}_{current_time}.png"
    plt.savefig(filename)

    # show
    plt.show()
