from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# Visualize first few images
for i in range(9):
    plt.subplot(330 + 1 + i)                            # define subplot
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))   # plot raw pixel data

plt.show()  # show the figure

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# init model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        # layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 15

# model.compile(loss="mse", optimizer="sgd", metrics=["accuracy"])

model.compile(
    # loss="mse",
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# Save the model in .keras  file
model.save('my_model.keras')
print(f'Model saved!')
# Recreate the exact same model from the file:
# model = keras.models.load_model("my_model.keras")

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


(_, _), (x_test, _) = keras.datasets.mnist.load_data()
x_test = x_test[:9]

# show 9 images from test dataset
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_test[i], cmap=plt.get_cmap('gray'))

plt.show()
# save to file
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # YYYY-MM-DD_HH-MM-SS
filename = f"test_{current_time}.png"
plt.savefig(filename)

# prepare for predicting
x_test = x_test.astype("float32") / 255
x_test = np.expand_dims(x_test, -1)
print(x_test.shape[0], "test samples")
print("x_test shape:", x_test.shape)

# perform predictions
predictions = model.predict(x_test)
image_classes = list(range(10))
print("Results of prediction: ")
for prediction in predictions:
    print(image_classes[np.argmax(prediction)])
