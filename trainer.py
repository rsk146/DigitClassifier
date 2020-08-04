import tensorflow
import keras
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self):
        #setup dataset
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        #setup network and add layers
        self.network = models.Sequential()
        self.network.add(layers.Dense(512, activation= 'relu', input_shape=(28*28,)))
        self.network.add(layers.Dense(10, activation='softmax'))
        self.network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        #reformat inputs
        train_images = train_images.astype('float32')/255
        train_images = train_images.reshape(60000, 28*28)
        train_labels = to_categorical(train_labels)

        for image in train_images:
            darken(image)
        #train_images = train_images.reshape(60000, 28, 28)
        #for i in range(10):
        #    draw(train_images[i])
        #train_images = train_images.reshape(60000, 28*28)

        #train model 5 epoch
        self.network.fit(train_images, train_labels, epochs= 5, batch_size=128)


#draw digit
def draw(digit):
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()


#possiblity: get rid of grayscale and make it binary and see if it performs better
def darken(image):
    for i in range(28*28):
        if image[i] >= .25:
            image[i] = 1
        else:
            image[i] = 0

