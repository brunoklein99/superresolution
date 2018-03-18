import cv2
import numpy as np
import settings
import keras

from data_load import load_data
from model import get_model
from math import ceil


def get_generator(files):
    for file in files:
        y = cv2.imread(file)
        y = cv2.resize(y, (settings.TARGET_SIZE, settings.TARGET_SIZE))
        x = cv2.resize(y, (settings.DOWNSAMPLE_SIZE, settings.DOWNSAMPLE_SIZE))
        x = cv2.resize(x, (settings.TARGET_SIZE, settings.TARGET_SIZE))

        # cv2.imshow('x', x)
        # cv2.imshow('y', y)
        # cv2.waitKey()

        yield x, y


def batch_generator(gen, batch_size, length):
    batch_x = np.zeros((batch_size, settings.TARGET_SIZE, settings.TARGET_SIZE, 3))
    batch_y = np.zeros((batch_size, settings.TARGET_SIZE, settings.TARGET_SIZE, 3))
    while True:
        count = length
        while count > 0:
            bsize = batch_size if count > batch_size else count
            for i in range(bsize):
                x, y = next(gen)
                batch_x[i] = x
                batch_y[i] = y
            yield batch_x[:bsize], batch_y[:bsize]
            count -= batch_size


def train_model(train, valid):
    model = get_model((settings.TARGET_SIZE, settings.TARGET_SIZE, 3))

    model.compile(optimizer='sgd', loss=keras.metrics.mean_squared_error)

    train_gen = batch_generator(get_generator(train), settings.BATCH_SIZE, len(train))
    valid_gen = batch_generator(get_generator(valid), settings.BATCH_SIZE, len(valid))

    steps_train = int(ceil(len(train) / settings.BATCH_SIZE))
    steps_valid = int(ceil(len(valid) / settings.BATCH_SIZE))

    model.fit_generator(generator=train_gen,
                        steps_per_epoch=steps_train,
                        epochs=settings.EPOCHS,
                        validation_data=valid_gen,
                        validation_steps=steps_valid)


if __name__ == "__main__":
    train, valid, test = load_data()

    train_model(train, valid)
