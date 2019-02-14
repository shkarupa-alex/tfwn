from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
from matplotlib import pyplot
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tfwn import WeightNorm


class Cifar10(keras.Model):
    NUM_CLASSES = 10

    def __init__(self, weight_norm=False):
        inp = keras.layers.Input(shape=(32, 32, 3))

        # ZCA whitening should be applied to input
        noise = keras.layers.GaussianNoise(0.15)

        conv1 = keras.layers.Conv2D(96, 3, strides=1, activation=K.nn.leaky_relu, padding='same')
        conv2 = keras.layers.Conv2D(96, 3, strides=1, activation=K.nn.leaky_relu, padding='same')
        conv3 = keras.layers.Conv2D(96, 3, strides=1, activation=K.nn.leaky_relu, padding='same')

        pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        drop1 = keras.layers.Dropout(0.5)

        conv4 = keras.layers.Conv2D(192, 3, strides=1, activation=K.nn.leaky_relu, padding='same')
        conv5 = keras.layers.Conv2D(192, 3, strides=1, activation=K.nn.leaky_relu, padding='same')
        conv6 = keras.layers.Conv2D(192, 3, strides=1, activation=K.nn.leaky_relu, padding='same')

        pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        drop2 = keras.layers.Dropout(0.5)

        conv7 = keras.layers.Conv2D(192, 3, strides=2, activation=K.nn.leaky_relu, padding='same')
        conv8 = keras.layers.Conv2D(192, 1, strides=1, activation=K.nn.leaky_relu, padding='same')
        conv9 = keras.layers.Conv2D(192, 1, strides=1, activation=K.nn.leaky_relu, padding='same')

        pool3 = keras.layers.GlobalAveragePooling2D()

        dense = keras.layers.Dense(self.NUM_CLASSES, activation='softmax')

        if weight_norm:
            conv1 = WeightNorm(conv1)
            conv2 = WeightNorm(conv2)
            conv3 = WeightNorm(conv3)
            conv4 = WeightNorm(conv4)
            conv5 = WeightNorm(conv5)
            conv6 = WeightNorm(conv6)
            conv7 = WeightNorm(conv7)
            conv8 = WeightNorm(conv8)
            conv9 = WeightNorm(conv9)
            dense = WeightNorm(dense)

        out = noise(inp)

        out = conv1(out)
        out = conv2(out)
        out = conv3(out)

        out = pool1(out)
        out = drop1(out)

        out = conv4(out)
        out = conv5(out)
        out = conv6(out)

        out = pool2(out)
        out = drop2(out)

        out = conv7(out)
        out = conv8(out)
        out = conv9(out)

        out = pool3(out)
        out = dense(out)

        super(Cifar10, self).__init__(inputs=inp, outputs=out)


def run_cifar10(weight_norm, batch_size, epoch_count, learning_rate):
    (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
    train_x, test_x = train_x.astype(np.float32) / 255, test_x.astype(np.float32) / 255
    train_y = keras.utils.to_categorical(train_y.astype(np.int), 10)
    test_y = keras.utils.to_categorical(test_y.astype(np.int), 10)

    datagen = ImageDataGenerator(featurewise_center=True, zca_whitening=True)
    datagen.fit(train_x)

    model = Cifar10(weight_norm)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate, beta_1=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    lr_decay = keras.callbacks.LearningRateScheduler(
        lambda epoch: learning_rate * np.minimum(2. - epoch * 2 / epoch_count, 1.))

    beta1_decay = keras.callbacks.LambdaCallback(
        on_epoch_begin=lambda epoch, logs:
        K.set_value(model.optimizer.beta_1, 0.5) if epoch > epoch_count // 2 else None,
    )

    return model.fit_generator(
        datagen.flow(train_x, train_y, batch_size=batch_size),
        steps_per_epoch=len(train_x) / batch_size,
        epochs=epoch_count,
        validation_data=datagen.flow(test_x, test_y, batch_size=batch_size),
        callbacks=[lr_decay, beta1_decay],
    ).history


def draw_metrics(regular_metrics, weighted_metrics, batch_size, epoch_count):
    interval = np.linspace(0, epoch_count, epoch_count)

    pyplot.plot(interval, regular_metrics['acc'], color='blue', dashes=[6, 2], label='Regular train')
    pyplot.plot(interval, weighted_metrics['acc'], color='green', dashes=[6, 2], label='Weighted train')
    pyplot.plot(interval, regular_metrics['val_acc'], color='blue', label='Regular valid')
    pyplot.plot(interval, weighted_metrics['val_acc'], color='green', label='Weighted valid')
    pyplot.title('Weight normalization on CIFAR10. Batch size: {}'.format(batch_size))
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.legend()
    pyplot.savefig('cifar10_accuracy_{}.png'.format(batch_size))
    pyplot.close()

    pyplot.plot(interval, regular_metrics['loss'], color='blue', dashes=[6, 2], label='Regular train')
    pyplot.plot(interval, weighted_metrics['loss'], color='green', dashes=[6, 2], label='Weighted train')
    pyplot.title('Weight normalization on CIFAR10. Batch size: {}'.format(batch_size))
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.legend()
    pyplot.savefig('cifar10_loss_{}.png'.format(batch_size))
    pyplot.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test weight normalization on CIFAR10')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Size of batch')
    parser.add_argument(
        '--epoch_count',
        type=int,
        default=200,
        help='Number of epochs')
    parser.add_argument(
        '--initial_lr',
        type=float,
        default=0.001,
        help='Initial learning rate')
    parser.add_argument(
        '--random_seed',
        type=int,
        default=1,
        help='Random seed')

    argv, _ = parser.parse_known_args()

    np.random.seed(argv.random_seed)
    K.random_ops.random_seed.set_random_seed(argv.random_seed)

    regular_metrics = run_cifar10(
        weight_norm=False,
        batch_size=argv.batch_size,
        epoch_count=argv.epoch_count,
        learning_rate=argv.initial_lr,
    )
    weighted_metrics = run_cifar10(
        weight_norm=True,
        batch_size=argv.batch_size,
        epoch_count=argv.epoch_count,
        learning_rate=argv.initial_lr,
    )
    draw_metrics(
        regular_metrics=regular_metrics,
        weighted_metrics=weighted_metrics,
        batch_size=argv.batch_size,
        epoch_count=argv.epoch_count,
    )
