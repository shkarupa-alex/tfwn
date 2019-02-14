from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import testing_utils
from .. import WeightNorm


class WeightNormWrapperTest(tf.test.TestCase):
    def testDoubleWrap(self):
        dense_layer = tf.layers.Dense(7)

        WeightNorm(dense_layer)
        with self.assertRaisesRegexp(ValueError, 'Weight normalization already applied'):
            WeightNorm(dense_layer)

    def testNoKernel(self):
        with self.assertRaisesRegexp(ValueError, 'Parameter .* not found in layer'):
            WeightNorm(tf.keras.layers.MaxPooling2D(2, 2)).build((2, 2))

    def testVarsAndShapes(self):
        source_input = tf.random_normal([3, 5])

        dense_layer = tf.layers.Dense(7)
        dense_wrapper = WeightNorm(dense_layer)
        weighted_output = dense_wrapper(source_input)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(weighted_output)

        self.assertEqual(dense_layer.kernel_v.shape.as_list(), dense_layer.kernel.shape.as_list())
        self.assertEqual(dense_layer.kernel_g.shape, [7])

    def testDenseDecomposition(self):
        source_input = tf.random_normal([3, 5])

        original_layer = tf.layers.Dense(7)
        original_output = original_layer(source_input)

        wrapper_layer = WeightNorm(original_layer)
        weighted_output = wrapper_layer(source_input)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            original_result, weighted_result = sess.run([original_output, weighted_output])

        self.assertAllClose(original_result, weighted_result)
        self.assertNotEqual(original_result.tolist(), weighted_result.tolist())

    def testConvDecomposition(self):
        source_input = tf.random_normal([3, 5, 7, 9])

        original_layer = tf.layers.Conv2D(2, 4)
        original_output = original_layer(source_input)

        wrapper_layer = WeightNorm(original_layer)
        weighted_output = wrapper_layer(source_input)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            original_result, weighted_result = sess.run([original_output, weighted_output])

        self.assertAllClose(original_result, weighted_result)
        self.assertNotEqual(original_result.tolist(), weighted_result.tolist())

    def testLayer(self):
        with self.cached_session(use_gpu=True):
            with tf.keras.utils.custom_object_scope({'WeightNorm': WeightNorm}):
                testing_utils.layer_test(
                    WeightNorm,
                    kwargs={
                        'layer': tf.layers.Dense(1),
                    },
                    input_shape=(3, 7)
                )


class WeightNormModelTest(tf.test.TestCase):
    def setUp(self):
        self.num_classes = 10

        (train_x, train_y), _ = tf.keras.datasets.cifar10.load_data()
        self.train_x = train_x[:10].astype(np.float32) / 255
        self.train_y = tf.keras.utils.to_categorical(train_y[:10].astype(np.int), self.num_classes)

    def testFunctionStyle(self):
        model = tf.keras.Sequential()
        model.add(WeightNorm(tf.keras.layers.Conv2D(6, 5, activation='relu')))
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        model.add(tf.keras.layers.Flatten())
        model.add(WeightNorm(tf.keras.layers.Dense(self.num_classes, activation='softmax')))
        model.compile(optimizer=tf.train.AdamOptimizer(), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(self.train_x, self.train_y, epochs=1)

    def testClassStyle(self):
        num_classes = self.num_classes

        class TestModel(tf.keras.Model):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv = WeightNorm(tf.keras.layers.Conv2D(6, 5, activation='relu'))
                self.maxpool = tf.keras.layers.MaxPooling2D(2, 2)
                self.flatten = tf.keras.layers.Flatten()
                self.dense = WeightNorm(tf.keras.layers.Dense(num_classes, activation='softmax'))

            def call(self, input, training=None, mask=None):
                x = self.conv(input)
                x = self.maxpool(x)
                x = self.flatten(x)
                x = self.dense(x)

                return x

        model = TestModel()
        model.compile(optimizer=tf.train.AdamOptimizer(), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(self.train_x, self.train_y, epochs=1)


if __name__ == "__main__":
    tf.test.main()
