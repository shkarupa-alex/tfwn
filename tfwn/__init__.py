from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import backend as K

_DEFAULT_WEIGHT_NAME = 'kernel'


class WeightNorm(tf.keras.layers.Wrapper):
    """ Applies weight normalization to a layer. Weight normalization is a reparameterization that decouples the
    magnitude of a weight tensor from its direction. This speeds up convergence by improving the conditioning of the
    optimization problem.

    Reference: https://arxiv.org/abs/1602.07868
    Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks
    Tim Salimans, Diederik P. Kingma (2016)

    Arguments:
      layer: The `Layer` instance to be wrapped.
      weight_name: name of weights `Tensor` in wrapped layer to be normalized
    """

    def __init__(self, layer, weight_name=_DEFAULT_WEIGHT_NAME, **kwargs):
        self.weight_name = weight_name

        if not hasattr(layer, '_weight_norm'):
            layer._weight_norm = {weight_name}
        elif weight_name not in layer._weight_norm:
            layer._weight_norm.add(weight_name)
        else:
            raise ValueError(
                'Weight normalization already applied to parameter {} in layer {}'.format(weight_name, layer))

        self.g_name = '{}_g'.format(weight_name)
        self.v_name = '{}_v'.format(weight_name)

        super(WeightNorm, self).__init__(layer, **kwargs)
        self.supports_masking = True
        self._track_checkpointable(layer, name='layer')

    def build(self, input_shape=None):
        self.input_spec = tf.keras.layers.InputSpec(shape=tf.TensorShape(input_shape).as_list())

        if not self.layer.built:
            self.layer.build(input_shape)

        if hasattr(self.layer, self.g_name) or hasattr(self.layer, self.v_name):
            raise ValueError(
                'Weight normalization already applied to {} in layer {}'.format(self.weight_name, self.layer))

        if not hasattr(self.layer, self.weight_name):
            raise ValueError('Parameter {} not found in layer {}'.format(self.weight_name, self.layer))

        v = getattr(self.layer, self.weight_name)
        weight_depth = v.shape[-1]
        norm_axes = list(range(v.shape.ndims - 1))

        def g_init(v_init):
            c = K.shape(v_init)[-1]
            v_init = K.reshape(v_init, [-1, c])

            return K.linalg_ops.norm(v_init, ord=2, axis=0)

        self.layer.built = False
        g = self.layer.add_variable(
            name=self.g_name,
            shape=(weight_depth,),
            initializer=lambda *args, **kwargs: g_init(v.initialized_value()),
            dtype=v.dtype,
            trainable=True,
            aggregation=tf.VariableAggregation.MEAN
        )
        self.layer.built = True

        w = K.nn.l2_normalize(v, axis=norm_axes) * g

        setattr(self.layer, self.g_name, g)
        setattr(self.layer, self.v_name, v)
        setattr(self.layer, self.weight_name, w)

        super(WeightNorm, self).build()

    def call(self, inputs, **kwargs):
        return self.layer.call(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def get_config(self):
        config = super(WeightNorm, self).get_config()

        if _DEFAULT_WEIGHT_NAME != self.weight_name:
            config['weight_name'] = self.weight_name

        return config
