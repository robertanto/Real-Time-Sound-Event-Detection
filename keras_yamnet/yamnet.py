# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Core model definition of YAMNet."""

import csv

from tensorflow.keras import Model, layers
import numpy as np

from . import params


def _batch_norm(name):
    def _bn_layer(layer_input):
        return layers.BatchNormalization(
            name=name,
            center=params.BATCHNORM_CENTER,
            scale=params.BATCHNORM_SCALE,
            epsilon=params.BATCHNORM_EPSILON)(layer_input)
    return _bn_layer


def _conv(name, kernel, stride, filters):
    def _conv_layer(layer_input):
        output = layers.Conv2D(name='{}/conv'.format(name),
                               filters=filters,
                               kernel_size=kernel,
                               strides=stride,
                               padding=params.CONV_PADDING,
                               use_bias=False,
                               activation=None)(layer_input)
        output = _batch_norm(name='{}/conv/bn'.format(name))(output)
        output = layers.ReLU(name='{}/relu'.format(name))(output)
        return output
    return _conv_layer


def _separable_conv(name, kernel, stride, filters):
    def _separable_conv_layer(layer_input):
        output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
                                        kernel_size=kernel,
                                        strides=stride,
                                        depth_multiplier=1,
                                        padding=params.CONV_PADDING,
                                        use_bias=False,
                                        activation=None)(layer_input)
        output = _batch_norm(name='{}/depthwise_conv/bn'.format(name))(output)
        output = layers.ReLU(
            name='{}/depthwise_conv/relu'.format(name))(output)
        output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
                               filters=filters,
                               kernel_size=(1, 1),
                               strides=1,
                               padding=params.CONV_PADDING,
                               use_bias=False,
                               activation=None)(output)
        output = _batch_norm(name='{}/pointwise_conv/bn'.format(name))(output)
        output = layers.ReLU(
            name='{}/pointwise_conv/relu'.format(name))(output)
        return output
    return _separable_conv_layer


_YAMNET_LAYER_DEFS = [
    # (layer_function, kernel, stride, num_filters)
    (_conv,          [3, 3], 2,   32),
    (_separable_conv, [3, 3], 1,   64),
    (_separable_conv, [3, 3], 2,  128),
    (_separable_conv, [3, 3], 1,  128),
    (_separable_conv, [3, 3], 2,  256),
    (_separable_conv, [3, 3], 1,  256),
    (_separable_conv, [3, 3], 2,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 2, 1024),
    (_separable_conv, [3, 3], 1, 1024)
]


def YAMNet(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling='avg',
        classes=521,
        classifier_activation="sigmoid"):
    """Define the core YAMNet mode in Keras."""

    if input_tensor is None:
        input_shape = input_shape if input_shape is not None else (
            params.PATCH_FRAMES, params.PATCH_BANDS)

        input_tensor = layers.Input(shape=input_shape)

    net = layers.Reshape(input_shape+(1,))(input_tensor)

    for (i, (layer_fun, kernel, stride, filters)) in enumerate(_YAMNET_LAYER_DEFS):
        net = layer_fun('layer{}'.format(i + 1), kernel, stride, filters)(net)

    if include_top:
        if weights is not None and classes != params.NUM_CLASSES:
          model_temp = Model(inputs=input_tensor, outputs=net)
    
          if weights is not None:
            model_temp.load_weights(weights)

          net = model_temp.output

        net = layers.GlobalAveragePooling2D()(net)
        logits = layers.Dense(units=classes, use_bias=True)(net)
        predictions = layers.Activation(
            name=params.EXAMPLE_PREDICTIONS_LAYER_NAME,
            activation=classifier_activation)(logits)
        
    else:
        if weights is not None:
          model_temp = Model(inputs=input_tensor, outputs=net)
    
          if weights is not None:
            model_temp.load_weights(weights)

          net = model_temp.output

        if pooling == 'avg':
            predictions = layers.GlobalAveragePooling2D()(net)
        elif pooling == 'max':
            predictions = layers.GlobalMaxPooling2D()(net)
        else:
            predictions = net
    
    model = Model(inputs=input_tensor, outputs=predictions)
    
    if weights is not None and classes == params.NUM_CLASSES:
      model.load_weights(weights)

    return model


def class_names(class_map_csv):
  """Read the class name definition file and return a list of strings."""
  with open(class_map_csv) as csv_file:
    reader = csv.reader(csv_file)
    next(reader)   # Skip header
    return np.array([display_name for (_, _, display_name) in reader])