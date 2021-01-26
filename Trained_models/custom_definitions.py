from PrognosAIs.Model.Architectures.UNet import Unet
from PrognosAIs.Model.Architectures.ResNet import ResNet
from tensorflow.keras.layers import SpatialDropout3D, Concatenate, BatchNormalization, GaussianDropout,GaussianNoise, Conv3D, MaxPooling3D, GlobalAveragePooling3D, Flatten, Dense, Activation, Conv2D, ReLU, Add, Conv3DTranspose, GlobalMaxPool3D
from tensorflow.keras import Model, Input
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import losses_utils
import copy
import tensorflow
import tensorflow_addons as tfa
from tensorflow.python.keras.utils import metrics_utils
import numpy as np
from tensorflow.python.keras import backend as K
import logging

class PSNET_3D(Unet):
    dims = 3

    def make_inputs(
        self, input_shapes: dict, input_dtype: str, squeeze_inputs: bool = True
    ):
        inputs = {}
        for i_input_name, i_input_shape in input_shapes.items():
            inputs[i_input_name] = Input(shape=i_input_shape, name=i_input_name, dtype="float16")

        if squeeze_inputs and len(inputs) == 1:
            inputs = list(inputs.values())[0]

        return inputs

    def get_init_filter_size(self):
        if self.model_config is not None and "filter_size" in self.model_config:
            return self.model_config["filter_size"]
        else:
            return 7

    def get_init_stride_size(self):
        if self.model_config is not None and "stride_size" in self.model_config:
            return self.model_config["stride_size"]
        else:
            return 3

    def make_norm_layer(self, layer):
        if self.model_config is not None and "norm_layer" in self.model_config:
            norm_setting = self.model_config["norm_layer"]
            if norm_setting == "batch":
                return BatchNormalization()(layer)
            elif norm_setting == "batch_sync":
                return tf.keras.layers.experimental.SyncBatchNormalization()(layer)
            elif norm_setting == "instance":
                return tfa.layers.InstanceNormalization()(layer)
            else:
                return layer
        else:
            return layer

    def get_stride_activations(self):
        if self.model_config is not None and "stride_activation" in self.model_config:
            return self.model_config["stride_activation"]
        else:
            return "linear"

    def get_output_type(self):
        if self.model_config is not None and "output_type" in self.model_config:
            return self.model_config["output_type"]
        else:
            return "softmax"

    def make_global_pool_layer(self, layer):
        if self.model_config is not None and "global_pool" in self.model_config:
            layer_setting = self.model_config["global_pool"]
            if layer_setting == "average":
                layer = GlobalAveragePooling3D()(layer)
            elif layer_setting == "max":
                layer = GlobalMaxPool3D()(layer)
            else:
                layer = GlobalAveragePooling3D()(layer)
        else:
            layer = GlobalAveragePooling3D()(layer)

        return layer

    def get_gap_after_dropout(self):
        if self.model_config is not None and "gap_after_dropout" in self.model_config:
            return self.model_config["gap_after_dropout"]
        else:
            return False

    def get_final_dense_units(self):
        if self.model_config is not None and "dense_units" in self.model_config:
            return self.model_config["dense_units"]
        else:
            return 512

    def get_kernel_regularizer(self):
        if self.model_config is not None and "l2_norm" in self.model_config:
            return tf.keras.regularizers.l2(l=self.model_config["l2_norm"])
        else:
            return None

    def get_use_additional_convs(self):
        if self.model_config is not None and "convs" in self.model_config:
            return self.model_config["convs"]
        else:
            return False

    def get_use_upsample_genetic_features(self):
        if self.model_config is not None and "upsample_features" in self.model_config:
            return self.model_config["upsample_features"]
        else:
            return True

    def get_final_conv_layers(self):
        if self.model_config is not None and "final_conv_layers" in self.model_config:
            return self.model_config["final_conv_layers"]
        else:
            return 256



    def create_model(self):
        self.init_dimensionality(self.dims)
        self.inputs = self.make_inputs(self.input_shapes, self.input_data_type)

        self.N_filters = self.get_number_of_filters()
        self.depth = self.get_depth()
        filter_size = self.get_init_filter_size()
        stride_size = self.get_init_stride_size()
        activations = self.get_stride_activations()
        output_type = self.get_output_type()
        gap_after_dropout = self.get_gap_after_dropout()
        final_dense_unit = self.get_final_dense_units()
        kernel_regularizer = self.get_kernel_regularizer()
        additional_convs = self.get_use_additional_convs()
        upsample_features = self.get_use_upsample_genetic_features()
        final_conv_layers = self.get_final_conv_layers()

        head = self.inputs
        skip_layers = []
        gap_layers = []

        for i_depth in range(self.depth - 1):
            head = self.get_conv_block(head, self.N_filters * (2 ** i_depth), activation=activations, kernel_regularizer=kernel_regularizer)

            if i_depth == 0:
                if not gap_after_dropout:
                    gap_layers.append(self.make_global_pool_layer(head))
                # head = self.make_norm_layer(head)
                head = self.make_dropout_layer(head)
                if gap_after_dropout:
                    gap_layers.append(self.make_global_pool_layer(head))
                skip_layers.append(head)
                head = Conv3D(self.N_filters * (2 ** i_depth), filter_size, strides=stride_size, padding="same", activation=activations)(head)
            else:
                head = self.get_conv_block(head, self.N_filters * (2 ** i_depth), activation=activations, kernel_regularizer=kernel_regularizer)
                if not gap_after_dropout:
                    gap_layers.append(self.make_global_pool_layer(head))
                # head = self.make_norm_layer(head)
                head = self.make_dropout_layer(head)
                if gap_after_dropout:
                    gap_layers.append(self.make_global_pool_layer(head))
                skip_layers.append(head)
                head = self.get_padding_block(head)
                head = self.get_pool_block(head)
            head = self.make_norm_layer(head)

        head = self.get_conv_block(head, self.N_filters * (2 ** (self.depth - 1)), activation=activations, kernel_regularizer=kernel_regularizer)
        head = self.get_conv_block(head, self.N_filters * (2 ** (self.depth - 1)), activation=activations, kernel_regularizer=kernel_regularizer)
        # head = self.make_norm_layer(head)
        if not gap_after_dropout:
            gap_layers.append(self.make_global_pool_layer(head))
        head = self.make_dropout_layer(head)
        if gap_after_dropout:
            gap_layers.append(self.make_global_pool_layer(head))
        head = self.make_norm_layer(head)
        head_lowest = head

        for i_depth in range(self.depth - 2, -1, -1):
            if i_depth == 0:
                head = Conv3DTranspose(self.N_filters * (2 ** i_depth), filter_size, strides=stride_size, padding="same", activation=activations)(head)
            else:
                head = self.get_upsampling_block(head, self.N_filters * (2 ** i_depth), activation=activations, kernel_regularizer=kernel_regularizer)

            head = self.get_cropping_block(skip_layers[i_depth], head)
            head = Concatenate()([skip_layers[i_depth], head])
            head = self.get_conv_block(head, self.N_filters * (2 ** i_depth), activation=activations, kernel_regularizer=kernel_regularizer)
            head = self.get_conv_block(head, self.N_filters * (2 ** i_depth), activation=activations, kernel_regularizer=kernel_regularizer)
            # head = self.make_norm_layer(head)
            if not gap_after_dropout and upsample_features:
                gap_layers.append(self.make_global_pool_layer(head))
            head = self.make_dropout_layer(head)
            if gap_after_dropout and upsample_features:
                gap_layers.append(self.make_global_pool_layer(head))
            head = self.make_norm_layer(head)

        # head = self.make_dropout_layer(head)
        if output_type == "softmax":
            head = self.conv_func(
                        filters=2,
                        kernel_size=1,
                        padding="same"
            )(head)
            out_mask = Activation(activation="softmax", dtype="float32", name="MASK")(head)
        elif output_type == "sigmoid":
            head = self.conv_func(
            filters=1,
            kernel_size=1,
            padding="same"
            )(head)
            out_mask = Activation(activation="sigmoid", dtype="float32", name="MASK")(head)


        genetic_features = Concatenate()(gap_layers)
        # if not gap_after_dropout:
        genetic_features = self.make_dropout_layer(genetic_features)


        branch_IDH = Dense(final_dense_unit, activation="relu")(genetic_features)

        branch_IDH = Dense(2)(branch_IDH)
        out_IDH = Activation(activation="softmax", dtype="float32", name="IDH")(branch_IDH)

        branch_1p19q = Dense(final_dense_unit, activation="relu")(genetic_features)

        branch_1p19q = Dense(2)(branch_1p19q)
        out_1p19q = Activation(activation="softmax", dtype="float32", name="1p19q")(branch_1p19q)


        branch_grade = Dense(final_dense_unit, activation="relu")(genetic_features)

        branch_grade = Dense(3)(branch_grade)
        out_grade = Activation(activation="softmax", dtype="float32", name="Grade")(branch_grade)


        predictions = [out_IDH, out_1p19q, out_grade, out_mask]

        model = Model(inputs=self.inputs, outputs=predictions)

        return model

class AdamW(tfa.optimizers.AdamW):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class AverageEarlyStopping(tensorflow.keras.callbacks.Callback):


  def __init__(self,
               monitor='val_loss',
               min_delta=0,
               patience=0,
               verbose=0,
               mode='auto',
               baseline=None,
               restore_best_weights=False,
               average_fraction=5):
    super().__init__()

    self.monitor = monitor
    self.patience = patience
    self.verbose = verbose
    self.baseline = baseline
    self.min_delta = abs(min_delta)
    self.wait = 0
    self.stopped_epoch = 0
    self.restore_best_weights = restore_best_weights
    self.best_weights = None
    self.average_fraction = average_fraction
    self.metric_history = []

    if mode not in ['auto', 'min', 'max']:
      logging.warning('EarlyStopping mode %s is unknown, '
                      'fallback to auto mode.', mode)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
    elif mode == 'max':
      self.monitor_op = np.greater
    else:
      if 'acc' in self.monitor:
        self.monitor_op = np.greater
      else:
        self.monitor_op = np.less

    if self.monitor_op == np.greater:
      self.min_delta *= 1
    else:
      self.min_delta *= -1

  def on_train_begin(self, logs=None):
    # Allow instances to be re-used
    self.wait = 0
    self.stopped_epoch = 0
    if self.baseline is not None:
      self.best = self.baseline
    else:
      self.best = np.Inf if self.monitor_op == np.less else -np.Inf

  def on_epoch_end(self, epoch, logs=None):
    current = self.get_monitor_value(logs)
    if current is None:
        return

    self.metric_history.append(current)

    if epoch > 5:
        if len(self.metric_history) > 5:
            self.metric_history.pop(0)
        current = np.mean(self.metric_history)
        # print("Best mean: {mean}, current mean: {current}".format(mean=self.best, current=current))
        print("Validation loss changed by {change}".format(change=current - self.best))
        if self.monitor_op(current - self.min_delta, self.best):

            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0 and self.verbose > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

  def get_monitor_value(self, logs):
    logs = logs or {}
    monitor_value = logs.get(self.monitor)
    if monitor_value is None:
      logging.warning('Early stopping conditioned on metric `%s` '
                      'which is not available. Available metrics are: %s',
                      self.monitor, ','.join(list(logs.keys())))
    return monitor_value
