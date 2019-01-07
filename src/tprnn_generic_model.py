
"""
TP-RNN generic model for human motion prediction.
We can create TP-RNN model with different architecture parameters:
K: scales
M: number of hierarchy levels. 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

import random

import numpy as np
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import data_utils
from tprnn_basic_model import TPRNNBasicModel

from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Activation, Convolution2D, MaxPooling2D, Flatten, Input, merge, Lambda, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
import keras.backend as K


class TPRNNGenericModel(TPRNNBasicModel):
  """Sequence-to-sequence model for human motion prediction"""

  def __init__(self,
               dataset,
               source_seq_len,
               target_seq_len,
               rnn_size, # hidden recurrent layer size
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               summaries_dir,
               dtype=tf.float32,
               tprnn_scale=2,
               tprnn_layers=2,
               dropout_keep=1.0
               ):
    """Create the model.

    Args:
      source_seq_len: lenght of the input sequence.
      target_seq_len: lenght of the target sequence.
      rnn_size: number of units in the rnn.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      summaries_dir: where to log progress for tensorboard.
      dtype: the data type to use to store internal variables.
      tprnn_scale: TP-RNN scale parameter.
      tprnn_layers: TP-RNN number of hierarchical levels parameter.
      dropout_keep: dropout keep probability
    """
    self.dataset = dataset
    self.input_size = 54 if dataset == 'human' else 26

    self.tprnn_scale = tprnn_scale
    assert tprnn_scale > 0

    self.tprnn_layers = tprnn_layers
    # use 2 frame to generate 1 velocity
    self.window_size = 2

    # Model hyperparameter
    self.dropout_keep = dropout_keep
    # Run-time dropout placeholder, use dropout_keep == 1.0 during evaluation
    self.dropout_keep_placeholder = tf.placeholder(tf.float32, [])

    self.summaries_dir = summaries_dir
    print( "Input size is %d" % self.input_size )
    print( "Dropout keep: ", self.dropout_keep)


    self.source_seq_len = source_seq_len
    self.target_seq_len = target_seq_len
    self.rnn_size = rnn_size
    self.max_gradient_norm = max_gradient_norm
    self.batch_size = batch_size
    self.dtype = dtype
    self.learning_rate = tf.Variable( float(learning_rate), trainable=False, dtype=dtype )
    self.learning_rate_decay_op = self.learning_rate.assign( self.learning_rate * learning_rate_decay_factor )
    self.global_step = tf.Variable(0, trainable=False)

    self.setup_forward_path()
    self.setup_optimizer()

    self.setup_log_common()
    if dataset == 'human':
      self.setup_log_human()
    else:
      self.setup_log_penn()


  def setup_optimizer(self):
    # Gradients and Adam update operation for training the model.
    opt = tf.train.AdamOptimizer( self.learning_rate )

    params = tf.trainable_variables()
    grads_and_vars = opt.compute_gradients(self.loss, params)
    grads = [gvs[0] for gvs in grads_and_vars]
    varss = [gvs[1] for gvs in grads_and_vars]

    grads, _ = tf.clip_by_global_norm(grads, self.max_gradient_norm)
    grads_and_vars = [(grads[i], varss[i]) for i in range(len(grads))]

    self.gradient_norms = tf.global_norm(grads)
    self.updates = opt.apply_gradients(grads_and_vars)


  # helpers for tprnn_one_step
  def channel_to_layer_phase(self, channel):
    remain = channel
    layer = 0
    while(remain + 1 > np.power(self.tprnn_scale, layer)):
      remain -= int(np.power(self.tprnn_scale, layer))
      layer += 1
    return layer, remain

  def layer_phase_to_channel(self, layer, phase):
    return int(np.sum([np.power(self.tprnn_scale, i) for i in range(layer)]) + phase)
  
  def match_phase_from_step_layer(self, step, layer):
    return int((step-1)%(np.power(self.tprnn_scale, layer)))


  def tprnn_one_step(self, current_inputs, current_states, lstms, step_counter, previous_contexts):
    current_outputs = []
    next_states = []

    for i in range(len(current_inputs)):
      layer, phase = self.channel_to_layer_phase(i)
      with tf.variable_scope("lstm_" + str(layer)):

        if self.match_phase_from_step_layer(step_counter, layer) == phase:
          # update
          if layer == 0:
            current_input = current_inputs[0]
          else:
            input_layer = layer - 1
            input_phase = phase % (np.power(self.tprnn_scale, layer - 1))
            current_input = current_outputs[self.layer_phase_to_channel(input_layer, input_phase)]

          # apply dropout on the input data first
          current_input = tf.nn.dropout(current_input, self.dropout_keep_placeholder)
          output_state_pair = lstms[layer](current_input, current_states[i])
        else:
          # copy
          output_state_pair = (previous_contexts[i], current_states[i])

        current_outputs.append(output_state_pair[0])
        next_states.append(output_state_pair[1])

    return current_outputs, next_states
  
  def apply_filters(self, current_window, filters):
    current_velocities = [tf.reshape(tf.matmul(tf.reshape(current_window, [-1, self.window_size]), filters[i]), [-1, self.input_size]) for i in range(len(filters))] 
    return current_velocities


  def setup_context_lstm(self, append_activity_prediction=False):
    """
    Input:
    self.encoder_inputs: [None, source_seq_len-1, self.input_size]
    self.decoder_inputs: [None, target_seq_len, self.input_size]
    self.decoder_outputs: [None, target_seq_len, self.input_size]
    Output:
    outputs: [None, target_seq_len, self.input_size]
    """

    supervised_loss_list = []
    prediction_list = []
    velocity_list = []
    acceleration_list = []

    with tf.variable_scope("setup_context_lstm_filter"):
      f0 = tf.get_variable("filter0", dtype = tf.float32, initializer = np.array([[-1.0], [1.0]], dtype=np.float32), trainable=False)
      print(f0.shape)
    self.num_channels = self.layer_phase_to_channel(self.tprnn_layers, 0)  
    print("number of channels: %d", self.num_channels)
    print("number of layers: %d", self.tprnn_layers)
    print("scale: %d", self.tprnn_scale)
    filters = [f0] * self.num_channels

    step_counter = 0
    #previous_previous_previous_pose = self.encoder_inputs[:, 0, :]
    previous_previous_pose = self.encoder_inputs[:, 0, :]
    previous_pose = self.encoder_inputs[:, 0, :]
    current_pose = self.encoder_inputs[:, 0, :]
    current_window = tf.stack([previous_pose, current_pose], axis = 2) # shape (N, P, W)
    print(current_window.shape)

    current_contexts = [tf.identity(self.init_context) for i in range(self.num_channels)]
    init_state = tf.concat([self.init_hidden, self.init_hidden], axis=1)
    print(init_state.shape)
    current_states = [init_state for i in range(self.num_channels)]

    current_velocities = self.apply_filters(current_window, filters)
    # shape (N, P)
    print(current_velocities[0].shape)

    with tf.variable_scope("setup_context_lstm"):
      lstms = [tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=False) for i in range(self.tprnn_layers)]

      # Read encoder_inputs to generate the history context
      previous_contexts = current_contexts
      next_contexts, next_states = self.tprnn_one_step(current_velocities, current_states, lstms, step_counter, previous_contexts)
      step_counter += 1

      # Not used in encoding stage
      next_velocity = self.velocity_generator(tf.concat(current_velocities[:1], axis = 1), tf.concat([current_contexts[self.match_phase_from_step_layer(step_counter, layer)] for layer in range(self.tprnn_layers)], axis = 1))
      prediction = current_pose + next_velocity

      for i in range(1, self.source_seq_len-1):
        current_pose = self.encoder_inputs[:, i, :]
        previous_pose = self.encoder_inputs[:, max(0, i-1), :]
        previous_previous_pose = self.encoder_inputs[:, max(0, i-2), :]
        current_window = tf.stack([previous_pose, current_pose], axis = 2) # shape (N, P, W)

        current_velocities = self.apply_filters(current_window, filters)

        previous_contexts = current_contexts
        current_contexts = next_contexts
        current_states = next_states

        next_contexts, next_states = self.tprnn_one_step(current_velocities, current_states, lstms, step_counter, previous_contexts)
        step_counter += 1

        # Not used in encoding stage
        next_velocity = self.velocity_generator(tf.concat(current_velocities[:1], axis = 1), tf.concat([current_contexts[self.match_phase_from_step_layer(step_counter, layer)] for layer in range(self.tprnn_layers)], axis = 1))
        prediction = current_pose + next_velocity

 
      # Start to predict decoder_outputs
      previous_previous_pose = self.encoder_inputs[:, self.source_seq_len-3, :]
      previous_pose = self.encoder_inputs[:, self.source_seq_len-2, :]
      current_pose = self.decoder_inputs[:, 0, :]
      current_window = tf.stack([previous_pose, current_pose], axis = 2) # shape (N, P, W)

      current_velocities = self.apply_filters(current_window, filters)

      previous_contexts = current_contexts
      current_contexts = next_contexts
      current_states = next_states

      next_contexts, next_states = self.tprnn_one_step(current_velocities, current_states, lstms, step_counter, previous_contexts)
      step_counter += 1

      # here we finished feed all the input poses into tprnn, 
      next_velocity = self.velocity_generator(tf.concat(current_velocities[:1], axis = 1), tf.concat([current_contexts[self.match_phase_from_step_layer(step_counter, layer)] for layer in range(self.tprnn_layers)], axis = 1))
      prediction = current_pose + next_velocity

      velocity_list.append(next_velocity)
      prediction_list.append(prediction)
      supervised_loss_list.append(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(prediction - self.decoder_outputs[:, 0, :]), axis=1))))

      for i in range(1, self.target_seq_len):
        previous_previous_pose = previous_pose
        previous_pose = current_pose
        current_pose = prediction
        current_window = tf.stack([previous_pose, current_pose], axis = 2) # shape (N, P, W)

        current_velocities = self.apply_filters(current_window, filters)

        previous_contexts = current_contexts
        current_contexts = next_contexts
        current_states = next_states

        next_contexts, next_states = self.tprnn_one_step(current_velocities, current_states, lstms, step_counter, previous_contexts)
        step_counter += 1 

        next_velocity = self.velocity_generator(tf.concat(current_velocities[:1], axis = 1), tf.concat([current_contexts[self.match_phase_from_step_layer(step_counter, layer)] for layer in range(self.tprnn_layers)], axis = 1))
        prediction = current_pose + next_velocity

        velocity_list.append(next_velocity)
        prediction_list.append(prediction)
        supervised_loss_list.append(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(prediction - self.decoder_outputs[:, 0, :]), axis=1))))


    self.acceleration_list = acceleration_list
    self.velocity_list = velocity_list
    return prediction_list, supervised_loss_list 

  def velocity_generator(self, poses, contexts):
    with tf.variable_scope("generator") as scope:
      inputs = tf.concat([poses, contexts], axis=1)
      inputs = tf.layers.dense(inputs, 256, name="generator1", reuse=tf.AUTO_REUSE)
      inputs = tf.nn.leaky_relu(inputs)
      inputs = tf.nn.dropout(inputs, self.dropout_keep_placeholder)
      inputs = tf.layers.dense(inputs, 128, name="generator2", reuse=tf.AUTO_REUSE)
      inputs = tf.nn.leaky_relu(inputs)
      inputs = tf.nn.dropout(inputs, self.dropout_keep_placeholder)
      outputs = tf.layers.dense(inputs, self.input_size, name="generator_output", reuse=tf.AUTO_REUSE)
    return outputs
