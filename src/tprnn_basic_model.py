
"""
TP-RNN generic model for human motion prediction.
We can create TP-RNN model with the most basic architecture parameters:
K: scales = 2
M: number of hierarchy levels = 2

Some evaluation code are adopted from https://github.com/una-dinosauria/human-motion-prediction
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

from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Activation, Convolution2D, MaxPooling2D, Flatten, Input, merge, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
import keras.backend as K


class TPRNNBasicModel(object):
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
      num_layers: number of rnns to stack.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      summaries_dir: where to log progress for tensorboard.
      dtype: the data type to use to store internal variables.
    """
    self.dataset = dataset
    self.input_size = 54 if dataset == 'human' else 26

    # for tprnn basic model, tprnn_scale == 2
    self.tprnn_scale = tprnn_scale
    assert tprnn_scale == 2

    # for tprnn basic model, tprnn_layers == 2
    self.tprnn_layers = tprnn_layers
    assert self.tprnn_layers == 2 

    # use 2 frame to generate 1 velocity
    self.window_size = 2

    # for tprnn basic model, dropout_keep == 1.0
    self.dropout_keep = dropout_keep
    assert self.dropout_keep == 1.0
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


  def setup_forward_path(self):
    # === Transform the inputs ===
    with tf.name_scope("inputs"):

      enc_in = tf.placeholder(self.dtype, shape=[None, self.source_seq_len-1, self.input_size], name="enc_in")
      dec_in = tf.placeholder(self.dtype, shape=[None, self.target_seq_len, self.input_size], name="dec_in")
      dec_out = tf.placeholder(self.dtype, shape=[None, self.target_seq_len, self.input_size], name="dec_out")

      self.encoder_inputs = enc_in
      self.decoder_inputs = dec_in
      self.decoder_outputs = dec_out

      enc_in = tf.transpose(enc_in, [1, 0, 2])
      dec_in = tf.transpose(dec_in, [1, 0, 2])
      dec_out = tf.transpose(dec_out, [1, 0, 2])

      enc_in = tf.reshape(enc_in, [-1, self.input_size])
      dec_in = tf.reshape(dec_in, [-1, self.input_size])
      dec_out = tf.reshape(dec_out, [-1, self.input_size])

      enc_in = tf.split(enc_in, self.source_seq_len-1, axis=0)
      dec_in = tf.split(dec_in, self.target_seq_len, axis=0)
      dec_out = tf.split(dec_out, self.target_seq_len, axis=0)


    self.init_context = tf.placeholder(tf.float32, shape=[None, self.rnn_size])
    self.init_hidden = tf.placeholder(tf.float32, shape=[None, self.rnn_size])
    outputs, losses_debug_only = self.setup_context_lstm()

    self.outputs = outputs
    with tf.name_scope("loss_angles"):
      loss_angles = tf.reduce_mean(tf.square(tf.subtract(dec_out, outputs)))

    self.loss_pose = loss_angles
    self.loss = self.loss_pose

 
  def setup_optimizer(self): 
    # Gradients and SGD update operation for training the model.
    opt = tf.train.GradientDescentOptimizer( self.learning_rate )

    params = tf.trainable_variables()
    gradients = tf.gradients( self.loss, params )

    clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
    self.gradient_norms = norm
    self.updates = opt.apply_gradients(
      zip(clipped_gradients, params), global_step=self.global_step)
  
  
  def setup_log_common(self):
    # Summary writers for train and test runs
    self.train_writer = tf.summary.FileWriter(os.path.normpath(os.path.join( self.summaries_dir, 'train')))
    self.test_writer  = tf.summary.FileWriter(os.path.normpath(os.path.join( self.summaries_dir, 'test')))
    
    self.loss_summary = tf.summary.scalar('loss/loss', self.loss)

    # Keep track of the learning rate
    self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)

    self.saver = tf.train.Saver( tf.global_variables() , max_to_keep=10)


  def setup_log_human(self):
    # === variables for loss in Euler Angles -- for each action
    with tf.name_scope( "euler_error_walking" ):
      self.walking_err80   = tf.placeholder( tf.float32, name="walking_srnn_seeds_0080" )
      self.walking_err160  = tf.placeholder( tf.float32, name="walking_srnn_seeds_0160" )
      self.walking_err320  = tf.placeholder( tf.float32, name="walking_srnn_seeds_0320" )
      self.walking_err400  = tf.placeholder( tf.float32, name="walking_srnn_seeds_0400" )
      self.walking_err560  = tf.placeholder( tf.float32, name="walking_srnn_seeds_0560" )
      self.walking_err1000 = tf.placeholder( tf.float32, name="walking_srnn_seeds_1000" )
      self.walking_err80_summary   = tf.summary.scalar( 'euler_error_walking/srnn_seeds_0080', self.walking_err80 )
      self.walking_err160_summary  = tf.summary.scalar( 'euler_error_walking/srnn_seeds_0160', self.walking_err160 )
      self.walking_err320_summary  = tf.summary.scalar( 'euler_error_walking/srnn_seeds_0320', self.walking_err320 )
      self.walking_err400_summary  = tf.summary.scalar( 'euler_error_walking/srnn_seeds_0400', self.walking_err400 )
      self.walking_err560_summary  = tf.summary.scalar( 'euler_error_walking/srnn_seeds_0560', self.walking_err560 )
      self.walking_err1000_summary = tf.summary.scalar( 'euler_error_walking/srnn_seeds_1000', self.walking_err1000 )

    with tf.name_scope( "euler_error_eating" ):
      self.eating_err80   = tf.placeholder( tf.float32, name="eating_srnn_seeds_0080" )
      self.eating_err160  = tf.placeholder( tf.float32, name="eating_srnn_seeds_0160" )
      self.eating_err320  = tf.placeholder( tf.float32, name="eating_srnn_seeds_0320" )
      self.eating_err400  = tf.placeholder( tf.float32, name="eating_srnn_seeds_0400" )
      self.eating_err560  = tf.placeholder( tf.float32, name="eating_srnn_seeds_0560" )
      self.eating_err1000 = tf.placeholder( tf.float32, name="eating_srnn_seeds_1000" )
      self.eating_err80_summary   = tf.summary.scalar( 'euler_error_eating/srnn_seeds_0080', self.eating_err80 )
      self.eating_err160_summary  = tf.summary.scalar( 'euler_error_eating/srnn_seeds_0160', self.eating_err160 )
      self.eating_err320_summary  = tf.summary.scalar( 'euler_error_eating/srnn_seeds_0320', self.eating_err320 )
      self.eating_err400_summary  = tf.summary.scalar( 'euler_error_eating/srnn_seeds_0400', self.eating_err400 )
      self.eating_err560_summary  = tf.summary.scalar( 'euler_error_eating/srnn_seeds_0560', self.eating_err560 )
      self.eating_err1000_summary = tf.summary.scalar( 'euler_error_eating/srnn_seeds_1000', self.eating_err1000 )

    with tf.name_scope( "euler_error_smoking" ):
      self.smoking_err80   = tf.placeholder( tf.float32, name="smoking_srnn_seeds_0080" )
      self.smoking_err160  = tf.placeholder( tf.float32, name="smoking_srnn_seeds_0160" )
      self.smoking_err320  = tf.placeholder( tf.float32, name="smoking_srnn_seeds_0320" )
      self.smoking_err400  = tf.placeholder( tf.float32, name="smoking_srnn_seeds_0400" )
      self.smoking_err560  = tf.placeholder( tf.float32, name="smoking_srnn_seeds_0560" )
      self.smoking_err1000 = tf.placeholder( tf.float32, name="smoking_srnn_seeds_1000" )
      self.smoking_err80_summary   = tf.summary.scalar( 'euler_error_smoking/srnn_seeds_0080', self.smoking_err80 )
      self.smoking_err160_summary  = tf.summary.scalar( 'euler_error_smoking/srnn_seeds_0160', self.smoking_err160 )
      self.smoking_err320_summary  = tf.summary.scalar( 'euler_error_smoking/srnn_seeds_0320', self.smoking_err320 )
      self.smoking_err400_summary  = tf.summary.scalar( 'euler_error_smoking/srnn_seeds_0400', self.smoking_err400 )
      self.smoking_err560_summary  = tf.summary.scalar( 'euler_error_smoking/srnn_seeds_0560', self.smoking_err560 )
      self.smoking_err1000_summary = tf.summary.scalar( 'euler_error_smoking/srnn_seeds_1000', self.smoking_err1000 )

    with tf.name_scope( "euler_error_discussion" ):
      self.discussion_err80   = tf.placeholder( tf.float32, name="discussion_srnn_seeds_0080" )
      self.discussion_err160  = tf.placeholder( tf.float32, name="discussion_srnn_seeds_0160" )
      self.discussion_err320  = tf.placeholder( tf.float32, name="discussion_srnn_seeds_0320" )
      self.discussion_err400  = tf.placeholder( tf.float32, name="discussion_srnn_seeds_0400" )
      self.discussion_err560  = tf.placeholder( tf.float32, name="discussion_srnn_seeds_0560" )
      self.discussion_err1000 = tf.placeholder( tf.float32, name="discussion_srnn_seeds_1000" )
      self.discussion_err80_summary   = tf.summary.scalar( 'euler_error_discussion/srnn_seeds_0080', self.discussion_err80 )
      self.discussion_err160_summary  = tf.summary.scalar( 'euler_error_discussion/srnn_seeds_0160', self.discussion_err160 )
      self.discussion_err320_summary  = tf.summary.scalar( 'euler_error_discussion/srnn_seeds_0320', self.discussion_err320 )
      self.discussion_err400_summary  = tf.summary.scalar( 'euler_error_discussion/srnn_seeds_0400', self.discussion_err400 )
      self.discussion_err560_summary  = tf.summary.scalar( 'euler_error_discussion/srnn_seeds_0560', self.discussion_err560 )
      self.discussion_err1000_summary = tf.summary.scalar( 'euler_error_discussion/srnn_seeds_1000', self.discussion_err1000 )

    with tf.name_scope( "euler_error_directions" ):
      self.directions_err80   = tf.placeholder( tf.float32, name="directions_srnn_seeds_0080" )
      self.directions_err160  = tf.placeholder( tf.float32, name="directions_srnn_seeds_0160" )
      self.directions_err320  = tf.placeholder( tf.float32, name="directions_srnn_seeds_0320" )
      self.directions_err400  = tf.placeholder( tf.float32, name="directions_srnn_seeds_0400" )
      self.directions_err560  = tf.placeholder( tf.float32, name="directions_srnn_seeds_0560" )
      self.directions_err1000 = tf.placeholder( tf.float32, name="directions_srnn_seeds_1000" )
      self.directions_err80_summary   = tf.summary.scalar( 'euler_error_directions/srnn_seeds_0080', self.directions_err80 )
      self.directions_err160_summary  = tf.summary.scalar( 'euler_error_directions/srnn_seeds_0160', self.directions_err160 )
      self.directions_err320_summary  = tf.summary.scalar( 'euler_error_directions/srnn_seeds_0320', self.directions_err320 )
      self.directions_err400_summary  = tf.summary.scalar( 'euler_error_directions/srnn_seeds_0400', self.directions_err400 )
      self.directions_err560_summary  = tf.summary.scalar( 'euler_error_directions/srnn_seeds_0560', self.directions_err560 )
      self.directions_err1000_summary = tf.summary.scalar( 'euler_error_directions/srnn_seeds_1000', self.directions_err1000 )

    with tf.name_scope( "euler_error_greeting" ):
      self.greeting_err80   = tf.placeholder( tf.float32, name="greeting_srnn_seeds_0080" )
      self.greeting_err160  = tf.placeholder( tf.float32, name="greeting_srnn_seeds_0160" )
      self.greeting_err320  = tf.placeholder( tf.float32, name="greeting_srnn_seeds_0320" )
      self.greeting_err400  = tf.placeholder( tf.float32, name="greeting_srnn_seeds_0400" )
      self.greeting_err560  = tf.placeholder( tf.float32, name="greeting_srnn_seeds_0560" )
      self.greeting_err1000 = tf.placeholder( tf.float32, name="greeting_srnn_seeds_1000" )
      self.greeting_err80_summary   = tf.summary.scalar( 'euler_error_greeting/srnn_seeds_0080', self.greeting_err80 )
      self.greeting_err160_summary  = tf.summary.scalar( 'euler_error_greeting/srnn_seeds_0160', self.greeting_err160 )
      self.greeting_err320_summary  = tf.summary.scalar( 'euler_error_greeting/srnn_seeds_0320', self.greeting_err320 )
      self.greeting_err400_summary  = tf.summary.scalar( 'euler_error_greeting/srnn_seeds_0400', self.greeting_err400 )
      self.greeting_err560_summary  = tf.summary.scalar( 'euler_error_greeting/srnn_seeds_0560', self.greeting_err560 )
      self.greeting_err1000_summary = tf.summary.scalar( 'euler_error_greeting/srnn_seeds_1000', self.greeting_err1000 )

    with tf.name_scope( "euler_error_phoning" ):
      self.phoning_err80   = tf.placeholder( tf.float32, name="phoning_srnn_seeds_0080" )
      self.phoning_err160  = tf.placeholder( tf.float32, name="phoning_srnn_seeds_0160" )
      self.phoning_err320  = tf.placeholder( tf.float32, name="phoning_srnn_seeds_0320" )
      self.phoning_err400  = tf.placeholder( tf.float32, name="phoning_srnn_seeds_0400" )
      self.phoning_err560  = tf.placeholder( tf.float32, name="phoning_srnn_seeds_0560" )
      self.phoning_err1000 = tf.placeholder( tf.float32, name="phoning_srnn_seeds_1000" )
      self.phoning_err80_summary   = tf.summary.scalar( 'euler_error_phoning/srnn_seeds_0080', self.phoning_err80 )
      self.phoning_err160_summary  = tf.summary.scalar( 'euler_error_phoning/srnn_seeds_0160', self.phoning_err160 )
      self.phoning_err320_summary  = tf.summary.scalar( 'euler_error_phoning/srnn_seeds_0320', self.phoning_err320 )
      self.phoning_err400_summary  = tf.summary.scalar( 'euler_error_phoning/srnn_seeds_0400', self.phoning_err400 )
      self.phoning_err560_summary  = tf.summary.scalar( 'euler_error_phoning/srnn_seeds_0560', self.phoning_err560 )
      self.phoning_err1000_summary = tf.summary.scalar( 'euler_error_phoning/srnn_seeds_1000', self.phoning_err1000 )

    with tf.name_scope( "euler_error_posing" ):
      self.posing_err80   = tf.placeholder( tf.float32, name="posing_srnn_seeds_0080" )
      self.posing_err160  = tf.placeholder( tf.float32, name="posing_srnn_seeds_0160" )
      self.posing_err320  = tf.placeholder( tf.float32, name="posing_srnn_seeds_0320" )
      self.posing_err400  = tf.placeholder( tf.float32, name="posing_srnn_seeds_0400" )
      self.posing_err560  = tf.placeholder( tf.float32, name="posing_srnn_seeds_0560" )
      self.posing_err1000 = tf.placeholder( tf.float32, name="posing_srnn_seeds_1000" )
      self.posing_err80_summary   = tf.summary.scalar( 'euler_error_posing/srnn_seeds_0080', self.posing_err80 )
      self.posing_err160_summary  = tf.summary.scalar( 'euler_error_posing/srnn_seeds_0160', self.posing_err160 )
      self.posing_err320_summary  = tf.summary.scalar( 'euler_error_posing/srnn_seeds_0320', self.posing_err320 )
      self.posing_err400_summary  = tf.summary.scalar( 'euler_error_posing/srnn_seeds_0400', self.posing_err400 )
      self.posing_err560_summary  = tf.summary.scalar( 'euler_error_posing/srnn_seeds_0560', self.posing_err560 )
      self.posing_err1000_summary = tf.summary.scalar( 'euler_error_posing/srnn_seeds_1000', self.posing_err1000 )

    with tf.name_scope( "euler_error_purchases" ):
      self.purchases_err80   = tf.placeholder( tf.float32, name="purchases_srnn_seeds_0080" )
      self.purchases_err160  = tf.placeholder( tf.float32, name="purchases_srnn_seeds_0160" )
      self.purchases_err320  = tf.placeholder( tf.float32, name="purchases_srnn_seeds_0320" )
      self.purchases_err400  = tf.placeholder( tf.float32, name="purchases_srnn_seeds_0400" )
      self.purchases_err560  = tf.placeholder( tf.float32, name="purchases_srnn_seeds_0560" )
      self.purchases_err1000 = tf.placeholder( tf.float32, name="purchases_srnn_seeds_1000" )
      self.purchases_err80_summary   = tf.summary.scalar( 'euler_error_purchases/srnn_seeds_0080', self.purchases_err80 )
      self.purchases_err160_summary  = tf.summary.scalar( 'euler_error_purchases/srnn_seeds_0160', self.purchases_err160 )
      self.purchases_err320_summary  = tf.summary.scalar( 'euler_error_purchases/srnn_seeds_0320', self.purchases_err320 )
      self.purchases_err400_summary  = tf.summary.scalar( 'euler_error_purchases/srnn_seeds_0400', self.purchases_err400 )
      self.purchases_err560_summary  = tf.summary.scalar( 'euler_error_purchases/srnn_seeds_0560', self.purchases_err560 )
      self.purchases_err1000_summary = tf.summary.scalar( 'euler_error_purchases/srnn_seeds_1000', self.purchases_err1000 )

    with tf.name_scope( "euler_error_sitting" ):
      self.sitting_err80   = tf.placeholder( tf.float32, name="sitting_srnn_seeds_0080" )
      self.sitting_err160  = tf.placeholder( tf.float32, name="sitting_srnn_seeds_0160" )
      self.sitting_err320  = tf.placeholder( tf.float32, name="sitting_srnn_seeds_0320" )
      self.sitting_err400  = tf.placeholder( tf.float32, name="sitting_srnn_seeds_0400" )
      self.sitting_err560  = tf.placeholder( tf.float32, name="sitting_srnn_seeds_0560" )
      self.sitting_err1000 = tf.placeholder( tf.float32, name="sitting_srnn_seeds_1000" )
      self.sitting_err80_summary   = tf.summary.scalar( 'euler_error_sitting/srnn_seeds_0080', self.sitting_err80 )
      self.sitting_err160_summary  = tf.summary.scalar( 'euler_error_sitting/srnn_seeds_0160', self.sitting_err160 )
      self.sitting_err320_summary  = tf.summary.scalar( 'euler_error_sitting/srnn_seeds_0320', self.sitting_err320 )
      self.sitting_err400_summary  = tf.summary.scalar( 'euler_error_sitting/srnn_seeds_0400', self.sitting_err400 )
      self.sitting_err560_summary  = tf.summary.scalar( 'euler_error_sitting/srnn_seeds_0560', self.sitting_err560 )
      self.sitting_err1000_summary = tf.summary.scalar( 'euler_error_sitting/srnn_seeds_1000', self.sitting_err1000 )

    with tf.name_scope( "euler_error_sittingdown" ):
      self.sittingdown_err80   = tf.placeholder( tf.float32, name="sittingdown_srnn_seeds_0080" )
      self.sittingdown_err160  = tf.placeholder( tf.float32, name="sittingdown_srnn_seeds_0160" )
      self.sittingdown_err320  = tf.placeholder( tf.float32, name="sittingdown_srnn_seeds_0320" )
      self.sittingdown_err400  = tf.placeholder( tf.float32, name="sittingdown_srnn_seeds_0400" )
      self.sittingdown_err560  = tf.placeholder( tf.float32, name="sittingdown_srnn_seeds_0560" )
      self.sittingdown_err1000 = tf.placeholder( tf.float32, name="sittingdown_srnn_seeds_1000" )
      self.sittingdown_err80_summary   = tf.summary.scalar( 'euler_error_sittingdown/srnn_seeds_0080', self.sittingdown_err80 )
      self.sittingdown_err160_summary  = tf.summary.scalar( 'euler_error_sittingdown/srnn_seeds_0160', self.sittingdown_err160 )
      self.sittingdown_err320_summary  = tf.summary.scalar( 'euler_error_sittingdown/srnn_seeds_0320', self.sittingdown_err320 )
      self.sittingdown_err400_summary  = tf.summary.scalar( 'euler_error_sittingdown/srnn_seeds_0400', self.sittingdown_err400 )
      self.sittingdown_err560_summary  = tf.summary.scalar( 'euler_error_sittingdown/srnn_seeds_0560', self.sittingdown_err560 )
      self.sittingdown_err1000_summary = tf.summary.scalar( 'euler_error_sittingdown/srnn_seeds_1000', self.sittingdown_err1000 )

    with tf.name_scope( "euler_error_takingphoto" ):
      self.takingphoto_err80   = tf.placeholder( tf.float32, name="takingphoto_srnn_seeds_0080" )
      self.takingphoto_err160  = tf.placeholder( tf.float32, name="takingphoto_srnn_seeds_0160" )
      self.takingphoto_err320  = tf.placeholder( tf.float32, name="takingphoto_srnn_seeds_0320" )
      self.takingphoto_err400  = tf.placeholder( tf.float32, name="takingphoto_srnn_seeds_0400" )
      self.takingphoto_err560  = tf.placeholder( tf.float32, name="takingphoto_srnn_seeds_0560" )
      self.takingphoto_err1000 = tf.placeholder( tf.float32, name="takingphoto_srnn_seeds_1000" )
      self.takingphoto_err80_summary   = tf.summary.scalar( 'euler_error_takingphoto/srnn_seeds_0080', self.takingphoto_err80 )
      self.takingphoto_err160_summary  = tf.summary.scalar( 'euler_error_takingphoto/srnn_seeds_0160', self.takingphoto_err160 )
      self.takingphoto_err320_summary  = tf.summary.scalar( 'euler_error_takingphoto/srnn_seeds_0320', self.takingphoto_err320 )
      self.takingphoto_err400_summary  = tf.summary.scalar( 'euler_error_takingphoto/srnn_seeds_0400', self.takingphoto_err400 )
      self.takingphoto_err560_summary  = tf.summary.scalar( 'euler_error_takingphoto/srnn_seeds_0560', self.takingphoto_err560 )
      self.takingphoto_err1000_summary = tf.summary.scalar( 'euler_error_takingphoto/srnn_seeds_1000', self.takingphoto_err1000 )

    with tf.name_scope( "euler_error_waiting" ):
      self.waiting_err80   = tf.placeholder( tf.float32, name="waiting_srnn_seeds_0080" )
      self.waiting_err160  = tf.placeholder( tf.float32, name="waiting_srnn_seeds_0160" )
      self.waiting_err320  = tf.placeholder( tf.float32, name="waiting_srnn_seeds_0320" )
      self.waiting_err400  = tf.placeholder( tf.float32, name="waiting_srnn_seeds_0400" )
      self.waiting_err560  = tf.placeholder( tf.float32, name="waiting_srnn_seeds_0560" )
      self.waiting_err1000 = tf.placeholder( tf.float32, name="waiting_srnn_seeds_1000" )
      self.waiting_err80_summary   = tf.summary.scalar( 'euler_error_waiting/srnn_seeds_0080', self.waiting_err80 )
      self.waiting_err160_summary  = tf.summary.scalar( 'euler_error_waiting/srnn_seeds_0160', self.waiting_err160 )
      self.waiting_err320_summary  = tf.summary.scalar( 'euler_error_waiting/srnn_seeds_0320', self.waiting_err320 )
      self.waiting_err400_summary  = tf.summary.scalar( 'euler_error_waiting/srnn_seeds_0400', self.waiting_err400 )
      self.waiting_err560_summary  = tf.summary.scalar( 'euler_error_waiting/srnn_seeds_0560', self.waiting_err560 )
      self.waiting_err1000_summary = tf.summary.scalar( 'euler_error_waiting/srnn_seeds_1000', self.waiting_err1000 )

    with tf.name_scope( "euler_error_walkingdog" ):
      self.walkingdog_err80   = tf.placeholder( tf.float32, name="walkingdog_srnn_seeds_0080" )
      self.walkingdog_err160  = tf.placeholder( tf.float32, name="walkingdog_srnn_seeds_0160" )
      self.walkingdog_err320  = tf.placeholder( tf.float32, name="walkingdog_srnn_seeds_0320" )
      self.walkingdog_err400  = tf.placeholder( tf.float32, name="walkingdog_srnn_seeds_0400" )
      self.walkingdog_err560  = tf.placeholder( tf.float32, name="walkingdog_srnn_seeds_0560" )
      self.walkingdog_err1000 = tf.placeholder( tf.float32, name="walkingdog_srnn_seeds_1000" )
      self.walkingdog_err80_summary   = tf.summary.scalar( 'euler_error_walkingdog/srnn_seeds_0080', self.walkingdog_err80 )
      self.walkingdog_err160_summary  = tf.summary.scalar( 'euler_error_walkingdog/srnn_seeds_0160', self.walkingdog_err160 )
      self.walkingdog_err320_summary  = tf.summary.scalar( 'euler_error_walkingdog/srnn_seeds_0320', self.walkingdog_err320 )
      self.walkingdog_err400_summary  = tf.summary.scalar( 'euler_error_walkingdog/srnn_seeds_0400', self.walkingdog_err400 )
      self.walkingdog_err560_summary  = tf.summary.scalar( 'euler_error_walkingdog/srnn_seeds_0560', self.walkingdog_err560 )
      self.walkingdog_err1000_summary = tf.summary.scalar( 'euler_error_walkingdog/srnn_seeds_1000', self.walkingdog_err1000 )

    with tf.name_scope( "euler_error_walkingtogether" ):
      self.walkingtogether_err80   = tf.placeholder( tf.float32, name="walkingtogether_srnn_seeds_0080" )
      self.walkingtogether_err160  = tf.placeholder( tf.float32, name="walkingtogether_srnn_seeds_0160" )
      self.walkingtogether_err320  = tf.placeholder( tf.float32, name="walkingtogether_srnn_seeds_0320" )
      self.walkingtogether_err400  = tf.placeholder( tf.float32, name="walkingtogether_srnn_seeds_0400" )
      self.walkingtogether_err560  = tf.placeholder( tf.float32, name="walkingtogether_srnn_seeds_0560" )
      self.walkingtogether_err1000 = tf.placeholder( tf.float32, name="walkingtogether_srnn_seeds_1000" )
      self.walkingtogether_err80_summary   = tf.summary.scalar( 'euler_error_walkingtogether/srnn_seeds_0080', self.walkingtogether_err80 )
      self.walkingtogether_err160_summary  = tf.summary.scalar( 'euler_error_walkingtogether/srnn_seeds_0160', self.walkingtogether_err160 )
      self.walkingtogether_err320_summary  = tf.summary.scalar( 'euler_error_walkingtogether/srnn_seeds_0320', self.walkingtogether_err320 )
      self.walkingtogether_err400_summary  = tf.summary.scalar( 'euler_error_walkingtogether/srnn_seeds_0400', self.walkingtogether_err400 )
      self.walkingtogether_err560_summary  = tf.summary.scalar( 'euler_error_walkingtogether/srnn_seeds_0560', self.walkingtogether_err560 )
      self.walkingtogether_err1000_summary = tf.summary.scalar( 'euler_error_walkingtogether/srnn_seeds_1000', self.walkingtogether_err1000 )

    with tf.name_scope( "euler_error_average" ):
      self.average_err80   = tf.placeholder( tf.float32, name="average_srnn_seeds_0080" )
      self.average_err160  = tf.placeholder( tf.float32, name="average_srnn_seeds_0160" )
      self.average_err320  = tf.placeholder( tf.float32, name="average_srnn_seeds_0320" )
      self.average_err400  = tf.placeholder( tf.float32, name="average_srnn_seeds_0400" )
      self.average_err560  = tf.placeholder( tf.float32, name="average_srnn_seeds_0560" )
      self.average_err1000 = tf.placeholder( tf.float32, name="average_srnn_seeds_1000" )
      self.average_errsum = tf.placeholder( tf.float32, name="average_srnn_seeds_sum" )
      self.average_err80_summary   = tf.summary.scalar( 'euler_error_average/srnn_seeds_0080', self.average_err80 )
      self.average_err160_summary  = tf.summary.scalar( 'euler_error_average/srnn_seeds_0160', self.average_err160 )
      self.average_err320_summary  = tf.summary.scalar( 'euler_error_average/srnn_seeds_0320', self.average_err320 )
      self.average_err400_summary  = tf.summary.scalar( 'euler_error_average/srnn_seeds_0400', self.average_err400 )
      self.average_err560_summary  = tf.summary.scalar( 'euler_error_average/srnn_seeds_0560', self.average_err560 )
      self.average_err1000_summary = tf.summary.scalar( 'euler_error_average/srnn_seeds_1000', self.average_err1000 )
      self.average_errsum_summary = tf.summary.scalar( 'euler_error_average/srnn_seeds_sum', self.average_errsum )


  def setup_log_penn(self):
    # === variables for loss in Euler Angles -- for each action
    with tf.name_scope( "pck" ):
      self.pck_step1   = tf.placeholder( tf.float32, name="pck_step1" )
      self.pck_step6   = tf.placeholder( tf.float32, name="pck_step6" )
      self.pck_step11   = tf.placeholder( tf.float32, name="pck_step11" )
      self.pck_step16   = tf.placeholder( tf.float32, name="pck_step16" )

      self.pck_step1_summary   = tf.summary.scalar( 'pck/pck_step1', self.pck_step1 )
      self.pck_step6_summary   = tf.summary.scalar( 'pck/pck_step6', self.pck_step6 )
      self.pck_step11_summary   = tf.summary.scalar( 'pck/pck_step11', self.pck_step11 )
      self.pck_step16_summary   = tf.summary.scalar( 'pck/pck_step16', self.pck_step16 )



  def tprnn_one_step(self, current_inputs, current_states, lstms, step_counter, previous_contexts):
    current_outputs = []
    next_states = []

    for i in range(len(current_inputs)):
      with tf.variable_scope("lstm_" + (str(0) if i == 0 else str(1)) ):

        if i == 0: # first level
          output_state_pair = lstms[0](current_inputs[i], current_states[i])
        elif i == 1: # second level, first phase
          if step_counter % 2 == 1:
            output_state_pair = lstms[1](current_outputs[0], current_states[i])
          else:
            output_state_pair = (previous_contexts[i], current_states[i])
        elif i == 2: # second level, second phase
          if step_counter % 2 == 0:
            output_state_pair = lstms[1](current_outputs[0], current_states[i])
          else:
            output_state_pair = (previous_contexts[i], current_states[i])

        current_outputs.append(output_state_pair[0])
        next_states.append(output_state_pair[1])

    return current_outputs, next_states

  
  def apply_filters(self, current_window, filters):
    current_velocities = [tf.reshape(tf.matmul(tf.reshape(current_window, [-1, self.window_size]), filters[i]), [-1, self.input_size]) for i in range(len(filters))] 
    return current_velocities


  def setup_context_lstm(self):
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
      f1 = tf.get_variable("filter1", dtype = tf.float32, initializer = np.array([[-1.0], [1.0]], dtype=np.float32), trainable=False)
      f2 = tf.get_variable("filter2", dtype = tf.float32, initializer = np.array([[-1.0], [1.0]], dtype=np.float32), trainable=False)
      filters = [f0, f1, f2]

    self.num_filters = len(filters)
    self.f0 = f0
    self.f1 = f1
    self.f2 = f2
   
    step_counter = 0
    previous_previous_pose = self.encoder_inputs[:, 0, :]
    previous_pose = self.encoder_inputs[:, 0, :]
    current_pose = self.encoder_inputs[:, 0, :]
    current_window = tf.stack([previous_pose, current_pose], axis = 2) # shape (N, P, W)

    current_contexts = [tf.identity(self.init_context) for i in range(self.num_filters)]
    init_state = tf.concat([self.init_hidden, self.init_hidden], axis=1)
    current_states = [init_state for i in range(self.num_filters)]

    current_velocities = self.apply_filters(current_window, filters)
    # shape (N, P)

    self.velocity_generator = self.create_generator(tf.concat(current_velocities[:1], axis = 1), tf.concat(current_contexts[:2], axis = 1))

    with tf.variable_scope("setup_context_lstm"):
      lstms = [tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=False) for i in range(2)]

      # Read encoder_inputs to generate the history context
      previous_contexts = current_contexts
      next_contexts, next_states = self.tprnn_one_step(current_velocities, current_states, lstms, step_counter, previous_contexts)
      step_counter += 1

      # Not used in encoding stage
      if step_counter % 2 == 0:
        next_velocity = self.velocity_generator([tf.concat(current_velocities[:1], axis = 1), tf.concat(current_contexts[::2], axis = 1)])
      else: 
        next_velocity = self.velocity_generator([tf.concat(current_velocities[:1], axis = 1), tf.concat(current_contexts[:2], axis = 1)])
        
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
        if step_counter % 2 == 0:
          next_velocity = self.velocity_generator([tf.concat(current_velocities[:1], axis = 1), tf.concat(current_contexts[::2], axis = 1)])
        else:
          next_velocity = self.velocity_generator([tf.concat(current_velocities[:1], axis = 1), tf.concat(current_contexts[:2], axis = 1)])
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

      if step_counter % 2 == 0:
        next_velocity = self.velocity_generator([tf.concat(current_velocities[:1], axis = 1), tf.concat(current_contexts[::2], axis = 1)])
      else: 
        next_velocity = self.velocity_generator([tf.concat(current_velocities[:1], axis = 1), tf.concat(current_contexts[:2], axis = 1)])
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

        if step_counter % 2 == 0:
          next_velocity = self.velocity_generator([tf.concat(current_velocities[:1], axis = 1), tf.concat(current_contexts[::2], axis = 1)])
        else: 
          next_velocity = self.velocity_generator([tf.concat(current_velocities[:1], axis = 1), tf.concat(current_contexts[:2], axis = 1)])
        prediction = current_pose + next_velocity

        velocity_list.append(next_velocity)
        prediction_list.append(prediction)
        supervised_loss_list.append(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(prediction - self.decoder_outputs[:, 0, :]), axis=1))))


    self.acceleration_list = acceleration_list
    self.velocity_list = velocity_list
    return prediction_list, supervised_loss_list 

  def create_generator(self, poses, contexts):
    poses = Input(tensor=poses)
    contexts = Input(tensor=contexts)
    h = merge([poses, contexts], mode='concat')
    h = Dense(256)(h)
    h = LeakyReLU()(h)
    h = Dense(128)(h)
    h = LeakyReLU()(h)
    actions = Dense(self.input_size)(h)
    model = Model(input=[poses, contexts], output=actions)
    return model

  def step(self, session, encoder_inputs, decoder_inputs, decoder_outputs, 
           forward_only, srnn_seeds=False ):
    """Run a step of the model feeding the given inputs.

    Args
      session: tensorflow session to use.
      encoder_inputs: list of numpy vectors to feed as encoder inputs.
      decoder_inputs: list of numpy vectors to feed as decoder inputs.
      decoder_outputs: list of numpy vectors that are the expected decoder outputs.
      forward_only: whether to do the backward step or only forward.
      srnn_seeds: True if you want to evaluate using the sequences of SRNN
    Returns
      A triple consisting of gradient norm (or None if we did not do backward),
      mean squared error, and the outputs.
    Raises
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    input_feed = {self.encoder_inputs: encoder_inputs,
                  self.decoder_inputs: decoder_inputs,
                  self.decoder_outputs: decoder_outputs,
                  self.init_context: np.zeros([encoder_inputs.shape[0], self.rnn_size]),
                  self.init_hidden: np.zeros([encoder_inputs.shape[0], self.rnn_size])}

    # Output feed: depends on whether we do a backward step or not.
    if not srnn_seeds:
      if not forward_only:

        # Training step
        input_feed[self.dropout_keep_placeholder] = self.dropout_keep
        output_feed = [self.updates,         # Update Op that does SGD.
                       self.gradient_norms,  # Gradient norm.
                       self.loss,
                       self.loss_summary,
                       self.learning_rate_summary]

        outputs = session.run( output_feed, input_feed )
        return outputs[1], outputs[2], outputs[3], outputs[4] # Gradient norm, loss, summaries

      else:
        # Validation step, not on SRNN's seeds
        input_feed[self.dropout_keep_placeholder] = 1.0
        output_feed = [self.loss, # Loss for this batch.
                       self.outputs,
                       self.loss_summary]

        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2]  # No gradient norm
    else:
      # Validation on SRNN's seeds
      input_feed[self.dropout_keep_placeholder] = 1.0
      output_feed = [self.loss, # Loss for this batch.
                     self.outputs,
                     self.loss_summary]

      outputs = session.run(output_feed, input_feed)

      return outputs[0], outputs[1], outputs[2] # No gradient norm, loss, outputs.



  def get_batch( self, data, one_hot, actions ):
    """Get a random batch of data from the specified bucket, prepare for step.

    Args
      data: a list of sequences of size n-by-d to fit the model to.
      actions: a list of the actions we are using
    Returns
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """

    # Select entries at random
    all_keys    = list(data.keys())
    chosen_keys = np.random.choice( len(all_keys), self.batch_size )

    # How many frames in total do we need?
    total_frames = self.source_seq_len + self.target_seq_len

    encoder_inputs  = np.zeros((self.batch_size, self.source_seq_len-1, self.input_size), dtype=float)
    decoder_inputs  = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)
    decoder_outputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)

    for i in xrange( self.batch_size ):

      the_key = all_keys[ chosen_keys[i] ]

      # Get the number of frames
      n, _ = data[ the_key ].shape

      # Sample somewherein the middle
      idx = np.random.randint( 16, n-total_frames )

      # Select the data around the sampled points
      data_sel = data[ the_key ][idx:idx+total_frames ,:]

      # Add the data
      encoder_inputs[i,:,0:self.input_size]  = data_sel[0:self.source_seq_len-1, :]
      decoder_inputs[i,:,0:self.input_size]  = data_sel[self.source_seq_len-1:self.source_seq_len+self.target_seq_len-1, :]
      decoder_outputs[i,:,0:self.input_size] = data_sel[self.source_seq_len:, 0:self.input_size]

    return encoder_inputs, decoder_inputs, decoder_outputs 


  def find_indices_srnn( self, data, action ):
    """
    Find the same action indices as in SRNN.
    See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """

    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState( SEED )

    subject = 5
    subaction1 = 1
    subaction2 = 2

    T1 = data[ (subject, action, subaction1, 'even') ].shape[0]
    T2 = data[ (subject, action, subaction2, 'even') ].shape[0]
    prefix, suffix = 50, 100

    idx = []
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    return idx

  def get_batch_srnn(self, data, action, actions ):
    """
    Get a random batch of data from the specified bucket, prepare for step.

    Args
      data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
        v=nxd matrix with a sequence of poses
      action: the action to load data from
    Returns
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """

    #actions = ["directions", "discussion", "eating", "greeting", "phoning",
    #          "posing", "purchases", "sitting", "sittingdown", "smoking",
    #          "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

    if not action in actions:
      raise ValueError("Unrecognized action {0}".format(action))

    frames = {}
    frames[ action ] = self.find_indices_srnn( data, action )

    batch_size = 8 # we always evaluate 8 seeds
    subject    = 5 # we always evaluate on subject 5
    source_seq_len = self.source_seq_len
    target_seq_len = self.target_seq_len

    seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]

    encoder_inputs  = np.zeros( (batch_size, source_seq_len-1, self.input_size), dtype=float )
    decoder_inputs  = np.zeros( (batch_size, target_seq_len, self.input_size), dtype=float )
    decoder_outputs = np.zeros( (batch_size, target_seq_len, self.input_size), dtype=float )

    # Compute the number of frames needed
    total_frames = source_seq_len + target_seq_len

    # Reproducing SRNN's sequence subsequence selection as done in
    # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
    for i in xrange( batch_size ):

      _, subsequence, idx = seeds[i]
      idx = idx + 50

      data_sel = data[ (subject, action, subsequence, 'even') ]

      data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len) ,:]

      encoder_inputs[i, :, :]  = data_sel[0:source_seq_len-1, :]
      decoder_inputs[i, :, :]  = data_sel[source_seq_len-1:(source_seq_len+target_seq_len-1), :]
      decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]


    return encoder_inputs, decoder_inputs, decoder_outputs 
