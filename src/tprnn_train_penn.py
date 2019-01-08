
"""
Simple code for training an RNN for motion prediction on Penn Action Dataset.
Similar to https://github.com/una-dinosauria/human-motion-prediction
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import h5py

import numpy as np
from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import tprnn_basic_model
from load import load_sampled_fixed_length, distance

# Learning
tf.app.flags.DEFINE_float("learning_rate", .005, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate is multiplied by this much. 1 means no decay.")
tf.app.flags.DEFINE_integer("learning_rate_step", 10000, "Every this many steps, do decay.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("iterations", int(1e5), "Iterations to train for.")
# Architecture
tf.app.flags.DEFINE_integer("rnn_size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("seq_length_in", 50, "Number of frames to feed into the encoder. 25 fps")
tf.app.flags.DEFINE_integer("seq_length_out", 10, "Number of frames that the decoder has to predict. 25fps")
tf.app.flags.DEFINE_boolean("omit_one_hot", True, "Whether to remove one-hot encoding from the data")
# Directories
#tf.app.flags.DEFINE_string("data_dir", os.path.normpath("./data/h3.6m/dataset"), "Data directory")
tf.app.flags.DEFINE_string("train_dir", os.path.normpath("./experiments/"), "Training directory.")

tf.app.flags.DEFINE_string("action","all", "The action to train on. all means all the actions, all_periodic means walking, eating and smoking")
tf.app.flags.DEFINE_string("loss_to_use","sampling_based", "The type of loss to use, supervised or sampling_based")

tf.app.flags.DEFINE_integer("test_every", 1000, "How often to compute error on the test set.")
tf.app.flags.DEFINE_integer("save_every", 1000, "How often to compute error on the test set.")
tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")

# New model paramteres
tf.app.flags.DEFINE_string("dataset", 'human', "human or penn")
tf.app.flags.DEFINE_integer("tprnn_scale", 2, "scale of tprnn")
tf.app.flags.DEFINE_integer("tprnn_layers", 2, "number of layers of tprnn")
tf.app.flags.DEFINE_integer("more", 0, "More iterations to train for.")
tf.app.flags.DEFINE_float("dropout_keep", 1.0, "Dropout keep probability.")
tf.app.flags.DEFINE_string("model", 'basic', "basic or generic")
tf.app.flags.DEFINE_boolean("init_velocity", False, "use init velocity")


FLAGS = tf.app.flags.FLAGS


train_dir = os.path.normpath(os.path.join(FLAGS.train_dir, 'penn',
  'out_{0}'.format(FLAGS.seq_length_out),
  'iterations_{0}'.format(FLAGS.iterations),
  'rnn_size_{0}'.format(FLAGS.rnn_size),
  'lr_{0}'.format(FLAGS.learning_rate)))

if FLAGS.tprnn_scale > 0:
  train_dir = os.path.normpath(os.path.join(train_dir,
    'tprnn_scale_{0}'.format(FLAGS.tprnn_scale)))
if FLAGS.tprnn_layers > 0:
  train_dir = os.path.normpath(os.path.join(train_dir,
    'tprnn_layers_{0}'.format(FLAGS.tprnn_layers)))
train_dir = os.path.normpath(os.path.join(train_dir,
  'dropout_keep_{0}'.format(FLAGS.dropout_keep)))
if FLAGS.model == 'basic':
  train_dir = os.path.normpath(os.path.join(train_dir, 'basic'))
else:
  train_dir = os.path.normpath(os.path.join(train_dir, 'generic'))

if FLAGS.init_velocity:
  train_dir = os.path.normpath(os.path.join(train_dir, 'init_velocity'))
else:
  train_dir = os.path.normpath(os.path.join(train_dir, 'zero_init_velocity'))

summaries_dir = os.path.normpath(os.path.join( train_dir, "log" )) # Directory for TB summaries

def create_model(session, sampling=False):
  """Create translation model and initialize or load parameters in session."""

  from keras import backend as K
  K.set_session(session)

  assert(FLAGS.model == 'basic')
  model = tprnn_basic_model.TPRNNBasicModel(
      FLAGS.dataset,
      FLAGS.seq_length_in,# if not sampling else 50,
      FLAGS.seq_length_out,# if not sampling else 100,
      FLAGS.rnn_size, # hidden layer size
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      summaries_dir,
      tf.float32,
      FLAGS.tprnn_scale,
      FLAGS.tprnn_layers,
      FLAGS.dropout_keep
      )

  if FLAGS.load <= 0:
    print("Creating model with fresh parameters.")
    session.run(tf.global_variables_initializer())
    return model

  load_dir = train_dir
  ckpt = tf.train.get_checkpoint_state( load_dir, latest_filename="checkpoint")
  print( "load_dir", load_dir )

  if ckpt and ckpt.model_checkpoint_path:
    # Check if the specific checkpoint exists
    if FLAGS.load > 0:
      if os.path.isfile(os.path.join(load_dir,"checkpoint-{0}.index".format(FLAGS.load))):
        ckpt_name = os.path.normpath(os.path.join( os.path.join(load_dir,"checkpoint-{0}".format(FLAGS.load)) ))
      else:
        raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(FLAGS.load))
    else:
      ckpt_name = os.path.basename( ckpt.model_checkpoint_path )

    print("Loading model {0}".format( ckpt_name ))
    #model.saver.restore( session, ckpt.model_checkpoint_path )
    model.saver.restore( session, ckpt_name )
    return model
  else:
    print("Could not find checkpoint. Aborting.")
    raise( ValueError, "Checkpoint {0} does not seem to exist".format( ckpt.model_checkpoint_path ) )

  return model


def train():
  # Limit TF to take a fraction of the GPU memory
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}

  with tf.Session(config=tf.ConfigProto( gpu_options=gpu_options, device_count = device_count )) as sess:

    # === Create the model ===
    model = create_model( sess )
    model.train_writer.add_graph( sess.graph )
    print( "Model created" )

    #=== This is the training loop ===
    step_time, loss, val_loss = 0.0, 0.0, 0.0
    current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1
    previous_losses = []
    step_time, loss = 0, 0

    # Choose the best model during training using avg_mean_errors_sum
    best_pck16 = 0
    best_pck = np.zeros(16)
    best_current_step = 1

    iterations_to_train = FLAGS.iterations if FLAGS.more == 0 else FLAGS.more

    # Load Penn Action Data once, sample batch inside iteration loop
    print(FLAGS.init_velocity)
    penn_train = True
    start_in_first_step = False
    penn_train_data, N_train, L, P = load_sampled_fixed_length(penn_train, start_in_first_step, FLAGS.init_velocity)
    poses_train = np.reshape(penn_train_data["poses"], [N_train, L, P])
    # Duplicate and prepend the first frame
    first_frame_train = poses_train[:, :1, :].copy()
    poses_train = np.concatenate((first_frame_train, poses_train), axis = 1)

    penn_train = False
    penn_val_data, N_val, L, P = load_sampled_fixed_length(penn_train, start_in_first_step, FLAGS.init_velocity)
    poses_val = np.reshape(penn_val_data["poses"], [N_val, L, P])
    # Duplicate and prepend the first frame, 
    # equivalent to having 0 initial velocity
    first_frame_val = poses_val[:, :1, :].copy()
    poses_val = np.concatenate((first_frame_val, poses_val), axis = 1)
    visibilities = np.reshape(penn_val_data["visibilities"], [-1, L, 13])
    

    if FLAGS.init_velocity:
      encoder_length = 2
    else:
      encoder_length = 1

    for _ in xrange( iterations_to_train ):

      start_time = time.time()

      # === Training step ===
      # Penn Action Data
      poses_train_batch = poses_train[np.random.choice(N_train, FLAGS.batch_size)]
      encoder_inputs = poses_train_batch[:, :encoder_length, :]
      decoder_inputs = poses_train_batch[:, encoder_length:-1, :]
      decoder_outputs = poses_train_batch[:, encoder_length+1:, :]

      _, step_loss, loss_summary, lr_summary = model.step( sess, encoder_inputs, decoder_inputs, decoder_outputs, False )
      model.train_writer.add_summary( loss_summary, current_step )
      model.train_writer.add_summary( lr_summary, current_step )

      if current_step % 10 == 0:
        print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss ))

      step_time += (time.time() - start_time) / FLAGS.test_every
      loss += step_loss / FLAGS.test_every
      current_step += 1

      # === step decay ===
      if current_step % FLAGS.learning_rate_step == 0:
        sess.run(model.learning_rate_decay_op)

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.test_every == 0:

        # === Validation ===
        forward_only = True

        poses_val_batch = poses_val
        encoder_inputs = poses_val_batch[:, :encoder_length, :]
        decoder_inputs = poses_val_batch[:, encoder_length:-1, :]
        decoder_outputs = poses_val_batch[:, encoder_length+1:, :]
        step_loss, predicted_outputs, loss_summary = model.step(sess,
            encoder_inputs, decoder_inputs, decoder_outputs, forward_only)
        val_loss = step_loss # Loss book-keeping
        print("Validation loss: ", val_loss)

        # Evaluation using pck score
        pck = np.zeros(L - encoder_length)
        for i in range(L - encoder_length):
          l2_norm, euclidean, pck[i] = distance(
            predicted_outputs[i], 
            poses_val[:, i + encoder_length + 1, :],
            visibilities[:, i + encoder_length, :],
            penn_val_data["bounding_box_length"])
          print( "Predict next %d steps, l2_norm: %f, euclidean: %f, pck: %f" %(i+1, l2_norm, euclidean, pck[i]))

        model.test_writer.add_summary(loss_summary, current_step)

        print()
        print("{0: <16} |".format("milliseconds"), end="")
        for ms in [80, 160, 320, 400, 560, 1000]:
          print(" {0:5d} |".format(ms), end="")
        print()

          # Ugly massive if-then to log the error to tensorboard :shrug:
        pck_summaries = sess.run(
            [model.pck_step1_summary,
             model.pck_step6_summary,
             model.pck_step11_summary,
             model.pck_step16_summary],
            {model.pck_step1: pck[0],
             model.pck_step6: pck[5],
             model.pck_step11: pck[10],
             model.pck_step16: pck[15]})

        for i in np.arange(len( pck_summaries )):
          model.test_writer.add_summary(pck_summaries[i], current_step)

        if pck[15] >= best_pck16 or np.average(pck) >= np.average(best_pck):
          print('Found a better model')
          best_pck16 = pck[15]
          best_pck = pck.copy()
          best_current_step = current_step
          print( "Best model's checkpoint id: {0}".format(best_current_step))
          print( "Best model's pck16: {0}".format(best_pck16))
          print( "Best model's avg pck: {0}".format(np.average(best_pck)))

        # Save the model
        if current_step % FLAGS.save_every == 0 and current_step == best_current_step:
          print( "Saving the model..." ); start_time = time.time()
          print( "at {0}".format(train_dir))
          model.saver.save(sess, os.path.normpath(os.path.join(train_dir, 'checkpoint')), global_step=current_step )
          print( "done in {0:.2f} ms".format( (time.time() - start_time)*1000) )
          print( "Best model's checkpoint id: {0}".format(best_current_step))
          print( "Best model's pck16: {0}".format(best_pck16))
          print( "Best model's avg pck: {0}".format(np.average(best_pck)))

        # Reset global time and loss
        step_time, loss = 0, 0
        sys.stdout.flush()

    print( "Saving the last model..." ); start_time = time.time()
    print( "at {0}".format(train_dir))
    model.saver.save(sess, os.path.normpath(os.path.join(train_dir, 'checkpoint')), global_step=current_step )
    print( "done in {0:.2f} ms".format( (time.time() - start_time)*1000) )
    print( "Best model's checkpoint id: {0}".format(best_current_step))
    print( "Best model's pck16: {0}".format(best_pck16))
    print( "Best model's avg pck: {0}".format(np.average(best_pck)))


def sample():
  """Sample predictions for srnn's seeds"""

  if FLAGS.load <= 0:
    raise( ValueError, "Must give an iteration to read parameters from")

  # Use the CPU if asked to
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.Session(config=tf.ConfigProto( device_count = device_count )) as sess:

    # === Create the model ===
    sampling     = True
    model = create_model(sess, sampling)
    print("Model created")

    # Clean and create a new h5 file of samples
    SAMPLES_FNAME = 'samples.h5'
    try:
      os.remove( SAMPLES_FNAME )
    except OSError:
      pass

    start_in_first_step = False
    penn_train = False
    penn_val_data, N_val, L, P = load_sampled_fixed_length(penn_train, start_in_first_step, FLAGS.init_velocity)
    poses_val = np.reshape(penn_val_data["poses"], [N_val, L, P])
    # Duplicate and prepend the first frame
    # equivalent to having 0 initial velocity
    first_frame_val = poses_val[:, :1, :].copy()
    poses_val = np.concatenate((first_frame_val, poses_val), axis = 1)
    visibilities = np.reshape(penn_val_data["visibilities"], [-1, L, 13])

    # === Validation ===
    forward_only = True

    poses_val_batch = poses_val

    if FLAGS.init_velocity:
      encoder_length = 2
    else:
      encoder_length = 1

    encoder_inputs = poses_val_batch[:, :encoder_length, :]
    decoder_inputs = poses_val_batch[:, encoder_length:-1, :]
    decoder_outputs = poses_val_batch[:, encoder_length+1:, :]
    step_loss, predicted_outputs, loss_summary = model.step(sess,
        encoder_inputs, decoder_inputs, decoder_outputs, forward_only)
    val_loss = step_loss # Loss book-keeping
    print("Validation loss: ", val_loss)

    # Evaluation using pck score
    pck = np.zeros(L - encoder_length)
    for i in range(L - encoder_length):
      l2_norm, euclidean, pck[i] = distance(
        predicted_outputs[i],
        poses_val[:, i + encoder_length + 1, :],
        visibilities[:, i + encoder_length, :],
        penn_val_data["bounding_box_length"])
      print( "Predict next %d steps, l2_norm: %f, euclidean: %f, pck: %f" %(i+1, l2_norm, euclidean, pck[i]))
  return


def main(_):
  if FLAGS.sample:
    sample()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
