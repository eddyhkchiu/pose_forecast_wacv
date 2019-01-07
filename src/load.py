
"""
Loading Penn Action Dataset
Follow the preprocessing steps from 3D-PFNet: https://arxiv.org/abs/1704.03432
"""

import scipy.io as spio
import numpy as np

#from utils import distance

file_path = '/vision2/u/hkchiu/data/penn_action_data/labels/'

# Penn Action Only
penn_x_max = 640.0
penn_y_max = 640.0

num_files = 2326

# For original load
full_length = {2326: 86128, 23: 1483}

# For load sample fixed length 17, start from any frame 
num_sequences_train_any = {2326: 87543, 23: 0}
num_sequences_val_any = {2326: 76298, 23: 0}


# For load sample fixed length 17, start from first step range 
num_sequences_train_step = {2326: 4478, 23: 0}
num_sequences_val_step = {2326: 3905, 23: 0}



def load_sampled_fixed_length(train, start_in_first_step, init_velocity):
  '''
  Load Penn Action dataset, for InfoGAIL to read. 
  Follow the preprocessing from https://arxiv.org/pdf/1704.03432.pdf
  Input:
  x: x coordinate of the pose: np.array (T, J)
  y: y coordinate of the pose: np.array (T, J)
  J is number of joints in a pose.
  num_samples: length of each output sequence, default is 17, for evaluating
  pose prediction of next 1 step, to next 16 steps. 
  Input frames are evently sampled to generate the output sequence.
  
  Output:
  demo: 
  demo["poses"]: np.array (N * L, P),
  demo["actions"]: np.array (N * L, P), 
  Actions represent the poses in the next frame.
  L is the total number of frames,
  P is the size of the vector that represent a pose.
  If we have 13 joints in 2D space (for Penn Action), P = 26.
  Here we use (x0, y0, x1, y1, ... , x12, y12) to represent a pose vector.
  '''
  if init_velocity:
    num_samples = 18 # 2 input, 16 output
  else:
    num_samples = 17 # 1 input, 16 output


  demo = {}
  # Penn Action Data
  # (161131, 26) for full penn action data
  # (86128, 26) for train
  if start_in_first_step:
    if train:
      demo['poses'] = np.zeros([num_sequences_train_step[num_files], num_samples, 26])
      demo['actions'] = np.zeros([num_sequences_train_step[num_files], num_samples, 26])
      demo['visibilities'] = np.zeros([num_sequences_train_step[num_files], num_samples, 13])
      demo['bounding_box_length'] = np.zeros([num_sequences_train_step[num_files]])
    else:
      demo['poses'] = np.zeros([num_sequences_val_step[num_files],num_samples, 26])
      demo['actions'] = np.zeros([num_sequences_val_step[num_files], num_samples, 26])
      demo['visibilities'] = np.zeros([num_sequences_val_step[num_files], num_samples, 13])
      demo['bounding_box_length'] = np.zeros([num_sequences_val_step[num_files]])
  else:
    if train:
      demo['poses'] = np.zeros([num_sequences_train_any[num_files], num_samples, 26])
      demo['actions'] = np.zeros([num_sequences_train_any[num_files], num_samples, 26])
      demo['visibilities'] = np.zeros([num_sequences_train_any[num_files], num_samples, 13])
      demo['bounding_box_length'] = np.zeros([num_sequences_train_any[num_files]])
    else:
      demo['poses'] = np.zeros([num_sequences_val_any[num_files],num_samples, 26])
      demo['actions'] = np.zeros([num_sequences_val_any[num_files], num_samples, 26])
      demo['visibilities'] = np.zeros([num_sequences_val_any[num_files], num_samples, 13])
      demo['bounding_box_length'] = np.zeros([num_sequences_val_any[num_files]])

  count = 0 

  for i in range(num_files):
    file_name = format(i+1, '04') + '.mat'
    annotation = spio.loadmat(file_path + file_name, squeeze_me=True)

    if (train and annotation['train'] != 1) or (not train and annotation['train'] == 1):
      continue

    # bounding box for this video for normalize
    # do not use the bounding box provided by the dataset
    x_min, x_max, y_min, y_max = get_bounding_box(annotation['x'], annotation['y'])
    bounding_box_length = max(x_max - x_min, y_max - y_min) + 1e-8

    x = annotation['x'].copy()
    y = annotation['y'].copy()


    # Normalize
    x = normalize(x, x_min, x_max, bounding_box_length)
    y = normalize(y, y_min, y_max, bounding_box_length)
  
    T, J = x.shape
    P = 2 * J 


    # Evenly sample to generate L(default 17) frames, 
    # for each output sequence starts at any frame in the input sequence 
    # Repeat the last frame when targeting output frame is out of bound.
    step = (T - 1) / num_samples 
    if start_in_first_step:
      start_range = step
    else:
      start_range = T
    for start in range(start_range): 
      poses = np.zeros([num_samples, P])
      actions = np.zeros([num_samples, P])
      visibilities = np.zeros([num_samples, J])
      for t in range(num_samples):
        idx = min(start + step * t, T - 1)
        poses[t] = np.concatenate((x[idx], y[idx]))
        visibilities[t] = annotation['visibility'][idx]
      actions[:-1] = poses[1:].copy()
      last_action_idx = min(start + step * (t+1), T - 1)
      actions[-1] = np.concatenate((x[last_action_idx], y[last_action_idx])).copy()
     
 
      demo['poses'][count] = poses
      demo['actions'][count] = actions
      demo['visibilities'][count] = visibilities
      demo['bounding_box_length'][count] = bounding_box_length
      count += 1

  print 'Number of output sequence', count #train: 87543, val: 76298
  N_total, l, P = demo['poses'].shape
  print N_total, l, P

  demo['poses'] = np.reshape(demo['poses'], [N_total * l, P])
  demo['actions'] = np.reshape(demo['actions'], [N_total * l, P])
  demo['visibilities'] = np.reshape(demo['visibilities'], [N_total * l, J])
  print demo['bounding_box_length'].shape # (N_total, )

  print 'Finished loading data.'
  return demo, N_total, l, P

def get_bounding_box(x, y):
  '''
  Input:
  x, y: (T, J)
  Output:
  x_min, x_max, y_min, y_max
  '''
  return np.min(x), np.max(x), np.min(y), np.max(y)
  
def normalize(x, x_min, x_max, bounding_box_length):
    # range (x_min, x_max) ->  (-1, 1)
    x_mid = x_min + (x_max - x_min) / 2.0
    return (x - x_mid) / (bounding_box_length / 2.0)

def un_normalize(x, x_min, x_max, bounding_box_length):
    # range (-1, 1) -> (x_min, x_max)
    x_mid = x_min + (x_max - x_min) / 2.0
    return x * (bounding_box_length / 2.0) + x_mid

def un_normalize_2d(data, x_min, x_max, y_min, y_max, bounding_box_length):
    '''
    Input:
    data: (N, P)
    P: [x0, y0, x1, y1, ...]
    '''
    new_data = data.copy()
    new_data[:, ::2] = un_normalize(data[:, ::2], x_min, x_max, bounding_box_length)
    new_data[:, 1::2] = un_normalize(data[:, 1::2], y_min, y_max, bounding_box_length)
    return new_data

def evaluate_zero_velocity(demo, N, L, P):
    '''
    Zero velocity means we predict all future frames are the same as the initial frame
    Input:
    demo["poses"]: (N * L, P)
    demo["actions"]: (N * L, P)
    demo["visibilities"]: (N * L, P/2)
    demo["bounding_box_length"]: (N,)
    '''
    prediction_steps = L - 1
    poses = np.reshape(demo["poses"], [N, L, P])
    visibilities = np.reshape(demo["visibilities"], [N, L, P/2])
    for i in range(prediction_steps):
        l2_norm, euclidean, pck = distance(poses[:, 0, :], poses[:, i+1, :],
                                           visibilities[:, i+1, :],
                                           demo["bounding_box_length"])
        print "Predict next %d steps, l2_norm: %f, euclidean: %f, pck: %f" %(i+1, l2_norm, euclidean, pck)
    

def distance(prediction, ground_truth, visibility, bounding_box_length = None):
    '''
    Input:
    prediction, ground_truth: (N, P)
    visibility: (N, J), J = P / 2, value either 1 or 0
    bounding_box_length: (N,)
    Output:
    l2_norm: l2 norm of difference of the pose vectors
    euclidean: average euclidean distance of each joints 
    pck: percentage of prediction joints in threshold distance from ground_truth
    '''
    N, P = prediction.shape
    diff = prediction - ground_truth
    l2_norm = np.average(np.linalg.norm(diff, axis = 1))

    x = diff[:, ::2]
    y = diff[:, 1::2]
    joint_distance = np.sqrt(np.square(x) + np.square(y))
    euclidean = np.average(joint_distance)

    # pck: ignore invisible joints
    # we normalize (x, y) to -1 ~ 1 before training,
    # so pck with threshold 0.05 counts the joints whose distance < 0.1
    in_threshold_distance = (joint_distance < 0.1) * visibility

    pck = np.sum(in_threshold_distance) / np.sum(visibility)
    return l2_norm, euclidean, pck



if __name__ == "__main__":
   train_data, N, L, P = load_sampled_fixed_length(True, False, True)
   val_data, N, L, P = load_sampled_fixed_length(False, False, True)
