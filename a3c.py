import tensorflow as tf

def build_graph():
	minibatch_size  = None
	screen_height   = 83
	screen_width    = 83
	history_size    = 1
	init_stddev     = 10e-4

	filter_height   = 8
	filter_width    = 8
	conv1_features  = 16
	conv1_stride    = 4
	conv1_height    = (screen_height + conv1_stride) // conv1_stride
	conv1_width     = (screen_width  + conv1_stride) // conv1_stride

	conv2_features  = 32
	conv2_stride    = 2
	conv2_height    = (conv1_height + conv2_stride) // conv2_stride
	conv2_width     = (conv1_width  + conv2_stride) // conv2_stride

	conv2_flat_size = conv2_height * conv2_width * conv2_features
	fc1_features    = 256
	fc2a_features   = 6
	fc2b_features   = 1

	state = tf.placeholder(tf.float32, [minibatch_size, screen_width, screen_height, history_size])
	params = []

	def build_conv_layer(params, input, in_channels, out_channels, stride, activation=tf.nn.relu, stddev=init_stddev):
		weight_shape = [filter_height, filter_width, in_channels, out_channels]
		weights      = tf.Variable(tf.truncated_normal(weight_shape, stddev=stddev))
		bias         = tf.Variable(tf.zeros(out_channels))

		params.append(weights)
		params.append(bias)

		conv_out = tf.nn.conv2d(input, weights, [1, stride, stride, 1], padding='SAME')
		bias_out = tf.nn.bias_add(conv_out, bias)
		relu_out = tf.nn.relu(bias_out)

		return relu_out

	def build_fc_layer(params, input, n_in, n_out, activation=tf.nn.relu, stddev=init_stddev):
		weights = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=stddev))
		bias    = tf.Variable(tf.zeros(n_out))

		params.append(weights)
		params.append(bias)

		matmul_out = tf.matmul(input, weights)
		bias_out   = tf.nn.bias_add(matmul_out, bias)

		if activation:
			output = activation(bias_out)
		else:
			output = bias_out

		return output

	conv1_out  = build_conv_layer(params, state, history_size, conv1_features, conv1_stride)
	conv2_out  = build_conv_layer(params, conv1_out, conv1_features, conv2_features, conv2_stride)
	conv2_flat = tf.reshape(conv2_out, [-1, conv2_flat_size])
	fc1_out    = build_fc_layer(params, conv2_flat, conv2_flat_size, fc1_features)
	fc2a_out   = build_fc_layer(params, fc1_out, fc1_features, fc2a_features, activation=tf.nn.softmax)
	fc2b_out   = build_fc_layer(params, fc1_out, fc1_features, fc2b_features, activation=None)

	action_out = fc2a_out
	val_out    = fc2b_out

	return state, action_out, val_out, params

import sys
import time
import random
import cv2
import scipy.misc

sys.path.insert(0, '../upstream/Arcade-Learning-Environment')

import ale_python_interface

rom_path = 'space_invaders.bin'

import numpy as np
from numpy import newaxis
import threading

class AleThread(threading.Thread):
	def __init__(self, sess, coord, rom_path, count_op, state_input, action_out, local_params, shared_params):
		super(AleThread, self).__init__()
		self.ale = ale_python_interface.ALEInterface()
		self.ale.loadROM(rom_path)
		self.actions = self.ale.getMinimalActionSet()

		self.sess = sess
		self.coord = coord
		self.count_op = count_op
		self.state_input = state_input
		self.action_out = action_out
		self.local_params = local_params
		self.shared_params = shared_params
		self.copy_params_op = tf.group(*[tf.assign(local, shared) for local, shared in zip(local_params, shared_params)])

	def run(self):
		with coord.stop_on_exception():
			while not self.coord.should_stop():
				self.step()
	
	def get_state(self):
		frame = self.ale.getScreenGrayscale()
		frame_small = scipy.misc.imresize(frame[:,:,0], (83, 83), interp='bilinear')
		return frame_small

	def terminal_state(self):
		return self.ale.game_over()

	def start_new_episode(self):
		self.ale.reset_game()

	def eval_policy(self):
		self.sess.run(self.copy_params_op)
		state = self.get_state()
		state_batch = state[newaxis, :, :, newaxis]
		action_pr_batch = self.sess.run(self.action_out, feed_dict={self.state_input: state_batch})
		action_pr = action_pr_batch[0]
		return action_pr

	def sample_policy(self):
		action_pr = self.eval_policy()
		action = np.random.choice(self.actions, p=action_pr)
		return action

	def step(self):
		action = self.sample_policy()
		reward = self.ale.act(action)
		count = self.sess.run(self.count_op)

		if reward > 0:
			print count, ":", reward
		if self.terminal_state():
			print count, ":", "end of episode"
			self.start_new_episode()

bigT = tf.Variable(0, dtype=tf.int64)
count_op = bigT.assign_add(1)

_, _, _, shared_params = build_graph()

with tf.Session() as sess:
	def build_actor_thread():
		state, action_out, val_out, params = build_graph()
		thread = AleThread(sess, coord, rom_path, count_op, state, action_out, params, shared_params)
		return thread

	coord = tf.train.Coordinator()
	threads = [build_actor_thread() for i in range(3)]
	tf.initialize_all_variables().run()
	for t in threads:
		t.start()

	coord.join(threads)
