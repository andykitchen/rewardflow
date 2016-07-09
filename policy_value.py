import numpy as np
from numpy import newaxis

import tensorflow as tf


class PolicyValueApproximation(object):
	def eval(self, state):
		raise NotImplementedError

	def sample(self):
		raise NotImplementedError


def build_nature_atari_graph():
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
		weights      = tf.Variable(tf.truncated_normal(weight_shape, stddev=stddev), name='W')
		bias         = tf.Variable(tf.zeros(out_channels), name='b')

		params += [weights, bias]

		conv_out = tf.nn.conv2d(input, weights, [1, stride, stride, 1], padding='SAME')
		bias_out = tf.nn.bias_add(conv_out, bias)
		relu_out = tf.nn.relu(bias_out)

		return relu_out

	def build_fc_layer(params, input, n_in, n_out, activation=tf.nn.relu, stddev=init_stddev):
		weights = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=stddev), name='W')
		bias    = tf.Variable(tf.zeros(n_out), name='b')

		params += [weights, bias]

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

	action_pr = fc2a_out
	value     = fc2b_out

	return state, action_pr, value, params


def build_a3c_update(opt, params, state, pi, value, actions, bigR):
	# opt: shared optimizer
	# params: parameters of our policy / value network
	# state: minibatch of states
	# pi: policy probability tensor
	# value: value tensor
	# actions: actual actions taken
	# bigR: total discounted reward up to now from the mini-episode
	
	pi_sa    = tf.reduce_sum(pi * actions, 1) # actions is one hot of action chosen
	pi_sa_lp = tf.log(pi_sa)

	bigA = bigR - value
	bigA_const = tf.stop_gradient(bigA) # advantage needs to be a constant in the first loss term

	action_loss = pi_sa_lp*bigA_const
	value_loss  = tf.square(bigA)

	loss = action_loss + value_loss

	step = opt.compute_gradients(loss, params)
	return step





class NatureAtariModel(PolicyValueApproximation):
	def __init__(self):
		super(NatureAtariModel, self).__init__()
		state_input, action_pr_tensor, value_tensor, params = build_nature_atari_graph()
		self.state_input = state_input
		self.action_pr_tensor = action_pr_tensor
		self.value_tensor = value_tensor
		self.action_indices = np.arange(action_pr_tensor.get_shape()[-1].value)
		self.params = params

	def eval(self, state):
		sess = tf.get_default_session()
		state_batch = state[newaxis, :, :, newaxis]
		action_pr_batch, value_batch = sess.run([self.action_pr_tensor, self.value_tensor], feed_dict={self.state_input: state_batch})
		action_pr = action_pr_batch[0]
		value = value_batch[0]
		return action_pr, value

	def sample(self, state):
		action_pr, value = self.eval(state)
		action = np.random.choice(self.action_indices, p=action_pr)
		return action
