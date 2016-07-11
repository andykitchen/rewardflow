import numpy as np
from numpy import newaxis

import tensorflow as tf


class PolicyValueApproximation(object):
	def eval(self, state):
		raise NotImplementedError

	def sample(self):
		raise NotImplementedError


def build_nips_atari_graph():
	minibatch_size  = None
	screen_height   = 84
	screen_width    = 84
	history_size    = 4
	init_stddev     = 10e-4

	conv1_size      = (8, 8)
	conv1_features  = 16
	conv1_stride    = 4

	conv2_size      = (4, 4)
	conv2_features  = 32
	conv2_stride    = 2

	fc1_features    = 256
	fc2a_features   = 6
	fc2b_features   = 1

	state = tf.placeholder(tf.float32, [minibatch_size, screen_width, screen_height, history_size])
	params = []

	def build_conv_layer(params, input, filter_size, in_channels, out_channels, stride, activation=tf.nn.relu, stddev=init_stddev, scope='conv'):
		with tf.variable_scope(scope):
			filter_height, filter_width = filter_size
			weight_shape = [filter_height, filter_width, in_channels, out_channels]
			weights      = tf.Variable(tf.truncated_normal(weight_shape, stddev=stddev), name='W')
			bias         = tf.Variable(tf.zeros(out_channels), name='b')

			params += [weights, bias]

			conv_out = tf.nn.conv2d(input, weights, [1, stride, stride, 1], padding='VALID')
			bias_out = tf.nn.bias_add(conv_out, bias)
			relu_out = tf.nn.relu(bias_out)

		return relu_out

	def build_fc_layer(params, input, n_in, n_out, activation=tf.nn.relu, stddev=init_stddev, scope='fc'):
		with tf.variable_scope(scope):
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

	with tf.variable_scope('q_network'):
		conv1_out  = build_conv_layer(params, state, conv1_size, history_size, conv1_features, conv1_stride, scope='conv1')
		conv2_out  = build_conv_layer(params, conv1_out, conv2_size, conv1_features, conv2_features, conv2_stride, scope='conv2')

		conv2_flat_size = np.product(conv2_out.get_shape().as_list()[1:])

		conv2_flat = tf.reshape(conv2_out, [-1, conv2_flat_size])
		fc1_out    = build_fc_layer(params, conv2_flat, conv2_flat_size, fc1_features, scope='fc1')
		fc2a_out   = build_fc_layer(params, fc1_out, fc1_features, fc2a_features, activation=tf.nn.softmax, scope='fc2a')
		fc2b_out   = build_fc_layer(params, fc1_out, fc1_features, fc2b_features, activation=None, scope='fc2b')

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

	value = tf.squeeze(value, squeeze_dims=[1])
	bigA = bigR - value
	bigA_const = tf.stop_gradient(bigA) # advantage needs to be a constant in the first loss term

	pi_loss      = -pi_sa_lp*bigA_const
	value_loss   = tf.square(bigA)
	entropy_loss = -tf.reduce_sum(pi*tf.log(pi), 1)

	beta0 = 0.5
	beta1 = 0.01

	loss = pi_loss + beta0*value_loss + beta1*entropy_loss
	assert loss.get_shape().ndims == 1

	batch_loss = tf.reduce_mean(loss)

	step = opt.compute_gradients(batch_loss, params)
	return step, loss


class CentralizedThingy(object):
	def __init__(self, shared_optimizer, shared_model):
		self.shared_optimizer = shared_optimizer
		self.shared_model = shared_model

	def build_copy_op(self, model):
		shared_params = self.shared_model.params
		copy_ops = [tf.assign(local, shared) for local, shared in zip(model.params, shared_params)]
		copy_op  = tf.group(*copy_ops)
		return copy_op

	def build_update_op(self, model):
		actions = tf.placeholder(tf.float32, [None, 6])
		bigR = tf.placeholder(tf.float32, [None])

		step, loss = build_a3c_update(
			self.shared_optimizer,
			model.params,
			model.state_input,
			model.action_pr_tensor,
			model.value_tensor,
			actions,
			bigR)

		grad_shared = [(grad, shared_var) for (grad, var), shared_var in zip(step, self.shared_model.params)]
		update_op = self.shared_optimizer.apply_gradients(grad_shared)

		return update_op, actions, bigR


class AtariNIPSModel(PolicyValueApproximation):
	def __init__(self):
		super(AtariNIPSModel, self).__init__()
		state_input, action_pr_tensor, value_tensor, params = build_nips_atari_graph()
		self.state_input = state_input
		self.action_pr_tensor = action_pr_tensor
		self.value_tensor = value_tensor
		self.action_indices = np.arange(action_pr_tensor.get_shape()[-1].value)
		self.params = params

	def eval_policy(self, state):
		sess = tf.get_default_session()
		state_batch = state[newaxis, :, :]
		action_pr_batch = sess.run(self.action_pr_tensor,
			feed_dict={self.state_input: state_batch})
		action_pr = action_pr_batch[0]
		return action_pr

	def eval_value(self, state):
		sess = tf.get_default_session()
		state_batch = state[newaxis, :, :]
		value_batch = sess.run(self.value_tensor,
			feed_dict={self.state_input: state_batch})
		value = value_batch[0, 0]
		return value

	def sample_policy(self, state):
		action_pr = self.eval_policy(state)
		action = np.random.choice(self.action_indices, p=action_pr)
		return action
