import sys

sys.path.insert(0, '../upstream/Arcade-Learning-Environment')

import scipy.misc
import ale_python_interface

rom_path = 'space_invaders.bin'

import numpy as np
from numpy import newaxis
import tensorflow as tf

import time

import threading

import policy_value
from policy_value import NatureAtariModel

def one_hot(x, n):
    v = np.zeros((len(x), n), dtype=np.float32)
    v[np.arange(v.shape[0]), x] = 1.
    return v

class Environment(object):
	pass

class StandardAleEnvironment(Environment):
	def __init__(self, rom_path, frame_skip=4, clip_rewards=True):
		super(Environment, self).__init__()
		self.ale = ale_python_interface.ALEInterface()
		self.ale.loadROM(rom_path)

		self.frame_skip = frame_skip
		self.clip_rewards = clip_rewards

		self.minimal_action_set = self.ale.getMinimalActionSet()
		self.total_reward = 0

	def get_state(self):
		frame = self.ale.getScreenGrayscale()
		frame_small = scipy.misc.imresize(frame[:,:,0], (83, 83), interp='bilinear')
		return frame_small

	def terminal_state(self):
		return self.ale.game_over()

	def start_new_episode(self):
		self.total_reward = 0
		self.ale.reset_game()

	def act(self, action):
		ale_action = self.minimal_action_set[action]

		for i in range(self.frame_skip):
			reward = self.ale.act(ale_action)

		if self.clip_rewards:
			return np.clip(reward, -1, 1)
		else:
			return reward


class TrainingThread(threading.Thread):
	def __init__(self, sess, coord, env, count_op, centralised_thingy):
		super(TrainingThread, self).__init__()
		self.sess = sess
		self.coord = coord
		self.env = env
		self.count_op = count_op

		self.smallT = 0

		self.centralised_thingy = centralised_thingy

		self.policy_value_approx = NatureAtariModel()
		self.copy_params_op = centralised_thingy.build_copy_op(self.policy_value_approx)
		self.update_op, self.actions_tensor, self.bigR_tensor = centralised_thingy.build_update_op(self.policy_value_approx)


	def run(self):
		with coord.stop_on_exception():
			while not self.coord.should_stop():
				with self.sess.as_default():
					self.step()
	
	def sample_policy(self):
		state = self.env.get_state()
		action = self.policy_value_approx.sample(state)
		return action

	def current_value(self):
		state = self.env.get_state()
		_, value = self.policy_value_approx.eval(state)
		return value

	def compute_value(self):
		state = self.env.get_state()
		action = self.policy_value_approx.sample(state)

	def synchronize_params(self):		
		self.copy_params_op.run()

	def discount_rewards(self, rewards, R_boostrap, gamma=0.9):
		Rs = []
		R = R_boostrap
		for r in reversed(rewards):
			R = r + gamma*R    
			Rs.append(R)
		return list(reversed(Rs))

	def step(self):
		t_max = 6
		t_start = self.smallT
		self.synchronize_params()

		states  = []
		actions = []
		rewards = []

		terminated = False
		while not terminated and self.smallT - t_start < t_max:
			states.append(self.get_state())

			action = self.sample_policy()
			reward = self.ale.act(action)

			actions.append(action)
			rewards.append(reward)

			self.total_reward += reward
			self.smallT += 1
			self.count_op.eval()

			terminated = self.terminal_state()

		if terminated:
			bigR_boostrap = 0
		else:
			bigR_boostrap = self.current_value()

		bigR = self.discount_rewards(rewards, bigR_boostrap)

		sess.run(self.update_op, feed_dict={
			self.policy_value_approx.state_input: np.array(states)[:,:,:,newaxis],
			self.actions_tensor: one_hot(actions, len(self.minimal_actions)),
			self.bigR_tensor: np.array(bigR)[:,newaxis]
		})

		if terminated:
			print "episode reward:", self.total_reward
			self.start_new_episode()



learning_rate = 7e-4

bigT = tf.Variable(0, dtype=tf.int64)
count_op = bigT.assign_add(1)

shared_model = NatureAtariModel()
shared_optimizer = tf.train.RMSPropOptimizer(learning_rate, epsilon=0.1, decay=0.99)
centralized_thingy = policy_value.CentralizedThingy(shared_optimizer, shared_model)

def build_training_thread(sess):
	env = StandardAleEnvironment(rom_path)
	return TrainingThread(sess, coord, env, count_op, centralized_thingy)

with tf.Session() as sess:
	coord = tf.train.Coordinator()
	threads = [build_training_thread(sess) for i in range(3)]
	tf.initialize_all_variables().run()
	for t in threads:
		t.start()

	coord.join(threads)
