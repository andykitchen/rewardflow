import sys

sys.path.insert(0, '../upstream/Arcade-Learning-Environment')

import scipy.misc
import ale_python_interface

import numpy as np
from numpy import newaxis
import tensorflow as tf

import time

import threading
import collections

import policy_value
from policy_value import AtariNatureModel


def one_hot(x, n):
    v = np.zeros((len(x), n), dtype=np.float32)
    v[np.arange(v.shape[0]), x] = 1.
    return v


class AleEnvironment(object):
	def __init__(self, rom_path, frame_skip=4, clip_rewards=True, history_size=4):
		super(AleEnvironment, self).__init__()
		self.ale = ale_python_interface.ALEInterface()
		self.ale.loadROM(rom_path)

		self.frame_skip = frame_skip
		self.clip_rewards = clip_rewards

		self.minimal_action_set = self.ale.getMinimalActionSet()
		self.n_actions = len(self.minimal_action_set)
		self.episode_reward = 0
		self.frames = 0


		self.history = collections.deque(maxlen=history_size)
		self.act(0) # Is this a good way to initialise the history?

	def get_ale_state(self):
		frame       = self.ale.getScreenGrayscale()
		frame_small = scipy.misc.imresize(frame[:,:,0], (83, 83), interp='bilinear')
		frame_norm  = frame_small / 255.
		return frame_norm

	def push_history(self):
		self.history.append(self.get_ale_state())

	def get_state(self):
		return np.array(self.history).transpose(1, 2, 0)

	def is_terminal_state(self):
		return self.ale.game_over()

	def start_new_episode(self):
		self.episode_reward = 0
		self.ale.reset_game()

	def act(self, action):
		reward = 0
		ale_action = self.minimal_action_set[action]

		for i in range(self.frame_skip):
			reward += self.ale.act(ale_action)
			self.frames += 1
			self.push_history()

		self.episode_reward += reward

		if self.clip_rewards:
			return np.clip(reward, -1., 1.)
		else:
			return reward


class TrainingThread(threading.Thread):
	def __init__(self, sess, coord, env, policy_value_approx, count_op, centralised_thingy):
		super(TrainingThread, self).__init__()
		self.sess = sess
		self.coord = coord
		self.env = env
		self.policy_value_approx = policy_value_approx
		self.count_op = count_op
		self.centralised_thingy = centralised_thingy

		self.smallT = 0

		self.copy_params_op = centralised_thingy.build_copy_op(self.policy_value_approx)
		self.update_op, self.actions_tensor, self.bigR_tensor = \
			centralised_thingy.build_update_op(self.policy_value_approx)

	def run(self):
		with coord.stop_on_exception():
			while not self.coord.should_stop():
				with self.sess.as_default():
					self.step()
	
	def sample_policy(self, state):
		action = self.policy_value_approx.sample_policy(state)
		return action

	def current_estimated_value(self):
		state = self.env.get_state()
		value = self.policy_value_approx.eval_value(state)
		return value

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
			state  = self.env.get_state()
			action = self.sample_policy(state)
			reward = self.env.act(action)

			states.append(state)
			actions.append(action)
			rewards.append(reward)

			self.smallT += 1
			self.count_op.eval()

			terminated = self.env.is_terminal_state()

		if terminated:
			bigR_boostrap = 0
		else:
			bigR_boostrap = self.current_estimated_value()

		bigR = self.discount_rewards(rewards, bigR_boostrap)

		sess.run(self.update_op, feed_dict={
			self.policy_value_approx.state_input: np.array(states),
			self.actions_tensor: one_hot(actions, self.env.n_actions),
			self.bigR_tensor: np.array(bigR)
		})

		if terminated:
			print "episode reward:", self.env.episode_reward
			self.env.start_new_episode()


if __name__ == "__main__":
	rom_path = 'space_invaders.bin'
	learning_rate = 7e-4

	bigT = tf.Variable(0, dtype=tf.int64)
	count_op = bigT.assign_add(1)

	shared_model = AtariNatureModel()
	shared_optimizer = tf.train.RMSPropOptimizer(learning_rate, epsilon=0.1, decay=0.99)
	centralized_thingy = policy_value.CentralizedThingy(shared_optimizer, shared_model)

	def build_training_thread(sess):
		env = AleEnvironment(rom_path)
		policy_value_approx = AtariNatureModel()
		return TrainingThread(sess, coord, env, policy_value_approx, count_op, centralized_thingy)

	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		threads = [build_training_thread(sess) for i in range(3)]
		tf.initialize_all_variables().run()
		for t in threads:
			t.start()

		coord.join(threads)
