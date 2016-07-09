import sys

sys.path.insert(0, '../upstream/Arcade-Learning-Environment')

import scipy.misc
import ale_python_interface

rom_path = 'space_invaders.bin'

import numpy as np
from numpy import newaxis
import tensorflow as tf

import threading

from policy_value import NatureAtariModel

class AleThread(threading.Thread):
	def __init__(self, sess, coord, rom_path, count_op, shared_params):
		super(AleThread, self).__init__()
		self.ale = ale_python_interface.ALEInterface()
		self.ale.loadROM(rom_path)
		self.minimal_actions = self.ale.getMinimalActionSet()

		self.sess = sess
		self.coord = coord
		self.count_op = count_op

		self.policy_value_approx = NatureAtariModel()
		self.copy_params_op = tf.group(*[tf.assign(local, shared) for local, shared in zip(self.policy_value_approx.params, shared_params)])

	def run(self):
		with coord.stop_on_exception():
			while not self.coord.should_stop():
				with self.sess.as_default():
					self.step()
	
	def get_state(self):
		frame = self.ale.getScreenGrayscale()
		frame_small = scipy.misc.imresize(frame[:,:,0], (83, 83), interp='bilinear')
		return frame_small

	def terminal_state(self):
		return self.ale.game_over()

	def start_new_episode(self):
		self.ale.reset_game()

	def sample_policy(self):
		state = self.get_state()
		action = self.policy_value_approx.sample(state)
		ale_action = self.minimal_actions[action]
		return ale_action

	def step(self):
		self.sess.run(self.copy_params_op)
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

shared_model = NatureAtariModel()

with tf.Session() as sess:
	coord = tf.train.Coordinator()
	threads = [AleThread(sess, coord, rom_path, count_op, shared_model.params) for i in range(3)]
	tf.initialize_all_variables().run()
	for t in threads:
		t.start()

	coord.join(threads)
