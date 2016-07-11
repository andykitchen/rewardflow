from PyQt4 import QtCore, QtGui
from PyQt4 import QtOpenGL
import sys

sys.path.insert(0, '../upstream/Arcade-Learning-Environment')

import ale_python_interface

import numpy as np
import scipy.misc
import tensorflow as tf

import time
import random
import threading
import collections
import cv2

import deepq_loader

rom_path = 'space_invaders.bin'

ale_screen_height = 210
ale_screen_width = 160
ale_channels = 3


def np_rgb_to_qimage(x):
	qimage = QtGui.QImage(x.data, x.shape[1], x.shape[0], QtGui.QImage.Format_RGB888)
	return qimage


class AleGraphicsItem(QtGui.QGraphicsItem):
	def __init__(self):
		super(AleGraphicsItem, self).__init__()
		self.frame_image = None

	def boundingRect(self):
		return QtCore.QRectF(0, 0, ale_screen_width, ale_screen_height)

	def paint(self, painter, option, widget):
		if self.frame_image is None:
			painter.fillRect(0, 0, ale_screen_width, ale_screen_height, QtCore.Qt.gray)
		else:
			painter.drawImage(0, 0, self.frame_image)

	@QtCore.pyqtSlot()
	def show_frame(self, frame_image):
		self.update()
		self.frame_image = frame_image


class NeuralLayerGraphicsItem(QtGui.QGraphicsItem):
	def __init__(self, rows, cols, cell_size):
		super(NeuralLayerGraphicsItem, self).__init__()
		self.rows = rows
		self.cols = cols
		self.cell_size = cell_size

	def boundingRect(self):
		return QtCore.QRectF(0, 0, self.cols * self.cell_size, self.rows * self.cell_size)

	def paint(self, painter, option, widget):
		# painter.fillRect(0, 0, self.cols * self.cell_size, self.rows * self.cell_size, QtCore.Qt.gray)
		
		cell_size = self.cell_size
		for i in range(self.rows):
			for j in range(self.cols):
				qcolor = QtGui.QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
				painter.fillRect(j*cell_size, i*cell_size, cell_size, cell_size, qcolor)


class AleCompute(QtCore.QObject):
	frame = QtCore.pyqtSignal(QtGui.QImage)

	def __init__(self, rom_path):
		super(AleCompute, self).__init__()
		self.ale = ale_python_interface.ALEInterface()
		self.ale.loadROM(rom_path)
		self.actions = self.ale.getMinimalActionSet()

	@QtCore.pyqtSlot()
	def run(self):
		self.timer = QtCore.QTimer()
		self.timer.setSingleShot(True)
		self.timer.timeout.connect(self.step)

		history_size = 4
		screen_size = 84
		self.history = collections.deque(maxlen=history_size)
		for i in range(4):
			self.history.append(np.zeros((screen_size, screen_size)))

		self.elapsed_frames = 0
		self.last_action_index = 0

		self.sess = tf.Session()
		with self.sess.as_default():
			self.model = deepq_loader.AtariNIPSDeepQModel()
			self.saver = tf.train.Saver()
			self.sess.run(tf.initialize_all_variables())
			self.saver.restore(self.sess, 'pretrained_model.ckpt')

		self.step()

	@QtCore.pyqtSlot()
	def step(self):
		start = time.clock()

		if elapsed_frames % 4 == 0:
			with self.sess.as_default():
				state_action_values = self.model.eval(self.get_ale_state())
				action_index = np.argmax(state_action_values)
				self.last_action_index = action_index
		else:
			self.last_action_index = 0

		self.step_ale(action_index)

		finish = time.clock()
		elapsed = finish - start
		target = 1./60
		delay = target - elapsed if elapsed < target else 0
		self.timer.start(1000 * delay)

	def step_ale(self, action_index):
		self.ale.act(self.actions[action_index])
		# self.ale.act(random.choice(self.actions))
		frame = self.ale.getScreenRGB()
		qimage = np_rgb_to_qimage(frame).copy()
		self.frame.emit(qimage)

	def get_ale_state(self):
		frame       = self.ale.getScreenGrayscale()
		frame_small = scipy.misc.imresize(frame[:,:,0], (84, 84), interp='bilinear')
		# frame_small = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_LINEAR)

		frame_norm  = frame_small / 255.
		self.history.append(frame_norm)
		# return frame_norm[:,:,np.newaxis]
		# return np.tile(frame_norm[:,:,np.newaxis], (1, 1, 4))
		state = np.array(self.history).transpose(1, 2, 0)
		return state


def setup_ale_thread():
	thread = QtCore.QThread()
	compute = AleCompute(rom_path)
	ale_graphics_item = AleGraphicsItem()

	compute.moveToThread(thread)
	compute.frame.connect(ale_graphics_item.show_frame)
	thread.started.connect(compute.run)

	return thread, ale_graphics_item, compute


if __name__ == '__main__':
	app = QtGui.QApplication(sys.argv)
	scene = QtGui.QGraphicsScene()
	view = QtGui.QGraphicsView(scene)

	glwidget = QtOpenGL.QGLWidget(QtOpenGL.QGLFormat(QtOpenGL.QGL.SampleBuffers | QtOpenGL.QGL.DirectRendering))
	view.setViewport(glwidget)
	view.setViewportUpdateMode(QtGui.QGraphicsView.FullViewportUpdate)

	thread, ale_graphics_item, compute = setup_ale_thread()
	scene.addItem(ale_graphics_item)

	for n in range(5):
		nlgi = NeuralLayerGraphicsItem(16, 16, 10)
		scene.addItem(nlgi)
		nlgi.setPos(200 + n*180, 0)
		compute.frame.connect(nlgi.update)

	for n in range(10):
		agi = AleGraphicsItem()
		scene.addItem(agi)
		agi.setPos(n*ale_screen_width, 250)
		compute.frame.connect(agi.show_frame)

	view.show()
	thread.start()

	sys.exit(app.exec_())
