from PyQt4 import QtCore, QtGui
from PyQt4 import QtOpenGL
import sys

sys.path.insert(0, '../upstream/Arcade-Learning-Environment')

import ale_python_interface

import numpy as np
import scipy.misc
import scipy.ndimage
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
	qimage = QtGui.QImage(x.astype(np.uint8).data, x.shape[1], x.shape[0], QtGui.QImage.Format_RGB888)
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
	def __init__(self, rows, cols, cell_size, scale=127):
		super(NeuralLayerGraphicsItem, self).__init__()
		self.rows = rows
		self.cols = cols
		self.cell_size = cell_size
		self.scale = scale
		self.activity_array = np.zeros((self.rows, self.cols))

	def boundingRect(self):
		return QtCore.QRectF(0, 0, self.cols * self.cell_size, self.rows * self.cell_size)

	def paint(self, painter, option, widget):
		painter.fillRect(0, 0, self.cols * self.cell_size, self.rows * self.cell_size, QtCore.Qt.gray)
		
		# cell_size = self.cell_size
		# for i in range(self.rows):
		# 	for j in range(self.cols):
		# 		z = np.clip(self.scale*self.activity_array[i, j], 0, 255)
		# 		r = z; b = z; g = z
		# 		qcolor = QtGui.QColor(r, g, b)
		# 		painter.fillRect(j*cell_size, i*cell_size, cell_size, cell_size, qcolor)

	@QtCore.pyqtSlot()
	def show_activity(self, activity_array):
		self.update()
		self.activity_array = activity_array.reshape(self.rows, self.cols)


def as_grid(x, r, c, pad=2, pad_value=0):
    x = np.lib.pad(x, ((pad,pad), (pad,pad), (0,0)), 'constant', constant_values=pad_value)
    ir, ic, n = x.shape
    x = x.reshape(ir, ic, r, c).transpose(2,0,3,1).reshape(r*ir, c*ic)
    # x = np.lib.pad(x, pad, 'constant', constant_values=pad_value)
    return x

def to_rgb(x):
	return np.tile(x[:,:,np.newaxis],(1,1,3))

def to_rgb32(x):
	return np.tile(x[:,:,np.newaxis],(1,1,4))

def np_rgb32_to_qimage(x):
	qimage = QtGui.QImage(x.astype(np.uint8).data, x.shape[1], x.shape[0], QtGui.QImage.Format_RGB32)
	return qimage

def zoom_int(x, n):
	x = np.repeat(x, n, axis=0)
	x = np.repeat(x, n, axis=1)
	return x


def conv_activity_to_qimage(activity_array, rows, cols, pad, pad_value=127):
	im = 127*activity_array
	im = im.astype(np.uint8)
	im = zoom_int(im, 4)
	im = as_grid(im, rows, cols, pad=pad, pad_value=pad_value)
	# im = scipy.misc.imresize(im, 400, interp='nearest')
	im_rgb = to_rgb32(im)
	qimage = np_rgb32_to_qimage(im_rgb)
	return qimage


class RenderConvActivitySignale(QtCore.QObject):
	image = QtCore.pyqtSignal(QtGui.QImage)

class RenderConvActivityRunnable(QtCore.QRunnable):
	def __init__(self, activity_array, rows, cols, pad):
		super(RenderConvActivityRunnable, self).__init__()
		self.rows = rows
		self.cols = cols
		self.pad = pad
		self.activity_array = activity_array
		self.signals = RenderConvActivitySignale()

	def run(self):
		activity_image = conv_activity_to_qimage(
			self.activity_array,
			self.rows, self.cols, self.pad)
		self.signals.image.emit(activity_image.copy())


class ConvLayerGraphicsItem(QtGui.QGraphicsItem):
	def __init__(self, rows, cols, cell_rows, cell_cols, pad=2):
		super(ConvLayerGraphicsItem, self).__init__()
		self.rows = rows
		self.cols = cols
		self.pad = pad
		self.cell_rows = cell_rows
		self.cell_cols = cell_cols
		no_activity = np.zeros((cell_rows, cell_cols, rows*cols))
		self.activity_image = conv_activity_to_qimage(
			no_activity,
			rows, cols, pad)

	def boundingRect(self):
		return QtCore.QRectF(0, 0, self.activity_image.width(), self.activity_image.height())

	def paint(self, painter, option, widget):
		painter.drawImage(0, 0, self.activity_image)

	@QtCore.pyqtSlot()
	def show_image(self, activity_image):
		self.update()
		self.activity_image = activity_image

	@QtCore.pyqtSlot()
	def show_conv_activity(self, activity_array):
		runnable = RenderConvActivityRunnable(activity_array, self.rows, self.cols, self.pad)
		runnable.signals.image.connect(self.show_image)
		QtCore.QThreadPool.globalInstance().start(runnable)


class AleCompute(QtCore.QObject):
	frame = QtCore.pyqtSignal(QtGui.QImage)
	final_layer_activity = QtCore.pyqtSignal(np.ndarray)
	conv1_activity = QtCore.pyqtSignal(np.ndarray)
	conv2_activity = QtCore.pyqtSignal(np.ndarray)
	conv3_activity = QtCore.pyqtSignal(np.ndarray)

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
		self.history = collections.deque(maxlen=history_size)
		for i in range(4):
			self.history.append(np.zeros((84, 84)))

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

		with self.sess.as_default():
			activity = self.model.eval_activity(self.get_ale_state())
			state_action_values = activity[-1]
			action_index = np.argmax(state_action_values)
		self.step_ale(action_index)

		self.conv1_activity.emit(activity[0])
		self.conv2_activity.emit(activity[1])
		self.conv3_activity.emit(activity[2])
		self.final_layer_activity.emit(activity[-2])

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
		# frame_small = scipy.misc.imresize(frame[:,:,0], (84, 84), interp='bilinear', mode='F')
		frame_small = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_LINEAR)

		frame_norm  = frame_small / 255.
		self.history.append(frame_norm)
		# return frame_norm[:,:,np.newaxis]
		# return np.tile(frame_norm[:,:,np.newaxis], (1, 1, 4))
		state = np.array(self.history).transpose(1, 2, 0)
		return state


def setup_ale_thread():
	thread = QtCore.QThread()
	compute = AleCompute(rom_path)
	ale_gi = AleGraphicsItem()
	nl_gi = NeuralLayerGraphicsItem(32, 16, 10)
	cl_gi1 = ConvLayerGraphicsItem(4, 8, 20, 20, pad=2)
	cl_gi2 = ConvLayerGraphicsItem(4, 16, 9, 9, pad=2)
	cl_gi3 = ConvLayerGraphicsItem(4, 16, 7, 7, pad=2)

	compute.moveToThread(thread)
	compute.frame.connect(ale_gi.show_frame)
	compute.final_layer_activity.connect(nl_gi.show_activity)
	compute.conv1_activity.connect(cl_gi1.show_conv_activity)
	compute.conv2_activity.connect(cl_gi2.show_conv_activity)
	compute.conv3_activity.connect(cl_gi3.show_conv_activity)
	thread.started.connect(compute.run)

	return thread, compute, ale_gi, nl_gi, cl_gi1, cl_gi2, cl_gi3


if __name__ == '__main__':
	app = QtGui.QApplication(sys.argv)
	scene = QtGui.QGraphicsScene()
	view = QtGui.QGraphicsView(scene)

	glwidget = QtOpenGL.QGLWidget(QtOpenGL.QGLFormat(QtOpenGL.QGL.SampleBuffers | QtOpenGL.QGL.DirectRendering))
	view.setViewport(glwidget)
	# view.setViewportUpdateMode(QtGui.QGraphicsView.FullViewportUpdate)
	# view.setViewportUpdateMode(QtGui.QGraphicsView.NoViewportUpdate)



	thread, compute, ale_gi, nl_gi, cl_gi1, cl_gi2, cl_gi3 = setup_ale_thread()
	scene.addItem(ale_gi)

	scene.addItem(nl_gi)
	nl_gi.setPos(0, 250)

	scene.addItem(cl_gi1)
	cl_gi1.setPos(200, 0)

	scene.addItem(cl_gi2)
	cl_gi2.setPos(200, 350)

	scene.addItem(cl_gi3)
	cl_gi3.setPos(200, 520)

	view.show()
	thread.start()

	code = app.exec_()
	sys.exit()
