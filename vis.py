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

def as_grid(x, r, c, pad=2, pad_value=0):
    x = np.lib.pad(x, ((pad,pad), (pad,pad), (0,0)), 'constant', constant_values=pad_value)
    ir, ic, n = x.shape
    x = x.reshape(ir, ic, r, c).transpose(2,0,3,1).reshape(r*ir, c*ic)
    x = np.lib.pad(x, pad, 'constant', constant_values=pad_value)
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


class ImageGraphicsItem(QtGui.QGraphicsItem):
	def __init__(self, width=0, height=0):
		super(ImageGraphicsItem, self).__init__()
		self.qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

	def boundingRect(self):
		return QtCore.QRectF(0, 0, self.qimage.width(), self.qimage.height())

	def paint(self, painter, option, widget):
		painter.drawImage(0, 0, self.qimage)

	def show_image(self, qimage):
		self.update()
		self.qimage = qimage


def activity_to_qimage(activity_array, rows, cols, scale, zoom):
	im = scale*activity_array
	im = np.clip(im, 0, 255)
	im = im.reshape(rows, cols)
	im = im.astype(np.uint8)
	im = zoom_int(im, zoom)
	im = to_rgb32(im)
	im = np_rgb32_to_qimage(im)
	return im

class GridRenderer(QtCore.QObject):
	output = QtCore.pyqtSignal(QtGui.QImage)

	def __init__(self, rows, cols, scale=127, zoom=10):
		super(GridRenderer, self).__init__()
		self.rows  = rows
		self.cols  = cols
		self.scale = scale
		self.zoom  = zoom

	def build_image(self, activity_array):
		return activity_to_qimage(activity_array,
			self.rows,
			self.cols,
			self.scale,
			self.zoom)

	def render(self, activity_array):
		qimage = self.build_image(activity_array)
		self.output.emit(qimage.copy())


def conv_activity_to_qimage(activity_array, rows, cols, pad, pad_value=127):
	im = 127*activity_array
	im = np.clip(im, 0, 255)
	im = im.astype(np.uint8)
	im = zoom_int(im, 4)
	im = as_grid(im, rows, cols, pad=pad, pad_value=pad_value)
	im_rgb = to_rgb32(im)
	qimage = np_rgb32_to_qimage(im_rgb)
	return qimage

class RenderConvActivitySignals(QtCore.QObject):
	image = QtCore.pyqtSignal(QtGui.QImage)

class RenderConvActivityRunnable(QtCore.QRunnable):
	def __init__(self, activity_array, rows, cols, pad):
		super(RenderConvActivityRunnable, self).__init__()
		self.rows = rows
		self.cols = cols
		self.pad = pad
		self.activity_array = activity_array
		self.signals = RenderConvActivitySignals()

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

	def show_image(self, activity_image):
		self.update()
		self.activity_image = activity_image

	def show_conv_activity(self, activity_array):
		runnable = RenderConvActivityRunnable(activity_array, self.rows, self.cols, self.pad)
		runnable.signals.image.connect(self.show_image)
		QtCore.QThreadPool.globalInstance().start(runnable)


class AleCompute(QtCore.QObject):
	frame = QtCore.pyqtSignal(QtGui.QImage)
	conv1_activity = QtCore.pyqtSignal(np.ndarray)
	conv2_activity = QtCore.pyqtSignal(np.ndarray)
	conv3_activity = QtCore.pyqtSignal(np.ndarray)
	final_layer_activity = QtCore.pyqtSignal(np.ndarray)

	def __init__(self, rom_path):
		super(AleCompute, self).__init__()
		self.ale = ale_python_interface.ALEInterface()
		self.ale.loadROM(rom_path)
		self.actions = self.ale.getMinimalActionSet()
		self.paused = False

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

	def restart(self):
		self.ale.reset_game()

	def togglePause(self):
		self.paused = not self.paused
		if self.paused == False:
			self.step()

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

		if not self.paused:
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
		state = np.array(self.history).transpose(1, 2, 0)
		return state

class MainView(QtGui.QGraphicsView):
	restart = QtCore.pyqtSignal()
	togglePause = QtCore.pyqtSignal()
	stepOne = QtCore.pyqtSignal()

	def keyPressEvent(self, event):
		if event.text() == 'r':
			self.restart.emit()
		if event.text() == ' ':
			self.togglePause.emit()
		if event.text() == ']':
			self.stepOne.emit()
		super(MainView, self).keyPressEvent(event)

def setup_ale_thread():
	thread  = QtCore.QThread()
	compute = AleCompute(rom_path)

	compute.moveToThread(thread)

	ale_gi = ImageGraphicsItem(ale_screen_width, ale_screen_height)
	nl_gi  = ImageGraphicsItem(160, 320)

	cl_gi1 = ConvLayerGraphicsItem(4, 8,  20, 20, pad=2)
	cl_gi2 = ConvLayerGraphicsItem(4, 16, 9, 9,   pad=2)
	cl_gi3 = ConvLayerGraphicsItem(4, 16, 7, 7,   pad=2)

	grid_renderer = GridRenderer(32, 16, zoom=10)

	grid_renderer.output.connect(nl_gi.show_image)
	compute.final_layer_activity.connect(grid_renderer.render)

	compute.frame.connect(ale_gi.show_image)
	compute.conv1_activity.connect(cl_gi1.show_conv_activity)
	compute.conv2_activity.connect(cl_gi2.show_conv_activity)
	compute.conv3_activity.connect(cl_gi3.show_conv_activity)
	thread.started.connect(compute.run)

	return thread, compute, ale_gi, nl_gi, cl_gi1, cl_gi2, cl_gi3, grid_renderer


if __name__ == '__main__':
	app = QtGui.QApplication(sys.argv)
	scene = QtGui.QGraphicsScene()
	view = MainView(scene)

	glwidget = QtOpenGL.QGLWidget(QtOpenGL.QGLFormat(QtOpenGL.QGL.SampleBuffers | QtOpenGL.QGL.DirectRendering))
	view.setViewport(glwidget)

	thread, compute, ale_gi, nl_gi, cl_gi1, cl_gi2, cl_gi3, grid_renderer = setup_ale_thread()

	view.restart.connect(compute.restart)
	view.togglePause.connect(compute.togglePause)
	view.stepOne.connect(compute.step)

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

	sys.exit(app.exec_())
