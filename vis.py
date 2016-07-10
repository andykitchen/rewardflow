from PyQt4 import QtCore, QtGui
import sys

sys.path.insert(0, '../upstream/Arcade-Learning-Environment')

import ale_python_interface

import numpy as np
import tensorflow as tf

import time
import random
import threading

rom_path = 'space_invaders.bin'

ale_screen_height = 210
ale_screen_width = 160
ale_channels = 3


def np_rgb_to_qimage(x):
	qimage = QtGui.QImage(x.data, x.shape[1], x.shape[0], QtGui.QImage.Format_RGB888)
	return qimage

#frame = np.zeros((ale_screen_height, ale_screen_width, ale_channels))
#np.random.randn(ale_screen_height, ale_screen_width, ale_channels)


class AleGraphicsItem(QtGui.QGraphicsItem):
	def __init__(self):
		super(AleGraphicsItem, self).__init__()
		qimage = np_rgb_to_qimage(np.zeros((ale_screen_height, ale_screen_width, ale_channels)))
		self.frame_image = qimage

	def boundingRect(self):
		return QtCore.QRectF(0, 0, ale_screen_width, ale_screen_height)

	def paint(self, painter, option, widget):
		painter.drawImage(0, 0, self.frame_image)
	
	def show_frame(self, frame_image):
		self.update()
		self.frame_image = frame_image
	

class AleUpdateSignalHolder(QtCore.QObject):
	frame = QtCore.pyqtSignal(QtGui.QImage)

class AleThread(threading.Thread):
	def __init__(self, coord, rom_path):
		super(AleThread, self).__init__()
		self.coord = coord
		self.ale = ale_python_interface.ALEInterface()
		self.ale.loadROM(rom_path)
		self.actions = self.ale.getMinimalActionSet()
		self.signals = AleUpdateSignalHolder()

	def run(self):
		with coord.stop_on_exception():
			while not self.coord.should_stop():
				time.sleep(1./60)
				self.step()
	
	def step(self):
		self.ale.act(random.choice(self.actions))
		frame = self.ale.getScreenRGB()
		qimage = np_rgb_to_qimage(frame)
		self.signals.frame.emit(qimage)


def setup_ale_thread(coord):
	thread = AleThread(coord, rom_path)
	ale_graphics_item = AleGraphicsItem()
	thread.signals.frame.connect(ale_graphics_item.show_frame, type=QtCore.Qt.QueuedConnection)

	return thread, ale_graphics_item


def setup_view(ale_graphics_item):
	scene = QtGui.QGraphicsScene()
	view = QtGui.QGraphicsView(scene)

	scene.addItem(ale_graphics_item)

	# timer = QtCore.QTimer()
	# timer.timeout.connect(ale_graphics_item.step)

	return view, scene, ale_graphics_item


if __name__ == '__main__':
	app = QtGui.QApplication(sys.argv)

	coord = tf.train.Coordinator()

	thread, ale_graphics_item = setup_ale_thread(coord)
	view, scene, ale_graphics_item = setup_view(ale_graphics_item)

	thread.start()
	# timer.start()

	view.show()

	code = app.exec_()
	coord.request_stop()
	coord.join([thread])	
	sys.exit(code)
