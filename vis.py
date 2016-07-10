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


class AleCompute(QtCore.QObject):
	frame = QtCore.pyqtSignal(QtGui.QImage)

	def __init__(self, rom_path):
		super(AleCompute, self).__init__()
		self.ale = ale_python_interface.ALEInterface()
		self.ale.loadROM(rom_path)
		self.actions = self.ale.getMinimalActionSet()
		self.timer = QtCore.QTimer()
		self.timer.setSingleShot(True)
		self.timer.timeout.connect(self.step)

	@QtCore.pyqtSlot()
	def run(self):
		self.timer = QtCore.QTimer()
		self.timer.setSingleShot(True)
		self.timer.timeout.connect(self.step)
		self.step()

	@QtCore.pyqtSlot()
	def step(self):
		start = time.clock()
		self.step_ale()
		finish = time.clock()
		elapsed = finish - start
		target = 1./60
		delay = target - elapsed if elapsed < target else 0
		self.timer.start(1000 * delay)

	def step_ale(self):
		self.ale.act(random.choice(self.actions))
		frame = self.ale.getScreenRGB()
		qimage = np_rgb_to_qimage(frame).copy()
		self.frame.emit(qimage)


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

	thread, ale_graphics_item, compute = setup_ale_thread()
	scene.addItem(ale_graphics_item)

	view.show()
	thread.start()

	sys.exit(app.exec_())
