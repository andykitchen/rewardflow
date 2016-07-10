from PyQt4 import QtCore, QtGui
import sys

sys.path.insert(0, '../upstream/Arcade-Learning-Environment')

import ale_python_interface

import numpy as np
import random

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
	def __init__(self, rom):
		super(AleGraphicsItem, self).__init__()
		self.ale = ale_python_interface.ALEInterface()
		self.ale.loadROM(rom_path)
		self.actions = self.ale.getMinimalActionSet()
		frame = self.ale.getScreenRGB()
		self.set_frame(frame)

	def boundingRect(self):
		return QtCore.QRectF(0, 0, ale_screen_width, ale_screen_height)

	def paint(self, painter, option, widget):
		painter.drawImage(0, 0, self.frame_image)
	
	def set_frame(self, frame):
		self.update()
		self.frame = frame
		self.frame_image = np_rgb_to_qimage(frame)
	
	def step(self):
		self.ale.act(random.choice(self.actions))
		frame = self.ale.getScreenRGB()
		self.set_frame(frame)


class AleThread(threading.Thread):
    def __init__(self, sess, coord, rom_path, count_op, frame_tensor):
        super(AleThread, self).__init__()
        self.sess = sess
        self.coord = coord
        self.ale = ale_python_interface.ALEInterface()
        self.ale.loadROM(rom_path)
        self.count_op = count_op
        self.actions = self.ale.getMinimalActionSet()
        self.frame_placeholder = tf.placeholder(dtype=tf.uint8, shape=(210, 160, 3))
        self.assign_op = frame_tensor.assign(self.frame_placeholder)
    
    def run(self):
        with coord.stop_on_exception():
            while not self.coord.should_stop():
                time.sleep(1./60)
                self.step()
    
    def step(self):
        self.ale.act(random.choice(self.actions))
        frame = self.ale.getScreenRGB()
        self.sess.run([self.count_op, self.assign_op], feed_dict={self.frame_placeholder: frame})


def setup_view():
	scene = QtGui.QGraphicsScene()
	view = QtGui.QGraphicsView(scene)

	ale_graphics_item = AleGraphicsItem(rom_path)
	scene.addItem(ale_graphics_item)

	timer = QtCore.QTimer()
	timer.timeout.connect(ale_graphics_item.step)

	return view, scene, ale_graphics_item, timer


if __name__ == '__main__':
	app = QtGui.QApplication(sys.argv)

	view, scene, ale_graphics_item, timer = setup_view()

	timer.start()
	view.show()

	sys.exit(app.exec_())
