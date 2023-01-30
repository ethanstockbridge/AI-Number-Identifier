from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt
import numpy as np
import cv2

class Canvas(QtWidgets.QLabel):
    """Creates a canvas for the user to draw on

    Args:
        QtWidgets (QtWidgets.QLabel): Parent
    """

    def __init__(self, display_result, PIXELS_PER_PIXEL, trained_pixels):
        """Call super and set up the canvas

        Args:
            display_result (QtWidgets.QtWidget): Reference to the qtwidget incharge
            of changing the results
        """
        super().__init__()
        self.m_width = 28*PIXELS_PER_PIXEL
        self.m_height = 28*PIXELS_PER_PIXEL
        self.ppp = PIXELS_PER_PIXEL
        self.trained_pixels = trained_pixels
        pixmap = QtGui.QPixmap(self.m_width, self.m_height)
        pixmap.fill(Qt.white)
        self.setPixmap(pixmap)
        self.display_result=display_result

        self.last_x, self.last_y = None, None

    def mouseMoveEvent(self, e):
        """Edit the canvas when the mouse is moved (and clicked). This is scaled such that
        the pixels appear to be a TRAINED_PIXELS x TRAINED_PIXELS image

        Args:
            e (QMouseEvent): Mouse movement
        """
        canvas = self.pixmap()
        painter = QtGui.QPainter(canvas)

        low_x = (e.x()//self.ppp)*self.ppp
        high_x = ((e.x()//self.ppp)+1)*self.ppp
        low_y = (e.y()//self.ppp)*self.ppp
        high_y = ((e.y()//self.ppp)+1)*self.ppp
        painter.setPen(QtGui.QPen(QtGui.QColor(0,0,0,255), 1, Qt.SolidLine))

        for px_x in range(low_x,high_x):
            for px_y in range(low_y,high_y):
                painter.drawPoint(px_x,px_y)

        painter.end()
        self.setPixmap(canvas)

    def mouseReleaseEvent(self, e):
        """When the mouse is released, request the model to predict what has been drawn

        Args:
            e (QMouseEvent): Mouse movement
        """
        self.previous_pixmap = self.pixmap()
        canvas = self.pixmap()
        qimg = canvas.toImage()
        byte_str = qimg.bits().tobytes()
        img = np.frombuffer(byte_str, dtype=np.uint8).reshape((self.m_width,self.m_height,4))
        img = cv2.resize(img, dsize=(self.trained_pixels,self.trained_pixels), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img/255
        img = abs(img-1)
        self.display_result.update_result(img)
        pass

    def clearCanvas(self):
        """Clear/reset the canvas when requested by the pushbutton
        """
        pixmap = self.pixmap()
        pixmap.fill(Qt.white)
        self.setPixmap(pixmap)

    def undo(self):
        """Undo the previous pixmap and revert to a previous state
        """
        self.setPixmap(self.previous_pixmap)
