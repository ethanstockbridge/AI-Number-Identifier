from common.TensorResultWindow import TensorResultWindow
from common.Canvas import Canvas
from PySide6 import QtWidgets
from PySide6.QtCore import Qt

class MainWindow(QtWidgets.QMainWindow):
    """Main window that encompasses the drawing canvas and results

    Args:
        QtWidgets (QtWidgets.QMainWindow): Parent
    """
    def __init__(self, model, PIXELS_PER_PIXEL, TRAINED_PIXELS):
        """Call super and set up the main window with the canvas and qwidget results

        Args:
            model (tensorflow model): Trained model to be passed to our result manager
        """
        super().__init__()

        self.display_result = TensorResultWindow(model, PIXELS_PER_PIXEL)
        self.canvas = Canvas(self.display_result, PIXELS_PER_PIXEL, TRAINED_PIXELS)

        w = QtWidgets.QWidget()

        hl = QtWidgets.QHBoxLayout()

        vl = QtWidgets.QVBoxLayout()
        vl.addWidget(self.canvas)
        self.clear_button = QtWidgets.QPushButton()
        self.clear_button.setText("Clear canvas")
        self.clear_button.pressed.connect(self.clear_action)
        vl.addWidget(self.clear_button)
        hl.addLayout(vl)

        hl.addWidget(self.display_result)

        w.setLayout(hl)

        self.setCentralWidget(w)

    def clear_action(self):
        """Clear the canvas when the clear button is pressed
        """
        self.canvas.clearCanvas()
    
    def keyPressEvent(self, event):
        """Handle keypresses, exit when esc

        Args:
            event (Qt.KeyPressEvent): Keypress event
        """
        if event.key() == Qt.Key_Escape:
            self.close()
        if event.key() == Qt.Key_Undo:
            self.canvas.undo()