from PySide6 import QtWidgets
from utils.image_utils import fit_image
import numpy as np

class TensorResultWindow(QtWidgets.QWidget):
    """QWidget that shows the resulting tensorflow output and statistics

    Args:
        QtWidgets (QtWidgets.QWidget): parent
    """
    def __init__(self, model, PIXELS_PER_PIXEL):
        """Set up super and create basic layout for the display result which
        includes the output text and stats

        Args:
            model (_type_): _description_
        """
        super().__init__()
        self.model = model
        self.setMinimumSize(28*PIXELS_PER_PIXEL,28*PIXELS_PER_PIXEL)
        
        self.result_label = QtWidgets.QLabel()
        f = self.result_label.font()
        f.setPointSize(10)
        self.result_label.setFont(f)
        self.result_label.setText("Tensorflow calculated result:")

        self.real_result = QtWidgets.QLabel()
        f = self.real_result.font()
        f.setPointSize(10)
        self.real_result.setFont(f)

        self.numeric_resuilt = QtWidgets.QLabel()
        f = self.numeric_resuilt.font()
        f.setPointSize(80)
        self.numeric_resuilt.setFont(f)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.result_label)
        lay.addWidget(self.numeric_resuilt)
        lay.addWidget(self.real_result)

    def update_result(self, img):
        """Updates the result from the given image

        Args:
            img (numpy.ndarray): Input image
        """
        img = fit_image(img)
        img = np.array([img])
        predictions = self.model.predict(img, verbose=0)[0]
        results = ""
        for i, percent in enumerate(predictions):
            results += f"{i}: {percent}"
            results += "\n"
        self.real_result.setText(results)
        self.numeric_resuilt.setText(str(np.argmax(predictions)))
