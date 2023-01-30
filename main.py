import sys
from PySide6 import QtWidgets
import numpy as np
import cv2
import os
from utils.image_utils import fit_image, train_model, get_data
from common.MainWindow import MainWindow

PIXELS_PER_PIXEL = 12
TRAINED_PIXELS = 28

if __name__ == "__main__":
    """Perform the training and the launching of the pyside6 application
    """
    data_path = os.path.join(os.path.dirname(__file__),"data")
    x_train, y_train = get_data(data_path)
    model = train_model(x_train, y_train)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(model, PIXELS_PER_PIXEL, TRAINED_PIXELS)
    window.show()
    app.exec()