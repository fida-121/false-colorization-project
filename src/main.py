import sys
import os
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QComboBox, QSlider,
    QProgressBar, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage


class ImageColorizerApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Advanced Image Colorizer")
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        # Labels
        self.original_label = QLabel("Original Image")
        self.original_label.setAlignment(Qt.AlignCenter)

        self.colored_label = QLabel("Colored Image")
        self.colored_label.setAlignment(Qt.AlignCenter)

        # Colormap dropdown
        self.colormap_box = QComboBox()
        self.colormap_box.addItems([
            "Jet", "Parula", "Hot", "Cool", "Spring", "Summer",
            "Autumn", "Winter", "Bone", "HSV", "Pink", "Gray"
        ])

        # Slider
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(0, 100)
        self.intensity_slider.setValue(100)

        # Buttons
        self.load_button = QPushButton("Load Image")
        self.realistic_button = QPushButton("Realistic Colorization")
        self.apply_colormap_button = QPushButton("Apply Colormap")
        self.save_button = QPushButton("Save Image")

        self.progress_bar = QProgressBar()

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.realistic_button)
        layout.addWidget(self.colormap_box)
        layout.addWidget(QLabel("Intensity"))
        layout.addWidget(self.intensity_slider)
        layout.addWidget(self.apply_colormap_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.progress_bar)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.original_label)
        image_layout.addWidget(self.colored_label)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(image_layout)

        self.setLayout(main_layout)

        # Connections
        self.load_button.clicked.connect(self.load_image)
        self.realistic_button.clicked.connect(self.colorize_realistically)
        self.apply_colormap_button.clicked.connect(self.apply_colormap)
        self.save_button.clicked.connect(self.save_image)

        self.image = None
        self.colored_image = None

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_name:
            self.image = cv2.imread(file_name)
            self.display_image(self.image, self.original_label)
            self.progress_bar.setValue(0)
            self.colored_image = None

    def display_image(self, image, label):
        height, width, channel = image.shape
        bytes_per_line = 3 * width

        q_img = QImage(
            image.data, width, height,
            bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()

        pixmap = QPixmap.fromImage(q_img).scaled(
            400, 400, Qt.KeepAspectRatio
        )

        label.setPixmap(pixmap)

    def colorize_realistically(self):
        if self.image is None:
            return

        self.progress_bar.setValue(10)

        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, self.image)

        self.colored_image = self.colorize_image_realistically(temp_path)
        self.display_image(self.colored_image, self.colored_label)

        self.progress_bar.setValue(100)

    def colorize_image_realistically(self, image_path):
        DIR = "models"   # ✅ FIXED (important)

        proto = os.path.join(DIR, "colorization_deploy_v2.prototxt")
        model = os.path.join(DIR, "colorization_release_v2.caffemodel")
        points = os.path.join(DIR, "pts_in_hull.npy")

        net = cv2.dnn.readNetFromCaffe(proto, model)

        pts = np.load(points)
        pts = pts.transpose().reshape(2, 313, 1, 1)

        net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
        net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
            np.full([1, 313], 2.606, dtype="float32")
        ]

        image = cv2.imread(image_path)
        scaled = image.astype("float32") / 255.0

        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        resized = cv2.resize(lab, (224, 224))

        L = cv2.split(resized)[0]
        L -= 50

        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0].transpose((1, 2, 0))

        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
        L = cv2.split(lab)[0]

        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)

        colorized = np.clip(colorized, 0, 1)

        return (255 * colorized).astype("uint8")

    def apply_colormap(self):
        if self.image is None:
            return

        self.progress_bar.setValue(20)

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        selected_map = self.colormap_box.currentText().lower()
        intensity = self.intensity_slider.value() / 100.0

        colormap_dict = {
            "jet": cv2.COLORMAP_JET,
            "parula": cv2.COLORMAP_PARULA,
            "hot": cv2.COLORMAP_HOT,
            "cool": cv2.COLORMAP_COOL,
            "spring": cv2.COLORMAP_SPRING,
            "summer": cv2.COLORMAP_SUMMER,
            "autumn": cv2.COLORMAP_AUTUMN,
            "winter": cv2.COLORMAP_WINTER,
            "bone": cv2.COLORMAP_BONE,
            "hsv": cv2.COLORMAP_HSV,
            "pink": cv2.COLORMAP_PINK,
            "gray": -1
        }

        cmap = colormap_dict.get(selected_map, cv2.COLORMAP_JET)

        if cmap == -1:
            colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            colored = cv2.applyColorMap(gray, cmap)

        blended = cv2.addWeighted(
            colored, intensity,
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            1 - intensity, 0
        )

        self.colored_image = blended
        self.display_image(blended, self.colored_label)

        self.progress_bar.setValue(100)

    def save_image(self):
        if self.colored_image is None:
            QMessageBox.warning(self, "Warning", "No colored image to save!")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_name:
            save_image = cv2.cvtColor(self.colored_image, cv2.COLOR_BGR2RGB)
            success = cv2.imwrite(file_name, save_image)

            if not success:
                QMessageBox.critical(self, "Error", "Failed to save image!")
            else:
                QMessageBox.information(self, "Success", "Image saved successfully!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageColorizerApp()
    window.show()
    sys.exit(app.exec_())