import sys
import cv2
import numpy as np
import filtering
import noise
import morphology
import edge_gradient

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage    

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("IP-2")
        self.setGeometry(100, 100, 500, 500)

        self.image = None
        self.image_label = QLabel(self)

        # Submenu in File
        # Open image file
        open_file_act = QAction("Open file", self)
        open_file_act.setShortcut("Ctrl+O")
        open_file_act.setStatusTip("Open file from directory")
        open_file_act.triggered.connect(self.open_file)

        # Save image file
        save_file_act = QAction("Save file", self)
        save_file_act.setShortcut("Ctrl+S")
        save_file_act.setStatusTip("Save file to drive")
        save_file_act.triggered.connect(self.save_file)
        
        # Exit app
        exit_act = QAction("Exit", self)
        exit_act.setShortcut("Alt+F4")
        exit_act.setStatusTip("Leave the App")
        exit_act.triggered.connect(self.close_app)
        
        self.statusBar()

        # Filtering menu
        convolutional_2d_act = QAction("2D Convolution", self)
        convolutional_2d_act.triggered.connect(self.convolutional_2d)
        
        averaging_act = QAction("Averaging Blurr", self)
        averaging_act.triggered.connect(self.averaging)

        gaussian_blurr_act = QAction("Gaussian Blurr", self)
        gaussian_blurr_act.triggered.connect(self.gaussian)

        median_blurr_act = QAction("Median Blurr", self)
        median_blurr_act.triggered.connect(self.median)

        bilateral_filtering = QAction("Bilateral Filtering", self)
        bilateral_filtering.triggered.connect(self.bilateral)

        # Noise Menu
        gauss_noise_act = QAction("Gaussian Noise", self)
        gauss_noise_act.triggered.connect(self.gaussian_noise)

        salt_and_pepper_act = QAction("Salt and Pepper Noise", self)
        salt_and_pepper_act.triggered.connect(self.salt_and_pepper_noise)
        
        speckle_act = QAction("Speckle Noise", self)
        speckle_act.triggered.connect(self.speckle_noise)

        poisson_act = QAction("Poisson Noise", self)
        poisson_act.triggered.connect(self.poisson_noise)

        # Morphology Menu
        erosion_act = QAction("Erosion", self)
        erosion_act.triggered.connect(self.erosion)

        dilation_act = QAction("Dilation", self)
        dilation_act.triggered.connect(self.dilation)
        
        opening_act = QAction("Opening", self)
        opening_act.triggered.connect(lambda: self.transform(cv2.MORPH_OPEN))
        
        closing_act = QAction("Closing", self)
        closing_act.triggered.connect(lambda: self.transform(cv2.MORPH_CLOSE))
        
        gradien_act = QAction("Gradient", self)
        gradien_act.triggered.connect(lambda: self.transform(cv2.MORPH_GRADIENT))
        
        tophat_act = QAction("Top Hat", self)
        tophat_act.triggered.connect(lambda: self.transform(cv2.MORPH_TOPHAT))
        
        blackhat_act = QAction("Black Hat", self)
        blackhat_act.triggered.connect(lambda: self.transform(cv2.MORPH_BLACKHAT))

        # Edge menu
        laplacian_act = QAction("Laplacian", self)
        laplacian_act.triggered.connect(self.laplacian)

        sobel_act = QAction("Sobel", self)
        sobel_act.triggered.connect(self.sobel)

        canny_act = QAction("Canny", self)
        canny_act.triggered.connect(self.canny)
        
        prewitt_act = QAction("Prewitt", self)
        prewitt_act.triggered.connect(self.prewitt)

        robert_act = QAction("Robert", self)
        robert_act.triggered.connect(self.robert)

        # Adding action to menu bar
        main_menu = self.menuBar()

        file_menu = main_menu.addMenu('File')
        file_menu.addAction(open_file_act)
        file_menu.addAction(save_file_act)
        file_menu.addAction(exit_act)

        filtering_menu = main_menu.addMenu('Filtering')
        filtering_menu.addAction(convolutional_2d_act)
        filtering_menu.addAction(averaging_act)
        filtering_menu.addAction(gaussian_blurr_act)
        filtering_menu.addAction(median_blurr_act)
        filtering_menu.addAction(bilateral_filtering)

        noise_menu = main_menu.addMenu('Noise')
        noise_menu.addAction(gauss_noise_act)
        noise_menu.addAction(salt_and_pepper_act)
        noise_menu.addAction(speckle_act)
        noise_menu.addAction(poisson_act)
        
        morphology_menu = main_menu.addMenu('Morphology')
        morphology_menu.addAction(erosion_act)
        morphology_menu.addAction(dilation_act)
        morphology_menu.addAction(opening_act)
        morphology_menu.addAction(closing_act)
        morphology_menu.addAction(gradien_act)
        morphology_menu.addAction(tophat_act)
        morphology_menu.addAction(blackhat_act)

        edge_menu = main_menu.addMenu('Edge Detection')
        edge_menu.addAction(laplacian_act)
        edge_menu.addAction(sobel_act)
        edge_menu.addAction(canny_act)
        edge_menu.addAction(prewitt_act)
        edge_menu.addAction(robert_act)

    def open_file(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file')
        self.image = cv2.imread(fname[0])

        # convert to qpixmap
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w, ch = self.image.shape
        bpl = ch * w
        q_image = QImage(rgb, w, h, bpl, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.width(), pixmap.height())
        self.resize(pixmap.width(), pixmap.height())

    def save_file(self):
        if self.image.data:
            fname = QFileDialog.getSaveFileName(self, 'Save file')
            print(fname[0])
            cv2.imwrite(fname[0], self.image)
        else:
            # Open dialog load image first
            dialog = QDialog(self)
            dialog.setWindowTitle("Warning!")
            dialog.exec_()

    def convolutional_2d(self):
        kernel, ok_kernel = QInputDialog().getInt(self, "Kernel Size", "Kernel size:", min=1, step=2)
        depth, ok_depth = QInputDialog().getInt(self, "Depth", "Depth:", min=-1, max=1)

        print(kernel, depth)
        if ok_kernel and ok_depth:
            filtering.convolutional(self.image, kernel, depth)

    def averaging(self):
        kernel, ok_kernel = QInputDialog().getInt(self, "Kernel Size", "Kernel Size:", min=1, step=2)
        if ok_kernel:
            filtering.averaging(self.image, kernel)

    def gaussian(self):
        kernel, ok_kernel = QInputDialog().getInt(self, "Kernel Size", "Kernel Size", min=1, step=2)
        sigmaX, ok_sigma = QInputDialog().getDouble(self, "Sigma X", "Sigma X:", min=0)
        if ok_kernel and ok_sigma:
            filtering.gaussian(self.image, kernel, sigmaX)

    def median(self):
        kernel, ok_kernel = QInputDialog().getInt(self, "Kernel Size", "Kernel Size:", min=1, step=2)
        if ok_kernel:
            filtering.median(self.image, kernel)

    def bilateral(self):
        df, ok_df = QInputDialog().getInt(self, "Filter Size", "Filter Size:")
        sigma_color, ok_sc = QInputDialog().getDouble(self, "Sigma Color", "Sigma Color:")
        sigma_space, ok_ss = QInputDialog().getDouble(self, "Sigma Space", "Sigma Space:")
        
        if ok_df and ok_sc and ok_ss:
            filtering.bilateral(self.image, df, sigma_color, sigma_space)

    def gaussian_noise(self):
        mean, ok_mean = QInputDialog().getInt(self, "Mean", "Mean:")
        var, ok_var = QInputDialog().getDouble(self, "Var", "Var:")
        if ok_mean and ok_var:
            noise.gaussian_noise(self.image, mean, var)

    def salt_and_pepper_noise(self):
        salt_ratio, ok_salt_ratio = QInputDialog().getDouble(self, "Salt ratio", "Salt ratio:")
        amount, ok_amount = QInputDialog().getDouble(self, "Amount", "Amount:")
        if ok_salt_ratio and ok_amount:
            noise.salt_pepper_noise(self.image, salt_ratio, amount)

    def speckle_noise(self):
        noise.speckle_noise(self.image)

    def poisson_noise(self):
        noise.poisson_noise(self.image)

    #  Morphology 
    def erosion(self):
        kernel, ok_kernel = QInputDialog().getInt(self, "Structuring Element Size", "SE matrix size:", min=1, step=2)
        i, ok_itr = QInputDialog().getInt(self, "Iterations", "Number of iterations:", min=1)
        if ok_kernel and ok_itr:
            morphology.erosion(self.image, kernel, i)

    def dilation(self):
        kernel, ok_kernel = QInputDialog().getInt(self, "Structuring Element Size", "SE matrix size:", min=1, step=2)
        i, ok_itr = QInputDialog().getInt(self, "Iterations", "Number of iterations:", min=1)
        if ok_kernel and ok_itr:
            morphology.dilation(self.image, kernel, i)
    
    def transform(self, morph_type):
        kernel, ok_kernel = QInputDialog().getInt(self, "Structuring Element Size", "SE matrix size:", min=1, step=2)
        if ok_kernel:
            morphology.morphological_transformation(self.image, morph_type, kernel)

    def laplacian(self):
        edge_gradient.laplacian(self.image)

    def sobel(self):
        kernel, ok_kernel = QInputDialog().getInt(self, "Kernel Size", "Kernel size:", min=1, max=7, step=2)
        if ok_kernel:
            edge_gradient.sobel(self.image, kernel)
    
    def canny(self):
        treshold1, ok_t1 = QInputDialog().getDouble(self, "Treshold 1", "Treshold 1:")
        treshold2, ok_t2 = QInputDialog().getDouble(self, "Treshold 2", "Treshold 2:")
        if ok_t1 and ok_t2:
            edge_gradient.canny(self.image, treshold1, treshold2)

    def prewitt(self):
        edge_gradient.prewitt(self.image)

    def robert(self):
        edge_gradient.robert(self.image)

    def close_app(self):
        sys.exit()

def run():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()

run()
