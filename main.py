import sys

import matplotlib
import pandas as pd
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QHeaderView
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from UI.mainwindow import *
from UI.visualize import *

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class MainWindow(QMainWindow, Ui_MainWindow):
    Signal_dimension = pyqtSignal(int)
    Signal_data = pyqtSignal(list)

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.browse_file.clicked.connect(self.__browse_file)
        self.submit.clicked.connect(self.__execute)
        self.VisDialog = VisualizeWindow()
        self.Signal_dimension.connect(self.VisDialog.draw)
        self.Signal_data.connect(self.VisDialog.set_data)
        self.visualize_btn.clicked.connect(self.VisDialog.show)

        self.model = QStandardItemModel(10, 6)
        self.model.setHorizontalHeaderLabels(["聚类算法", "降维方法", "Accuracy", "F1", "MSE", "MAE"])
        self.result_list.setModel(self.model)
        self.result_list.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def __browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选取文件", "./", "All Files (*)")
        self.file_path.setText(file_path)

    def __execute(self):
        try:
            data = pd.read_csv(self.file_path.text(), header=None)
        except FileNotFoundError:
            return

        X = data.values[:, :-1]
        y = data.values[:, -1]

        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        dbscan = DBSCAN(eps=self.eps.value(), min_samples=self.minpts.value())
        kmeans = KMeans(n_clusters=self.cluster.value())
        dbscan.fit(X)
        kmeans.fit(X)

        self.model.setItem(0, 0,
                           QStandardItem("DBSCAN(" + str(self.eps.value()) + "," + str(self.minpts.value()) + ")"))
        self.model.setItem(0, 1, QStandardItem("None"))
        self.model.setItem(0, 2, QStandardItem("%.3f" % accuracy_score(y, dbscan.labels_)))
        self.model.setItem(0, 3, QStandardItem("%.3f" % f1_score(y, dbscan.labels_, average="micro")))
        self.model.setItem(0, 4, QStandardItem("%.3f" % mean_squared_error(y, dbscan.labels_)))
        self.model.setItem(0, 5, QStandardItem("%.3f" % mean_absolute_error(y, dbscan.labels_)))

        self.model.setItem(1, 0, QStandardItem("Kmeans(" + str(self.cluster.value()) + ")"))
        self.model.setItem(1, 1, QStandardItem("None"))
        self.model.setItem(1, 2, QStandardItem("%.3f" % accuracy_score(y, kmeans.labels_)))
        self.model.setItem(1, 3, QStandardItem("%.3f" % f1_score(y, kmeans.labels_, average="micro")))
        self.model.setItem(1, 4, QStandardItem("%.3f" % mean_squared_error(y, kmeans.labels_)))
        self.model.setItem(1, 5, QStandardItem("%.3f" % mean_absolute_error(y, kmeans.labels_)))

        pca = PCA(n_components=self.dimension.value())
        pca_X = pca.fit_transform(X)

        dbscan.fit(pca_X)
        self.model.setItem(2, 0,
                           QStandardItem("DBSCAN(" + str(self.eps.value()) + "," + str(self.minpts.value()) + ")"))
        self.model.setItem(2, 1, QStandardItem("PCA"))
        self.model.setItem(2, 2, QStandardItem("%.3f" % accuracy_score(y, dbscan.labels_)))
        self.model.setItem(2, 3, QStandardItem("%.3f" % f1_score(y, dbscan.labels_, average="micro")))
        self.model.setItem(2, 4, QStandardItem("%.3f" % mean_squared_error(y, dbscan.labels_)))
        self.model.setItem(2, 5, QStandardItem("%.3f" % mean_absolute_error(y, dbscan.labels_)))

        kmeans.fit(pca_X)
        self.model.setItem(3, 0, QStandardItem("Kmeans(" + str(self.cluster.value()) + ")"))
        self.model.setItem(3, 1, QStandardItem("PCA"))
        self.model.setItem(3, 2, QStandardItem("%.3f" % accuracy_score(y, kmeans.labels_)))
        self.model.setItem(3, 3, QStandardItem("%.3f" % f1_score(y, kmeans.labels_, average="micro")))
        self.model.setItem(3, 4, QStandardItem("%.3f" % mean_squared_error(y, kmeans.labels_)))
        self.model.setItem(3, 5, QStandardItem("%.3f" % mean_absolute_error(y, kmeans.labels_)))

        mds = MDS(n_components=self.dimension.value())
        mds_X = mds.fit_transform(X)

        dbscan.fit(mds_X)
        self.model.setItem(4, 0,
                           QStandardItem("DBSCAN(" + str(self.eps.value()) + "," + str(self.minpts.value()) + ")"))
        self.model.setItem(4, 1, QStandardItem("MDS"))
        self.model.setItem(4, 2, QStandardItem("%.3f" % accuracy_score(y, dbscan.labels_)))
        self.model.setItem(4, 3, QStandardItem("%.3f" % f1_score(y, dbscan.labels_, average="micro")))
        self.model.setItem(4, 4, QStandardItem("%.3f" % mean_squared_error(y, dbscan.labels_)))
        self.model.setItem(4, 5, QStandardItem("%.3f" % mean_absolute_error(y, dbscan.labels_)))

        kmeans.fit(mds_X)
        self.model.setItem(5, 0, QStandardItem("Kmeans(" + str(self.cluster.value()) + ")"))
        self.model.setItem(5, 1, QStandardItem("MDS"))
        self.model.setItem(5, 2, QStandardItem("%.3f" % accuracy_score(y, kmeans.labels_)))
        self.model.setItem(5, 3, QStandardItem("%.3f" % f1_score(y, kmeans.labels_, average="micro")))
        self.model.setItem(5, 4, QStandardItem("%.3f" % mean_squared_error(y, kmeans.labels_)))
        self.model.setItem(5, 5, QStandardItem("%.3f" % mean_absolute_error(y, kmeans.labels_)))

        isomap = Isomap(n_components=self.dimension.value())
        isomap_X = isomap.fit_transform(X)

        dbscan.fit(isomap_X)
        self.model.setItem(6, 0,
                           QStandardItem("DBSCAN(" + str(self.eps.value()) + "," + str(self.minpts.value()) + ")"))
        self.model.setItem(6, 1, QStandardItem("Isomap"))
        self.model.setItem(6, 2, QStandardItem("%.3f" % accuracy_score(y, dbscan.labels_)))
        self.model.setItem(6, 3, QStandardItem("%.3f" % f1_score(y, dbscan.labels_, average="micro")))
        self.model.setItem(6, 4, QStandardItem("%.3f" % mean_squared_error(y, dbscan.labels_)))
        self.model.setItem(6, 5, QStandardItem("%.3f" % mean_absolute_error(y, dbscan.labels_)))

        kmeans.fit(isomap_X)
        self.model.setItem(7, 0, QStandardItem("Kmeans(" + str(self.cluster.value()) + ")"))
        self.model.setItem(7, 1, QStandardItem("Isomap"))
        self.model.setItem(7, 2, QStandardItem("%.3f" % accuracy_score(y, kmeans.labels_)))
        self.model.setItem(7, 3, QStandardItem("%.3f" % f1_score(y, kmeans.labels_, average="micro")))
        self.model.setItem(7, 4, QStandardItem("%.3f" % mean_squared_error(y, kmeans.labels_)))
        self.model.setItem(7, 5, QStandardItem("%.3f" % mean_absolute_error(y, kmeans.labels_)))

        lle = LocallyLinearEmbedding(n_components=self.dimension.value())
        lle_X = lle.fit_transform(X)

        dbscan.fit(lle_X)
        self.model.setItem(8, 0,
                           QStandardItem("DBSCAN(" + str(self.eps.value()) + "," + str(self.minpts.value()) + ")"))
        self.model.setItem(8, 1, QStandardItem("LLE"))
        self.model.setItem(8, 2, QStandardItem("%.3f" % accuracy_score(y, dbscan.labels_)))
        self.model.setItem(8, 3, QStandardItem("%.3f" % f1_score(y, dbscan.labels_, average="micro")))
        self.model.setItem(8, 4, QStandardItem("%.3f" % mean_squared_error(y, dbscan.labels_)))
        self.model.setItem(8, 5, QStandardItem("%.3f" % mean_absolute_error(y, dbscan.labels_)))

        kmeans.fit(lle_X)
        self.model.setItem(9, 0, QStandardItem("Kmeans(" + str(self.cluster.value()) + ")"))
        self.model.setItem(9, 1, QStandardItem("LLE"))
        self.model.setItem(9, 2, QStandardItem("%.3f" % accuracy_score(y, kmeans.labels_)))
        self.model.setItem(9, 3, QStandardItem("%.3f" % f1_score(y, kmeans.labels_, average="micro")))
        self.model.setItem(9, 4, QStandardItem("%.3f" % mean_squared_error(y, kmeans.labels_)))
        self.model.setItem(9, 5, QStandardItem("%.3f" % mean_absolute_error(y, kmeans.labels_)))

        self.result_list.setModel(self.model)

        self.Signal_data.emit([pca_X, mds_X, isomap_X, lle_X, y])
        self.Signal_dimension.emit(self.dimension.value())


class VisualizeWindow(QMainWindow, Ui_Visualize):

    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.data = None

    def set_data(self, data):
        self.data = data

    def draw(self, dimension):
        if self.data is None:
            return
        if dimension == 2:
            self.draw_2D()
        elif dimension == 3:
            self.draw_3D()
        else:
            fig = plt.figure()
            canvas = FigureCanvas(fig)
            self.setCentralWidget(canvas)

    def draw_2D(self):
        fig = plt.figure()
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        ax_pca = fig.add_subplot(221)
        ax_pca.set_title("PCA")

        ax_mds = fig.add_subplot(222)
        ax_mds.set_title("MDS")

        ax_isomap = fig.add_subplot(223)
        ax_isomap.set_title("Isomap")

        ax_lle = fig.add_subplot(224)
        ax_lle.set_title("LLE")

        ax_pca.scatter(self.data[0][:, 0], self.data[0][:, 1], s=0.8, c=self.data[4])
        ax_mds.scatter(self.data[1][:, 0], self.data[1][:, 1], s=0.8, c=self.data[4])
        ax_isomap.scatter(self.data[2][:, 0], self.data[2][:, 1], s=0.8, c=self.data[4])
        ax_lle.scatter(self.data[3][:, 0], self.data[3][:, 1], s=0.8, c=self.data[4])

        canvas = FigureCanvas(fig)
        self.setCentralWidget(canvas)

    def draw_3D(self):
        fig = plt.figure()
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        ax_pca = fig.add_subplot(221, projection='3d')
        ax_pca.set_title("PCA")

        ax_mds = fig.add_subplot(222, projection='3d')
        ax_mds.set_title("MDS")

        ax_isomap = fig.add_subplot(223, projection='3d')
        ax_isomap.set_title("Isomap")

        ax_lle = fig.add_subplot(224, projection='3d')
        ax_lle.set_title("LLE")

        ax_pca.scatter(self.data[0][:, 0], self.data[0][:, 1], self.data[0][:, 2], s=0.8, c=self.data[4])
        ax_mds.scatter(self.data[1][:, 0], self.data[1][:, 1], self.data[1][:, 2], s=0.8, c=self.data[4])
        ax_isomap.scatter(self.data[2][:, 0], self.data[2][:, 1], self.data[2][:, 2], s=0.8, c=self.data[4])
        ax_lle.scatter(self.data[3][:, 0], self.data[3][:, 1], self.data[3][:, 2], s=0.8, c=self.data[4])

        canvas = FigureCanvas(fig)
        self.setCentralWidget(canvas)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    main = MainWindow()

    main.show()

    sys.exit(app.exec_())
