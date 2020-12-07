import sys

from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QHeaderView
from UI.mainwindow import *

import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, accuracy_score


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.browse_file.clicked.connect(self.__browse_file)
        self.submit.clicked.connect(self.__execute)

        self.model = QStandardItemModel(10, 6)
        self.model.setHorizontalHeaderLabels(["聚类算法", "降维方法", "Accuracy", "F1", "MSE", "MAE"])
        self.result_list.setModel(self.model)
        self.result_list.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def __browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选取文件", "./", "All Files (*)")
        self.file_path.setText(file_path)

    def __execute(self):
        data = pd.read_csv(self.file_path.text(), header=None)
        X = data.values[:, :-1]
        y = data.values[:, -1]

        dbscan = DBSCAN(eps=self.eps.value(), min_samples=self.minpts.value())
        kmeans = KMeans(n_clusters=self.cluster.value())
        dbscan.fit(X)
        kmeans.fit(X)

        self.model.setItem(0, 0, QStandardItem("DBSCAN(" + str(self.eps.value()) + "," + str(self.minpts.value()) + ")"))
        self.model.setItem(0, 1, QStandardItem("None"))
        self.model.setItem(0, 2, QStandardItem("%.3f" % accuracy_score(y, dbscan.labels_)))
        self.model.setItem(0, 3, QStandardItem("%.3f" % f1_score(y, dbscan.labels_, average="micro")))
        self.model.setItem(0, 4, QStandardItem("%.3f" % mean_squared_error(y, dbscan.labels_)))
        self.model.setItem(0, 5, QStandardItem("%.3f" % mean_absolute_error(y, dbscan.labels_)))

        self.result_list.setModel(self.model)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
