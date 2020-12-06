import sys

from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QHeaderView
from UI.mainwindow import *


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
        self.model.setItem(0, 0, QStandardItem(""))

        self.result_list.setModel(self.model)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
