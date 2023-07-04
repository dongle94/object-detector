import os
import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QTableWidget, QLineEdit, QLabel, QPushButton
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtCore import QSize, Slot


class ProdTable(QTableWidget):
    def __init__(self):
        super().__init__()


class MainWidget(QWidget):
    def __init__(self, parent):
        super().__init__()

        self.main_window = parent
        print(self.main_window)

        # top - layout
        self.top = QHBoxLayout()
        self.el_date = QLineEdit()
        self.el_client = QLineEdit()
        self.el_prod_name = QLineEdit()
        self.bt_search = QPushButton("발주 조회")
        self.bt_reset = QPushButton("초기화")

        self.top.addWidget(QLabel("날짜:"))
        self.top.addWidget(self.el_date)
        self.top.addWidget(QLabel("발주사:"))
        self.top.addWidget(self.el_client)
        self.top.addWidget(QLabel("제품명:"))
        self.top.addWidget(self.el_prod_name)
        self.top.addWidget(self.bt_search)
        self.top.addWidget(self.bt_reset)

        # middle - table
        self.middle = ProdTable()

        # bottom - layout
        self.bottom = QHBoxLayout()
        self.bt_start = QPushButton("수량 학인 시작")
        self.bt_stop = QPushButton("수량 확인 종료")

        self.bottom.addWidget(self.bt_start)
        self.bottom.addWidget(self.bt_stop)

        # Main Widget
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.main_layout.addLayout(self.top)
        self.main_layout.addWidget(self.middle)
        self.main_layout.addLayout(self.bottom)




class DaolCND(QMainWindow):
    def __init__(self):
        super().__init__()
        self.geo = self.screen().availableGeometry()
        self.resize(QSize(self.geo.width() * 0.5, self.geo.height() * 0.5))

        # MenuBar
        self.mbar = self.menuBar()
        self.file_menu = self.mbar.addMenu("File")

        # Menu - Exit QAction
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        self.file_menu.addAction(exit_action)

        # Status Bar
        self.statusBar().showMessage("ready state", 0)

        # Set Central Widget
        self.main_widget = MainWidget(self)
        self.setCentralWidget(self.main_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    app_gui = DaolCND()
    app_gui.setWindowTitle("다올씨앤디")
    app_gui.show()

    sys.exit(app.exec())
