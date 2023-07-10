import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton

from PySide6.QtCore import QSize, Qt


ROW = ["chicken", "sauce1", "sauce2", "powder", "발주물량"]

PROD = [
    {
        "name": "수원왕갈비 꾸닭",
        "quantity": 50,
        "chicken": ["태음융융소금염지닭", 50],
        "sauce1": ["닭갈비간장", 25],
        "sauce2": ["갈비레이", 25],
        "powder": ["에어크런치", 50]
    },
    {
        "name": "오리지널 치밥",
        "quantity": 30,
        "chicken": ["후라이드염지닭", 30],
        "sauce1": ["치밥버무림소스", 15],
        "sauce2": ["치밥벌크소스", 15],
        "powder": ["크리스피파우더", 30]
    },
    {
        "name": "소이퐁 튀닭",
        "quantity": 50,
        "chicken": ["후라이드염지닭", 50],
        "sauce1": ["소이퐁소스", 50],
        "sauce2": [],
        "powder": ["우리쌀 후레이크파우더", 50]
    },
    {
        "name": "공주매콤 닭갈비",
        "quantity": 50,
        "chicken": ["태음융융소금염지닭", 50],
        "sauce1": ["닭갈비 간장양념", 50],
        "sauce2": [],
        "powder": []
    },
    {
        "name": "크리스피 튀닭",
        "quantity": 50,
        "chicken": ["핫커리염지닭", 50],
        "sauce1": ["맛있게매운소스", 25],
        "sauce2": ["프리마늘소스", 25],
        "powder": ["크리스피파우더", 50]
    },
]


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setupUI()

    def setupUI(self):

        # Whole layout
        self.layout = QVBoxLayout(self)

        # Upper Editline & button
        self.widget_1 = QWidget()
        layout_1 = QHBoxLayout()

        label_1_1 = QLabel("날짜: ")
        self.lineEdit_1_1 = QLineEdit()
        label_1_2 = QLabel("발주사: ")
        self.lineEdit_1_2 = QLineEdit()
        label_1_3 = QLabel("제품명: ")
        self.lineEdit_1_3 = QLineEdit()
        self.button_1 = QPushButton("발주 조회")
        self.button_2 = QPushButton("초기화")

        layout_1.addWidget(label_1_1)
        layout_1.addWidget(self.lineEdit_1_1)
        layout_1.addWidget(label_1_2)
        layout_1.addWidget(self.lineEdit_1_2)
        layout_1.addWidget(label_1_3)
        layout_1.addWidget(self.lineEdit_1_3)
        layout_1.addWidget(self.button_1)
        layout_1.addWidget(self.button_2)

        self.widget_1.setLayout(layout_1)
        self.layout.addWidget(self.widget_1)

        # add table
        self.tb = MyTable()
        self.layout.addWidget(self.tb)

        # add lower buttons
        self.widget_2 = QWidget()
        layout_2 = QHBoxLayout()
        self.button_3 = QPushButton("수량 확인 시작")
        self.button_4 = QPushButton("종료")
        layout_2.addWidget(self.button_3, alignment=Qt.AlignRight)
        layout_2.addWidget(self.button_4, alignment=Qt.AlignRight)

        self.widget_2.setLayout(layout_2)
        # self.layout.addLayout(layout_2)
        self.layout.addWidget(self.widget_2)

        # Connect Function
        self.button_1.clicked.connect(self.show_table)
        self.button_2.clicked.connect(self.clear_table)

    def show_table(self):
        self.tb.show_header()
        self.tb.show_content()

        #self.setMaximumWidth(self.tb.width())

    def clear_table(self):
        self.layout.removeWidget(self.tb)
        self.tb.clear()
        self.tb.deleteLater()
        self.tb = MyTable()
        self.layout.insertWidget(1, self.tb)


class MyTable(QTableWidget):
    def __init__(self):
        super().__init__()
        self.content = None

    def show_header(self):
        self.setColumnCount(len(PROD))
        self.setRowCount(len(ROW))
        self.setHorizontalHeaderLabels([p["name"] for p in PROD])
        self.setVerticalHeaderLabels(ROW)

    def show_content(self):
        line_thick = 3
        len_col = len(PROD)

        th_name = 100
        th_num = 30
        th_cur_num = 30
        self.content = {}
        for col, prod in enumerate(PROD):
            self.content[prod['name']] = {}
            for row, r in enumerate(ROW[:4]):
                self.content[prod['name']][r] = {}
                if prod[r]:
                    _tb = QTableWidget()
                    _tb.setColumnCount(3)
                    _tb.setRowCount(1)

                    _tb.verticalHeader().hide()
                    _tb.horizontalHeader().hide()

                    # _tb.horizontalScrollBar().setDisabled(True)
                    item_name = QTableWidgetItem(prod[r][0])
                    _tb.setItem(0, 0, item_name)
                    _tb.setColumnWidth(0, th_name)
                    self.content[prod['name']][r]['name'] = str(prod[r][0])

                    item_num = QTableWidgetItem(str(prod[r][1]))
                    _tb.setItem(0, 1, item_num)
                    _tb.setColumnWidth(1, th_num)
                    self.content[prod['name']][r]['num'] = str(prod[r][1])

                    _tb.setItem(0, 2, QTableWidgetItem(str(0)))
                    _tb.setColumnWidth(2, th_cur_num)
                    self.content[prod['name']][r]['cur_num'] = str(0)

                    self.setCellWidget(row, col, _tb)

            num = QTableWidgetItem(str(prod['quantity'])).setTextAlignment(Qt.AlignLeft)
            print(num)
            self.setItem(4, col, num)
            for i in range(len_col):
                self.setColumnWidth(i, th_name + th_num + th_cur_num + line_thick)

        self.setMinimumWidth((th_name + th_num + th_cur_num) * len_col)
        # self.setMaximumWidth((th_name + th_num + th_cur_num) * len_col + 90)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    mw = MyWidget()
    mw.show()

    sys.exit(app.exec())