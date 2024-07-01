
from PyQt5.QtCore import pyqtSlot, Qt, QPoint
from PyQt5.QtCore import QEvent

# Add Filter on the events
def eventFilter(self, source, event):
    try:

        if source == self.lbl_line_p1:
            #print('selct p1')
            self.Event_lblDispP1_Pos(event)
            if self.f_pad_image == 1 and self.f_P1_Mou_Left == 1:
                self.slot_draw_main_imge()

        elif source == self.lbl_line_p2:
            #print('selct p2')
            self.Event_lblDispP2_Pos(event)
            if self.f_pad_image == 1 and self.f_P2_Mou_Left == 1:
                self.slot_draw_main_imge()

        return False

    except Exception as e:
        print('error=', e)


def Event_lblDispP1_Pos(self, event):
    if event.type() == QEvent.MouseMove:

        if event.buttons() == Qt.LeftButton:

            point = QPoint(event.pos())
            x = int(point.x())
            y = int(point.y())
            #self.statusBar().showMessage('X:' + str(x) + ', Y:' + str(y))

            chg_x = x - self.pre_p1_x
            chg_y = y - self.pre_p1_y

            self.p1_cx = int(self.p1_cx + chg_x)
            self.p1_cy = int(self.p1_cy + chg_y)

            self.statusBar().showMessage('X:' + str(self.p1_cx) + ', Y:' + str(self.p1_cy))

            self.f_P1_Mou_Left = 1

            if event.type() == QEvent.MouseButtonRelease:
                self.f_P1_Mou_Left = 0

    elif event.type() == QEvent.MouseButtonPress and event.buttons() == Qt.LeftButton:
        point = QPoint(event.pos())
        self.pre_p1_x = int(point.x())
        self.pre_p1_y = int(point.y())


def Event_lblDispP2_Pos(self, event):
    if event.type() == QEvent.MouseMove:

        if event.buttons() == Qt.LeftButton:

            point = QPoint(event.pos())
            x = int(point.x())
            y = int(point.y())
            #self.statusBar().showMessage('X:' + str(x) + ', Y:' + str(y))

            chg_x = x - self.pre_p2_x
            chg_y = y - self.pre_p2_y

            self.p2_cx = int(self.p2_cx + chg_x)
            self.p2_cy = int(self.p2_cy + chg_y)

            self.statusBar().showMessage('X:' + str(self.p2_cx) + ', Y:' + str(self.p2_cy))

            self.f_P2_Mou_Left = 1

            if event.type() == QEvent.MouseButtonRelease:
                self.f_P2_Mou_Left = 0

    elif event.type() == QEvent.MouseButtonPress and event.buttons() == Qt.LeftButton:
        point = QPoint(event.pos())
        self.pre_p2_x = int(point.x())
        self.pre_p2_y = int(point.y())

