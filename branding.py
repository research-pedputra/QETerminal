"""Proportionally scaled university logo and author credit."""
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets as W


class BrandingFooter(W.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logo = QtGui.QPixmap(str(Path(__file__).parent / 'assets' / 'ntu-logo.png'))
        self.setMinimumHeight(106)
        self.setMaximumHeight(130)
        self.setSizePolicy(W.QSizePolicy.Policy.Expanding, W.QSizePolicy.Policy.Fixed)
        self.setAccessibleName('Nanyang Technological University Singapore. Author: P E D Putra')

    def sizeHint(self):
        return QtCore.QSize(320, 120)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QtGui.QColor('#111c36'))
        area = self.rect().adjusted(12, 10, -12, -32)
        if not self.logo.isNull():
            size = self.logo.size().scaled(area.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            target = QtCore.QRect(QtCore.QPoint(0, 0), size)
            target.moveCenter(area.center())
            painter.drawPixmap(target, self.logo)
        painter.setPen(QtGui.QColor('white'))
        font = self.font(); font.setPointSize(10); painter.setFont(font)
        painter.drawText(self.rect().adjusted(8, self.height()-28, -8, -6), QtCore.Qt.AlignmentFlag.AlignCenter, 'P E D Putra')
