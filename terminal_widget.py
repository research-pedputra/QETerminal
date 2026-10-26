"""Basic ANSI/VT terminal screen with a remote PTY, scrollback, and keyboard input."""
from PySide6 import QtCore, QtGui, QtWidgets as W
import pyte


class Terminal(W.QWidget):
    send = QtCore.Signal(bytes)
    resized = QtCore.Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setFont(QtGui.QFont('Consolas', 10))
        self.setMinimumHeight(120)
        self.screen = pyte.HistoryScreen(100, 20, history=2000)
        self.screen.write_process_input = lambda text: self.send.emit(text.encode('utf-8'))
        self.stream = pyte.ByteStream(self.screen)
        self.online = False
        self.reader = None
        self.poll = QtCore.QTimer(self)
        self.poll.setInterval(16)
        self.poll.timeout.connect(self.drain_output)
        self.poll.start()
        self.selection_start = self.selection_end = None
        self.feed(b'Connect to an SSH server to use the terminal.\r\n')

    def feed(self, data): self.stream.feed(data); self.update()
    def drain_output(self):
        if self.reader is not None:
            data = self.reader(16384)
            if data: self.feed(data)
    def reset(self): self.screen.reset(); self.update()
    def metrics(self): return self.fontMetrics().horizontalAdvance('M'), self.fontMetrics().height()
    def resizeEvent(self, event):
        cw, ch = self.metrics()
        columns, rows = max(20, self.width()//cw), max(4, self.height()//ch)
        self.screen.resize(lines=rows, columns=columns)
        self.resized.emit(columns, rows); self.update()
    def color(self, value, background=False):
        colors = dict(default='#151b26' if background else '#dbe5f2', black='#151b26', red='#ef6666', green='#80ce87', brown='#dfbb76', blue='#78a8ec', magenta='#c991d8', cyan='#77d5cf', white='#dbe5f2')
        return QtGui.QColor(colors.get(value, '#' + value if len(value) == 6 else '#dbe5f2'))
    def selected_range(self):
        if self.selection_start is None or self.selection_end is None: return None
        return sorted((self.selection_start, self.selection_end))
    def paintEvent(self, event):
        p = QtGui.QPainter(self); p.fillRect(self.rect(), QtGui.QColor('#151b26'))
        cw, ch = self.metrics(); baseline = self.fontMetrics().ascent()
        selection = self.selected_range()
        for y in range(self.screen.lines):
            for x in range(self.screen.columns):
                char = self.screen.buffer[y][x]
                fg, bg = self.color(char.fg), self.color(char.bg, True)
                if char.reverse: fg, bg = bg, fg
                if selection and selection[0] <= (y, x) <= selection[1]: bg = QtGui.QColor('#3f5878')
                p.fillRect(x*cw, y*ch, cw, ch, bg)
                p.setPen(fg); font = self.font(); font.setBold(char.bold); font.setUnderline(char.underscore); p.setFont(font)
                p.drawText(x*cw, y*ch+baseline, char.data)
        if self.hasFocus() and self.online and not self.screen.cursor.hidden:
            p.setPen(QtGui.QColor('#e2e8f0'))
            p.drawRect(self.screen.cursor.x*cw, self.screen.cursor.y*ch, cw-1, ch-1)
    def send_text(self, text):
        if self.online: self.send.emit(text.encode('utf-8'))
    def keyPressEvent(self, event):
        key, mods = event.key(), event.modifiers()
        ctrl = bool(mods & QtCore.Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        if ctrl and shift and key == QtCore.Qt.Key.Key_C: self.copy(); return
        if ctrl and shift and key == QtCore.Qt.Key.Key_V: self.paste(); return
        if shift and key == QtCore.Qt.Key.Key_PageUp: self.screen.prev_page(); self.update(); return
        if shift and key == QtCore.Qt.Key.Key_PageDown: self.screen.next_page(); self.update(); return
        if not self.online: return
        mapping = {QtCore.Qt.Key.Key_Return:'\r', QtCore.Qt.Key.Key_Enter:'\r', QtCore.Qt.Key.Key_Backspace:'\x7f', QtCore.Qt.Key.Key_Tab:'\t', QtCore.Qt.Key.Key_Escape:'\x1b', QtCore.Qt.Key.Key_Up:'\x1b[A', QtCore.Qt.Key.Key_Down:'\x1b[B', QtCore.Qt.Key.Key_Right:'\x1b[C', QtCore.Qt.Key.Key_Left:'\x1b[D', QtCore.Qt.Key.Key_Home:'\x1b[H', QtCore.Qt.Key.Key_End:'\x1b[F', QtCore.Qt.Key.Key_Delete:'\x1b[3~', QtCore.Qt.Key.Key_PageUp:'\x1b[5~', QtCore.Qt.Key.Key_PageDown:'\x1b[6~'}
        if ctrl and QtCore.Qt.Key.Key_A <= key <= QtCore.Qt.Key.Key_Z:
            self.send.emit(bytes([key - QtCore.Qt.Key.Key_A + 1])); return
        controls = {QtCore.Qt.Key.Key_Space: b'\x00', QtCore.Qt.Key.Key_BracketLeft: b'\x1b',
                    QtCore.Qt.Key.Key_Backslash: b'\x1c', QtCore.Qt.Key.Key_BracketRight: b'\x1d',
                    QtCore.Qt.Key.Key_AsciiCircum: b'\x1e', QtCore.Qt.Key.Key_Underscore: b'\x1f'}
        if ctrl and key in controls: self.send.emit(controls[key]); return
        mapping.update({QtCore.Qt.Key.Key_Insert: '\x1b[2~', QtCore.Qt.Key.Key_Backtab: '\x1b[Z'})
        for index, suffix in enumerate(['OP', 'OQ', 'OR', 'OS', '[15~', '[17~', '[18~', '[19~', '[20~', '[21~', '[23~', '[24~']):
            mapping[int(QtCore.Qt.Key.Key_F1)+index] = '\x1b' + suffix
        if (1 << 5) in self.screen.mode:  # DEC private cursor-key mode.
            for arrow, suffix in [(QtCore.Qt.Key.Key_Up, 'A'), (QtCore.Qt.Key.Key_Down, 'B'), (QtCore.Qt.Key.Key_Right, 'C'), (QtCore.Qt.Key.Key_Left, 'D')]:
                mapping[arrow] = '\x1bO' + suffix
        text = mapping.get(key, event.text())
        if mods & QtCore.Qt.KeyboardModifier.AltModifier: text = '\x1b' + text
        self.send_text(text)
        event.accept()
    def event(self, event):
        # Stop application-wide Save/Undo/Open shortcuts from stealing shell controls.
        if event.type() == QtCore.QEvent.Type.ShortcutOverride and self.hasFocus():
            event.accept(); return True
        return super().event(event)
    def focusNextPrevChild(self, next): return False  # Tab belongs to the shell.
    def paste(self):
        text = W.QApplication.clipboard().text()
        if not text or not self.online: return
        if '\n' in text or '\r' in text:
            if W.QMessageBox.question(self, 'Paste multiple terminal lines', 'Pasting multiple lines may execute commands immediately. Paste?', W.QMessageBox.StandardButton.Yes | W.QMessageBox.StandardButton.No, W.QMessageBox.StandardButton.No) != W.QMessageBox.StandardButton.Yes: return
        self.send_text(text)
    def copy(self):
        selection = self.selected_range()
        if not selection: return
        (y1, x1), (y2, x2) = selection
        lines = []
        for y in range(y1, y2+1):
            first, last = (x1 if y == y1 else 0), (x2+1 if y == y2 else self.screen.columns)
            lines.append(''.join(self.screen.buffer[y][x].data for x in range(first, last)).rstrip())
        W.QApplication.clipboard().setText('\n'.join(lines))
    def cell(self, event):
        cw, ch = self.metrics()
        return (max(0, min(self.screen.lines-1, int(event.position().y())//ch)), max(0, min(self.screen.columns-1, int(event.position().x())//cw)))
    def mousePressEvent(self, event):
        self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.selection_start = self.selection_end = self.cell(event); self.update()
    def focusInEvent(self, event): super().focusInEvent(event); self.update()
    def focusOutEvent(self, event): super().focusOutEvent(event); self.update()
    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self.selection_end = self.cell(event); self.update()
    def wheelEvent(self, event):
        if event.angleDelta().y() > 0: self.screen.prev_page()
        else: self.screen.next_page()
        self.update()
    def contextMenuEvent(self, event):
        menu = W.QMenu(self); menu.addAction('Copy selection', self.copy); menu.addAction('Paste', self.paste)
        menu.exec(event.globalPos())
