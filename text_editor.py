"""Text editing with line numbers, QE highlighting, and byte-preserving encoding choices."""
import codecs
import re
from PySide6 import QtCore, QtGui, QtWidgets as W
from ssh_session import digest


def decode_text(data, encoding=None):
    if encoding is None:
        encoding = 'utf-8-sig' if data.startswith(codecs.BOM_UTF8) else ('utf-16' if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) else 'utf-8')
    text = data.decode(encoding)
    if '\x00' in text: raise ValueError('This looks like a binary file, not editable text.')
    newline = '\r\n' if '\r\n' in text else ('\r' if '\r' in text else '\n')
    # Use explicit endianness so UTF-16BE is not silently converted to native LE.
    if encoding == 'utf-16':
        encoding = 'utf-16-be' if data.startswith(codecs.BOM_UTF16_BE) else 'utf-16-le'
    return text.replace('\r\n', '\n').replace('\r', '\n'), encoding, newline


class QEHighlighter(QtGui.QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.rules = []
        for pattern, color, bold in [
            (r'(?i)\b(?:ATOMIC_SPECIES|ATOMIC_POSITIONS|CELL_PARAMETERS|K_POINTS|BEGIN|END|FINAL|COORDINATES)\b|&\w+', '#2855b0', True),
            (r'(?i)\b(?:crystal|angstrom|bohr|alat|automatic|gamma|true|false)\b', '#8754a1', False),
            (r'\b[A-Za-z_][\w()]*\s*(?==)', '#0f766e', False),
            (r'(?i)(?<!\w)[+-]?(?:\d+\.?\d*|\.\d+)(?:[de][+-]?\d+)?', '#b45309', False),
            (r"'[^']*'|\"[^\"]*\"", '#328142', False),
            (r'!.*$', '#7b8794', False),
        ]:
            fmt = QtGui.QTextCharFormat(); fmt.setForeground(QtGui.QColor(color))
            if bold: fmt.setFontWeight(QtGui.QFont.Weight.Bold)
            self.rules.append((re.compile(pattern), fmt))

    def highlightBlock(self, text):
        for regex, fmt in self.rules:
            for match in regex.finditer(text): self.setFormat(match.start(), len(match.group()), fmt)


class LineNumbers(W.QWidget):
    def __init__(self, editor): super().__init__(editor); self.editor = editor
    def sizeHint(self): return QtCore.QSize(self.editor.margin_width(), 0)
    def paintEvent(self, event): self.editor.paint_numbers(event)


class CodeEditor(W.QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setFont(QtGui.QFont('Consolas', 10))
        self.setLineWrapMode(W.QPlainTextEdit.LineWrapMode.NoWrap)
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(' ')*4)
        self.numbers = LineNumbers(self)
        self.blockCountChanged.connect(self.update_margin)
        self.updateRequest.connect(self.update_numbers)
        self.cursorPositionChanged.connect(self.highlight_line)
        self.update_margin(); self.highlight_line()

    def margin_width(self): return 12 + self.fontMetrics().horizontalAdvance('9')*len(str(max(1, self.blockCount())))
    def update_margin(self): self.setViewportMargins(self.margin_width(), 0, 0, 0)
    def update_numbers(self, rect, dy):
        if dy: self.numbers.scroll(0, dy)
        else: self.numbers.update(0, rect.y(), self.numbers.width(), rect.height())
        if rect.contains(self.viewport().rect()): self.update_margin()
    def resizeEvent(self, event):
        super().resizeEvent(event)
        rect = self.contentsRect(); self.numbers.setGeometry(rect.left(), rect.top(), self.margin_width(), rect.height())
    def paint_numbers(self, event):
        p = QtGui.QPainter(self.numbers); p.fillRect(event.rect(), QtGui.QColor('#f1f5f9'))
        block = self.firstVisibleBlock(); number = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        while block.isValid() and top <= event.rect().bottom():
            height = round(self.blockBoundingRect(block).height())
            if block.isVisible() and top+height >= event.rect().top():
                p.setPen(QtGui.QColor('#64748b')); p.drawText(0, top, self.numbers.width()-5, self.fontMetrics().height(), QtCore.Qt.AlignmentFlag.AlignRight, str(number+1))
            top += height; block = block.next(); number += 1
    def highlight_line(self):
        selection = W.QTextEdit.ExtraSelection()
        selection.format.setBackground(QtGui.QColor('#eef4ff'))
        selection.format.setProperty(QtGui.QTextFormat.Property.FullWidthSelection, True)
        selection.cursor = self.textCursor(); selection.cursor.clearSelection()
        self.setExtraSelections([selection])


class TextDocument(W.QWidget):
    save_requested = QtCore.Signal()
    save_as_requested = QtCore.Signal()
    def __init__(self, path, data, remote_session=None, encoding=None):
        super().__init__()
        self.path = path; self.remote_session = remote_session
        text, self.encoding, self.newline = decode_text(data, encoding)
        self.bom = codecs.BOM_UTF16_BE if data.startswith(codecs.BOM_UTF16_BE) else (codecs.BOM_UTF16_LE if data.startswith(codecs.BOM_UTF16_LE) else b'')
        self.original_data = data
        self.expected = digest(data)
        self.save_pending = False
        layout = W.QVBoxLayout(self); layout.setContentsMargins(4, 4, 4, 4)
        self.path_label = W.QLabel(('REMOTE: ' if remote_session else '') + path)
        self.path_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.path_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.path_label)
        find = W.QHBoxLayout()
        self.search = W.QLineEdit(); self.search.setPlaceholderText('Find text…')
        self.replace = W.QLineEdit(); self.replace.setPlaceholderText('Replace with…')
        find.addWidget(self.search); find.addWidget(self.replace)
        for label, action in [('Next', self.find_next), ('Replace', self.replace_one), ('Replace all', self.replace_all)]:
            button = W.QPushButton(label); button.clicked.connect(action); find.addWidget(button)
        save = W.QPushButton('Save'); save.clicked.connect(self.save_requested.emit); find.addWidget(save)
        save_as = W.QPushButton('Save As…'); save_as.clicked.connect(self.save_as_requested.emit); find.addWidget(save_as)
        layout.addLayout(find)
        self.editor = CodeEditor(); layout.addWidget(self.editor)
        self.editor.setPlainText(text); self.editor.document().setModified(False)
        self.initial_text = self.editor.toPlainText()
        self.highlighter = QEHighlighter(self.editor.document())
        self.info = W.QLabel(); layout.addWidget(self.info)
        self.editor.cursorPositionChanged.connect(self.status)
        self.editor.textChanged.connect(self.status)
        self.search.returnPressed.connect(self.find_next)
        self.status()

    def payload(self):
        text = self.editor.toPlainText()
        if text == self.initial_text: return self.original_data
        data = text.replace('\n', self.newline).encode(self.encoding)
        return self.bom + data if self.bom and not data.startswith(self.bom) else data
    def dirty(self): return self.editor.document().isModified()
    def status(self):
        c = self.editor.textCursor()
        eol = {'\r\n': 'CRLF', '\n': 'LF', '\r': 'CR'}[self.newline]
        self.info.setText(f'Line {c.blockNumber()+1}, column {c.positionInBlock()+1} · {self.encoding} · {eol}' + (' · modified' if self.dirty() else ''))
    def find_next(self):
        if not self.search.text(): return
        if not self.editor.find(self.search.text()):
            c = self.editor.textCursor(); c.movePosition(QtGui.QTextCursor.MoveOperation.Start); self.editor.setTextCursor(c)
            self.editor.find(self.search.text())
    def replace_one(self):
        c = self.editor.textCursor()
        if self.search.text() and c.selectedText() == self.search.text(): c.insertText(self.replace.text())
        self.find_next()
    def replace_all(self):
        if not self.search.text(): return
        text = self.editor.toPlainText().replace(self.search.text(), self.replace.text())
        c = self.editor.textCursor(); c.beginEditBlock(); c.select(QtGui.QTextCursor.SelectionType.Document)
        c.insertText(text); c.endEditBlock()
