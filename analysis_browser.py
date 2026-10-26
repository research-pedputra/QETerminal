"""Asynchronous SSH file/folder picker using the existing SFTP connection."""
import posixpath
from PySide6 import QtCore, QtWidgets as W


class SSHFileDialog(W.QDialog):
    def __init__(self, workspace, parent=None, directory=None, folders_only=False):
        super().__init__(parent or workspace.viewer)
        self.workspace = workspace; self.folders_only = folders_only
        self.session_id = workspace.session.session_id if workspace.online else None
        self.folder = None; self.paths = []; self.generation = 0; self.ready = False
        self.setWindowTitle('Choose SSH analysis folder' if folders_only else 'Open SSH analysis files')
        self.resize(760, 510)
        layout = W.QVBoxLayout(self); row = W.QHBoxLayout(); layout.addLayout(row)
        self.location = W.QLineEdit(directory or workspace.remote_folder)
        for label, callback in [('Up', self.up), ('Home', lambda: self.navigate('.'))]:
            button = W.QPushButton(label); button.clicked.connect(callback); row.addWidget(button)
        row.addWidget(self.location, 1)
        for label in ('Go', 'Refresh'):
            button = W.QPushButton(label); button.clicked.connect(lambda: self.navigate(self.location.text())); row.addWidget(button)
        self.location.returnPressed.connect(lambda: self.navigate(self.location.text()))
        self.filter = W.QLineEdit(); self.filter.setPlaceholderText('Filter filenames…'); layout.addWidget(self.filter)
        self.tree = W.QTreeWidget(); self.tree.setHeaderLabels(['Name', 'Type', 'Size (bytes)'])
        self.tree.setSelectionMode(W.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setRootIsDecorated(False); self.tree.header().setSectionResizeMode(0, W.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.tree)
        self.status = W.QLabel(); self.status.setWordWrap(True); self.status.setTextFormat(QtCore.Qt.TextFormat.PlainText); layout.addWidget(self.status)
        self.buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Open | W.QDialogButtonBox.StandardButton.Cancel)
        self.open_button = self.buttons.button(W.QDialogButtonBox.StandardButton.Open)
        self.open_button.setText('Use this folder' if folders_only else 'Open selected files')
        layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self.choose); self.buttons.rejected.connect(self.reject)
        self.tree.itemDoubleClicked.connect(self.activate); self.tree.itemSelectionChanged.connect(self.update_selection)
        self.filter.textChanged.connect(self.filter_items)
        self.finished.connect(self.invalidate)
        self.navigate(self.location.text())

    def connected(self):
        return self.workspace.online and self.workspace.session.session_id == self.session_id

    def invalidate(self, *args): self.generation += 1; self.ready = False

    def up(self):
        self.navigate(posixpath.dirname((self.folder or self.location.text()).rstrip('/')) or '/')

    def navigate(self, path):
        self.generation += 1; generation = self.generation; self.ready = False
        self.tree.clear(); self.open_button.setEnabled(False); self.status.setText('Loading…')
        if not self.connected(): self.status.setText('SSH connection changed. Reopen the browser after connecting.'); return
        path = path.strip() or '.'
        if not posixpath.isabs(path) and path != '.' and self.folder: path = posixpath.join(self.folder, path)
        def loaded(value):
            if generation != self.generation or not self.connected(): return
            self.folder, entries = value; self.location.setText(self.folder); self.ready = True
            for name, directory, size in entries:
                if self.folders_only and not directory: continue
                item = W.QTreeWidgetItem([name, 'Folder' if directory else 'File', '' if directory else str(size)])
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, (posixpath.join(self.folder, name), directory))
                item.setIcon(0, self.style().standardIcon(W.QStyle.StandardPixmap.SP_DirIcon if directory else W.QStyle.StandardPixmap.SP_FileIcon))
                self.tree.addTopLevelItem(item)
            self.filter_items(); self.status.setText('Double-click folders to browse. ' + ('Choose the current folder below.' if self.folders_only else 'Ctrl/Shift selects multiple files.'))
        def failed(message):
            if generation == self.generation: self.status.setText(message)
        try: self.workspace.request('list', (path,), loaded, failed)
        except Exception as exc: failed(str(exc))

    def filter_items(self, *args):
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item.setHidden(self.filter.text().casefold() not in item.text(0).casefold())
        self.update_selection()

    def selected_files(self):
        return [item.data(0, QtCore.Qt.ItemDataRole.UserRole)[0] for item in self.tree.selectedItems()
                if not item.isHidden() and not item.data(0, QtCore.Qt.ItemDataRole.UserRole)[1]]

    def update_selection(self): self.open_button.setEnabled(self.ready and (self.folders_only or bool(self.selected_files())))

    def activate(self, item, column):
        path, directory = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if directory: self.navigate(path)
        elif not self.folders_only: self.paths = [path]; self.choose(single=True)

    def choose(self, single=False):
        if not self.ready or not self.connected():
            self.status.setText('SSH connection changed or folder is still loading.'); return
        if not self.folders_only:
            if not single: self.paths = self.selected_files()
            if not self.paths: return
        self.accept()
