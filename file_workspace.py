"""Local/SSH explorer, editor tabs, and terminal integration for Structure Studio."""
import posixpath
import tempfile
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets as W
from ssh_session import SSHSession, EDITOR_LIMIT, STRUCTURE_LIMIT, digest
from text_editor import TextDocument
from terminal_widget import Terminal
from ssh_profiles import SSHProfiles

QE_SUFFIXES = {'.in', '.out', '.pwi', '.pwo', '.pwin', '.pwout'}
STRUCTURE_SUFFIXES = QE_SUFFIXES | {'.cif', '.xyz', '.extxyz', '.vasp', '.poscar'}


class ConnectDialog(W.QDialog):
    def __init__(self, parent, profiles=None):
        super().__init__(parent); self.setWindowTitle('Connect to SSH server'); self.resize(480, 330)
        layout = W.QVBoxLayout(self); form = W.QFormLayout(); layout.addLayout(form)
        self.profiles = profiles if profiles is not None else SSHProfiles()
        self.profile = W.QComboBox(); self.profile.setEditable(True)
        self.profile.setPlaceholderText('Profile name (optional)')
        self.profile.addItem(''); self.profile.addItems(sorted(self.profiles.all()))
        row = W.QHBoxLayout(); row.addWidget(self.profile, 1)
        save = W.QPushButton('Save'); remove = W.QPushButton('Delete')
        row.addWidget(save); row.addWidget(remove); form.addRow('Saved connection', row)
        self.host = W.QLineEdit(); self.host.setPlaceholderText('server.example.edu')
        self.user = W.QLineEdit(); self.port = W.QSpinBox(); self.port.setRange(1, 65535); self.port.setValue(22)
        self.password = W.QLineEdit(); self.password.setEchoMode(W.QLineEdit.EchoMode.Password)
        self.key = W.QLineEdit(); self.passphrase = W.QLineEdit(); self.passphrase.setEchoMode(W.QLineEdit.EchoMode.Password)
        self.directory = W.QLineEdit('.')
        for label, widget in [('Host', self.host), ('Username', self.user), ('Port', self.port), ('Password (optional)', self.password)]: form.addRow(label, widget)
        row = W.QHBoxLayout(); row.addWidget(self.key); browse = W.QPushButton('Browse key…'); row.addWidget(browse)
        browse.clicked.connect(self.browse); form.addRow('Private key (optional)', row)
        form.addRow('Key passphrase (optional)', self.passphrase); form.addRow('Initial remote folder', self.directory)
        self.remember = W.QCheckBox('Save connection settings when connecting'); self.remember.setChecked(True)
        layout.addWidget(self.remember)
        note = W.QLabel('Passwords and key passphrases in saved profiles are protected with Windows user credential encryption. Interactive verification codes are never saved.')
        note.setWordWrap(True); layout.addWidget(note)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Ok | W.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(W.QDialogButtonBox.StandardButton.Ok).setText('Connect')
        buttons.accepted.connect(self.connect_clicked); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
        self.profile.activated.connect(lambda _: self.load_profile())
        save.clicked.connect(self.save_profile); remove.clicked.connect(self.delete_profile)
        if self.profiles.last() in self.profiles.all():
            self.profile.setCurrentText(self.profiles.last()); self.load_profile()
    def load_profile(self):
        config = self.profiles.all().get(self.profile.currentText(), {})
        for key in ('host', 'user', 'key', 'directory'):
            getattr(self, key).setText(str(config.get(key, '.' if key == 'directory' else '')))
        self.port.setValue(int(config.get('port', 22)))
        self.password.setText(str(config.get('password', ''))); self.passphrase.setText(str(config.get('passphrase', '')))
    def save_profile(self):
        try:
            config = self.config()
            name = self.profile.currentText().strip() or f'{config["user"]}@{config["host"]}:{config["port"]}'
            self.profiles.save(name, config)
            if self.profile.findText(name) < 0: self.profile.addItem(name)
            self.profile.setCurrentText(name)
            return True
        except Exception as exc:
            W.QMessageBox.warning(self, 'SSH profile', str(exc)); return False
    def delete_profile(self):
        name = self.profile.currentText(); self.profiles.delete(name)
        index = self.profile.findText(name)
        if index >= 0: self.profile.removeItem(index)
        self.profile.setCurrentText(''); self.load_profile()
    def connect_clicked(self):
        try: self.config()
        except ValueError as exc:
            W.QMessageBox.warning(self, 'SSH connection', str(exc)); return
        if self.remember.isChecked() and not self.save_profile(): return
        self.accept()
    def browse(self):
        path, _ = W.QFileDialog.getOpenFileName(self, 'Select SSH private key', str(Path.home()/'.ssh'))
        if path: self.key.setText(path)
    def config(self):
        if not self.host.text().strip() or not self.user.text().strip(): raise ValueError('Enter a host and username.')
        return dict(host=self.host.text().strip(), user=self.user.text().strip(), port=self.port.value(),
                    password=self.password.text(), key=self.key.text().strip(), passphrase=self.passphrase.text(), directory=self.directory.text().strip())


class RemoteFolderDialog(W.QDialog):
    """Browse folders using the workspace's asynchronous SFTP connection."""
    def __init__(self, workspace):
        super().__init__(workspace.viewer)
        self.workspace = workspace; self.folder = None; self.generation = 0
        self.setWindowTitle('Open SSH folder'); self.resize(560, 420)
        layout = W.QVBoxLayout(self); row = W.QHBoxLayout()
        self.location = W.QLineEdit(getattr(workspace, 'remote_folder', '.'))
        row.addWidget(self.location)
        go = W.QPushButton('Go'); row.addWidget(go)
        layout.addLayout(row)
        self.tree = W.QTreeWidget(); self.tree.setHeaderLabels(['Folders'])
        layout.addWidget(self.tree)
        self.status = W.QLabel(); self.status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Ok | W.QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(W.QDialogButtonBox.StandardButton.Ok).setText('Select this folder')
        self.buttons.accepted.connect(self.accept); self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        go.clicked.connect(lambda: self.navigate(self.location.text()))
        self.location.returnPressed.connect(lambda: self.navigate(self.location.text()))
        self.tree.itemDoubleClicked.connect(lambda item, column: self.navigate(item.data(0, QtCore.Qt.ItemDataRole.UserRole)))
        self.finished.connect(lambda _: self.invalidate())
        self.navigate(self.location.text())

    def invalidate(self):
        self.generation += 1

    def navigate(self, path):
        self.generation += 1; generation = self.generation
        self.buttons.button(W.QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.status.setText('Loading folders…')
        def loaded(value):
            if generation != self.generation: return
            self.folder, entries = value; self.location.setText(self.folder); self.tree.clear()
            for name, directory, size in entries:
                if directory:
                    item = W.QTreeWidgetItem([name])
                    item.setData(0, QtCore.Qt.ItemDataRole.UserRole, posixpath.join(self.folder, name))
                    item.setIcon(0, self.style().standardIcon(W.QStyle.StandardPixmap.SP_DirIcon))
                    self.tree.addTopLevelItem(item)
            self.status.setText('Double-click a folder to explore it.')
            self.buttons.button(W.QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        def failed(message):
            if generation == self.generation: self.status.setText(message)
        try: self.workspace.request('list', (path.strip() or '.',), loaded, failed)
        except Exception as exc: failed(str(exc))


class FileWorkspace(QtCore.QObject):
    def __init__(self, viewer):
        super().__init__(viewer)
        self.viewer = viewer; self.session = None; self.online = False
        self.pending = {}; self.documents = []; self.cubes = []
        self.temp = tempfile.TemporaryDirectory(prefix='qe-workspace-')
        self.folder = str(Path.cwd())
        self.closing = False
        structure = viewer.takeCentralWidget()
        self.views = W.QTabWidget(); self.views.setTabsClosable(True); self.views.addTab(structure, '3D structure')
        self.views.tabBar().setTabButton(0, W.QTabBar.ButtonPosition.RightSide, None)
        self.views.tabCloseRequested.connect(self.close_tab)
        self.vertical = W.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.vertical.addWidget(self.views)
        terminal_panel = W.QWidget(); terminal_layout = W.QVBoxLayout(terminal_panel); terminal_layout.setContentsMargins(0, 0, 0, 0)
        row = W.QHBoxLayout(); self.connection_label = W.QLabel('SSH: disconnected'); self.connection_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        row.addWidget(self.connection_label, 1)
        terminal_layout.addLayout(row); self.terminal = Terminal(); terminal_layout.addWidget(self.terminal)
        self.terminal.send.connect(lambda data: self.session.send(data) if self.online else None)
        self.terminal.resized.connect(lambda cols, rows: self.session.resize_terminal(cols, rows) if self.session else None)
        self.vertical.addWidget(terminal_panel); self.vertical.setSizes([680, 220]); viewer.setCentralWidget(self.vertical)
        self.make_explorer(); self.make_menus()
        self.file_clipboard = None
        self.clipboard_worker = None
        self.install_file_clipboard()
        analysis = viewer.menuBar().addMenu('Analysis')
        analysis.addAction('Open analysis studio…', self.open_analysis)

    def open_analysis(self):
        from analysis_window import AnalysisWindow
        if not hasattr(self, 'analysis_window'):
            self.analysis_window = AnalysisWindow(self)
        self.analysis_window.show(); self.analysis_window.raise_(); self.analysis_window.activateWindow()

    def install_file_clipboard(self):
        for tree, remote in ((self.local_tree, False), (self.remote_tree, True)):
            tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.ActionsContextMenu)
            for label, key, callback in (
                ('Copy files', 'Ctrl+C', lambda r=remote: self.copy_files(r, False)),
                ('Cut files', 'Ctrl+X', lambda r=remote: self.copy_files(r, True)),
                ('Paste files into current folder', 'Ctrl+V', lambda r=remote: self.paste_files(r))):
                action = QtGui.QAction(label, tree); action.setShortcut(key)
                action.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
                action.triggered.connect(lambda checked=False, cb=callback: self.guard(cb)); tree.addAction(action)

    def copy_files(self, remote, cut):
        if self.transfer_busy: raise ValueError('Wait for the current transfer to finish.')
        if remote:
            if not self.online: raise ValueError('Connect SSH first.')
            sources = [i.data(0, QtCore.Qt.ItemDataRole.UserRole)[0] for i in self.remote_tree.selectedItems() if i.data(0, QtCore.Qt.ItemDataRole.UserRole)]
        else: sources = [self.local_model.filePath(i) for i in self.local_tree.selectionModel().selectedRows(0)]
        if not sources: return
        self.file_clipboard = (sources, remote, cut, self.session.session_id if remote else None)
        self.show_transfer_status(f'{len(sources)} file(s) ready to {"move" if cut else "copy"}. Paste in either pane’s current folder.')

    def paste_files(self, remote):
        if not self.file_clipboard or self.transfer_busy: return
        sources, source_remote, cut, session = self.file_clipboard
        if (remote or source_remote) and not self.online: raise ValueError('Connect SSH first.')
        if source_remote and session != self.session.session_id: raise ValueError('Copy the files again from the current SSH session.')
        args = (list(sources), source_remote, self.remote_folder if remote else self.folder, remote, cut)
        self.transfer_busy = True; self.update_transfer_controls(); self.show_transfer_status('Moving files…' if cut else 'Copying files…')
        def done(result):
            self.transfer_busy = False; self.update_transfer_controls(); self.refresh_folder()
            if cut:
                remaining = [s for s in sources if s not in result['completed']]
                self.file_clipboard = (remaining, source_remote, cut, session) if remaining else None
            self.show_transfer_status(f'{len(result["completed"])} file(s) completed. ' + '\n'.join(result['errors']))
        def failed(message):
            self.transfer_busy = False; self.update_transfer_controls(); self.show_transfer_status(message)
        if remote or source_remote: self.request('paste_files', args, done, failed)
        else:
            from file_clipboard import paste_files
            class Worker(QtCore.QThread):
                result = QtCore.Signal(object)
                def run(worker): worker.result.emit(paste_files(None, *args))
            self.clipboard_worker = Worker(self)
            self.clipboard_worker.result.connect(done); self.clipboard_worker.start()

    def button(self, layout, label, callback):
        button = W.QPushButton(label); button.clicked.connect(lambda: self.guard(callback)); layout.addWidget(button); return button
    def guard(self, callback):
        try: callback()
        except Exception as exc: W.QMessageBox.warning(self.viewer, 'Workspace', str(exc))
    def current(self):
        widget = self.views.currentWidget()
        return widget if isinstance(widget, TextDocument) else None

    def goto_line(self):
        doc = self.current()
        if not doc: return
        line, ok = W.QInputDialog.getInt(self.viewer, 'Go to line', 'Line number:', doc.editor.textCursor().blockNumber()+1, 1, doc.editor.blockCount())
        if ok:
            cursor = QtGui.QTextCursor(doc.editor.document().findBlockByNumber(line-1))
            doc.editor.setTextCursor(cursor); doc.editor.setFocus()

    def make_menus(self):
        bar = self.viewer.menuBar()
        menus = {'File': self.viewer.file_menu, 'Edit': self.viewer.edit_menu}
        workspace = bar.addMenu('Workspace'); ssh = bar.addMenu('SSH')
        groups = [
            (menus['File'], [('New text file', self.new_file, 'Ctrl+N'), ('Save current file', self.save_current, 'Ctrl+S'), ('Save local copy as…', self.save_copy, 'Ctrl+Shift+S')]),
            (workspace, [('Browse local folder…', self.open_folder, 'Ctrl+K'), ('Browse SSH folder…', self.open_remote_folder, ''), ('Refresh both panes', self.refresh_folder, ''), ('Reload text from disk/server', self.reload_current, ''), ('Load editor text into 3D', self.load_editor_geometry, 'Ctrl+Return')]),
            (ssh, [('Connect SSH…', self.connect_dialog, ''), ('Disconnect SSH', self.disconnect, '')]),
        ]
        self.transfer_actions = []
        for menu, entries in groups:
            if menu.actions(): menu.addSeparator()
            for label, callback, shortcut in entries:
                action = menu.addAction(label)
                action.triggered.connect(lambda checked=False, cb=callback: self.guard(cb))
                if shortcut: action.setShortcut(shortcut)
        workspace.addSeparator(); workspace.addAction(self.dock.toggleViewAction())
        find_action = menus['Edit'].addAction('Find in editor'); find_action.setShortcut('Ctrl+F')
        find_action.triggered.connect(lambda: self.current().search.setFocus() if self.current() else None)
        goto_action = menus['Edit'].addAction('Go to line…'); goto_action.setShortcut('Ctrl+G')
        goto_action.triggered.connect(self.goto_line)
        self.update_transfer_controls()

    def make_explorer(self):
        self.transfer_busy = False; self.remote_generation = 0; self.remote_folder = '.'
        self.dock = W.QDockWidget('Files & transfers', self.viewer); self.dock.setMinimumWidth(300)
        widget = W.QWidget(); layout = W.QVBoxLayout(widget); layout.setContentsMargins(8, 8, 8, 8)
        self.explorer_split = W.QSplitter(QtCore.Qt.Orientation.Vertical); layout.addWidget(self.explorer_split)
        self.local_panel = W.QGroupBox('Local • offline'); local = W.QVBoxLayout(self.local_panel)
        self.location = W.QLineEdit(self.folder); self.location.setPlaceholderText('Local folder'); local.addWidget(self.location)
        row = W.QHBoxLayout()
        self.button(row, 'Browse…', self.open_folder); self.button(row, 'Go', self.go_folder)
        self.button(row, 'New folder…', self.create_local_folder)
        self.button(row, 'Refresh', self.go_folder); local.addLayout(row)
        self.button(local, 'Back', self.back_local_folder)
        self.location.returnPressed.connect(lambda: self.guard(self.go_folder))
        self.local_model = W.QFileSystemModel(self)
        self.local_model.setFilter(QtCore.QDir.Filter.AllEntries | QtCore.QDir.Filter.NoDotAndDotDot | QtCore.QDir.Filter.Hidden)
        self.local_tree = W.QTreeView(); self.local_tree.setModel(self.local_model)
        self.local_tree.setRootIndex(self.local_model.setRootPath(self.folder))
        for col in (1, 2, 3): self.local_tree.hideColumn(col)
        self.local_tree.setSelectionMode(W.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.local_tree.setSelectionBehavior(W.QAbstractItemView.SelectionBehavior.SelectRows)
        self.local_tree.setEditTriggers(W.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.local_tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.local_tree.customContextMenuRequested.connect(lambda pos: self.show_file_context(self.local_tree, pos, False))
        self.local_tree.doubleClicked.connect(lambda index: self.guard(lambda: self.enter_folder(self.local_model.filePath(index)) if self.local_model.isDir(index) else self.open_local(self.local_model.filePath(index))))
        local.addWidget(self.local_tree)
        row = W.QHBoxLayout(); self.upload_button = self.button(row, 'Upload selected → SSH', lambda: self.transfer_selected('upload')); local.addLayout(row)
        self.explorer_split.addWidget(self.local_panel)
        self.remote_panel = W.QGroupBox('SSH • disconnected'); remote = W.QVBoxLayout(self.remote_panel)
        self.remote_location = W.QLineEdit('.'); self.remote_location.setPlaceholderText('Remote folder'); remote.addWidget(self.remote_location)
        row = W.QHBoxLayout()
        self.button(row, 'Browse…', self.open_remote_folder); self.button(row, 'Go', self.go_remote_folder)
        self.button(row, 'New folder…', self.create_remote_folder)
        self.button(row, 'Refresh', self.go_remote_folder); remote.addLayout(row)
        self.button(remote, 'Back', self.back_remote_folder)
        self.remote_location.returnPressed.connect(lambda: self.guard(self.go_remote_folder))
        self.remote_tree = W.QTreeWidget(); self.remote_tree.setHeaderLabels(['Name', 'Size'])
        self.remote_tree.setSelectionMode(W.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.remote_tree.setEditTriggers(W.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.remote_tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.remote_tree.customContextMenuRequested.connect(lambda pos: self.show_file_context(self.remote_tree, pos, True))
        self.remote_tree.itemExpanded.connect(lambda item: self.guard(lambda: self.expand_remote(item)))
        self.remote_tree.itemDoubleClicked.connect(lambda item, column: self.guard(lambda: self.activate_remote(item)))
        remote.addWidget(self.remote_tree)
        row = W.QHBoxLayout(); self.download_button = self.button(row, 'Download selected → Local', lambda: self.transfer_selected('download')); remote.addLayout(row)
        self.explorer_split.addWidget(self.remote_panel); self.remote_panel.setEnabled(False)
        self.transfer_status = W.QLabel(''); self.transfer_status.setWordWrap(True)
        self.transfer_status.setTextFormat(QtCore.Qt.TextFormat.PlainText); layout.addWidget(self.transfer_status)
        self.transfer_status.hide()
        self.transfer_progress = W.QProgressBar(); self.transfer_progress.setRange(0, 1000)
        self.transfer_progress.setFormat('%p%'); self.transfer_progress.hide(); layout.addWidget(self.transfer_progress)
        from branding import BrandingFooter
        self.branding = BrandingFooter(); layout.addWidget(self.branding)
        self.local_tree.selectionModel().selectionChanged.connect(self.update_transfer_controls)
        self.remote_tree.itemSelectionChanged.connect(self.update_transfer_controls)
        self.dock.setWidget(widget); self.viewer.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.dock)
        self.viewer.resizeDocks([self.dock], [340], QtCore.Qt.Orientation.Horizontal)

    def show_file_context(self, tree, pos, remote):
        index = tree.indexAt(pos) if not remote else None
        item = tree.itemAt(pos) if remote else None
        if remote:
            if item: tree.setCurrentItem(item)
            selected = bool(item)
        else:
            if index.isValid(): tree.setCurrentIndex(index)
            selected = index.isValid()
        menu = W.QMenu(tree)
        if selected:
            menu.addAction('View / open', lambda: self.guard(lambda: self.activate_remote(tree.currentItem()) if remote else self.open_local(self.local_model.filePath(tree.currentIndex()))))
            menu.addAction('Edit', lambda: self.guard(lambda: self.activate_remote(tree.currentItem()) if remote else self.open_local(self.local_model.filePath(tree.currentIndex()))))
            menu.addSeparator()
            menu.addAction('Copy', lambda: self.guard(lambda: self.copy_files(remote, False)))
            menu.addAction('Cut', lambda: self.guard(lambda: self.copy_files(remote, True)))
        menu.addAction('Paste into this folder', lambda: self.guard(lambda: self.paste_files(remote)))
        menu.exec(tree.viewport().mapToGlobal(pos))

    def show_transfer_status(self, message):
        self.transfer_status.setText(message)
        self.transfer_status.setVisible(bool(message))

    def show_progress(self, name, copied, total):
        percent = 100 if total <= 0 else min(100, int(copied * 100 / total))
        self.transfer_progress.setValue(percent * 10)
        self.transfer_progress.setFormat(f'{percent}% · {Path(name).name}')
        self.transfer_progress.show()

    def open_folder(self):
        path = W.QFileDialog.getExistingDirectory(self.viewer, 'Open local folder', self.folder)
        if path: self.enter_folder(path)
    def create_local_folder(self):
        name, ok = W.QInputDialog.getText(self.viewer, 'Create local folder', 'Folder name:')
        if not ok: return
        name = name.strip()
        if not name or name in ('.', '..') or Path(name).name != name: raise ValueError('Enter a valid folder name.')
        (Path(self.folder) / name).mkdir(); self.go_folder()
    def create_remote_folder(self):
        if not self.online: raise ValueError('Connect to an SSH server first.')
        name, ok = W.QInputDialog.getText(self.viewer, 'Create SSH folder', 'Folder name:')
        if not ok: return
        name = name.strip()
        if not name or name in ('.', '..') or '/' in name or '\\' in name: raise ValueError('Enter a valid folder name.')
        self.request('mkdir', (posixpath.join(self.remote_folder, name),), lambda _: self.go_remote_folder())
    def open_remote_folder(self):
        if not self.online: raise ValueError('Connect to an SSH server first.')
        dialog = RemoteFolderDialog(self)
        if dialog.exec() == W.QDialog.DialogCode.Accepted:
            self.remote_location.setText(dialog.folder); self.go_remote_folder()
        dialog.deleteLater()
    def enter_folder(self, path):
        self.location.setText(path); self.go_folder()
    def activate_remote(self, item):
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not data: return
        if data[1]:
            self.remote_location.setText(data[0]); self.go_remote_folder()
        else: self.remote_open(item)
    def go_folder(self):
        path = self.location.text().strip()
        if not Path(path).is_dir(): raise ValueError('Choose an existing local folder.')
        self.folder = str(Path(path).resolve()); self.location.setText(self.folder)
        self.local_tree.setRootIndex(self.local_model.setRootPath(self.folder))
    def back_local_folder(self):
        parent = str(Path(self.folder).parent)
        if parent != self.folder: self.enter_folder(parent)
    def back_remote_folder(self):
        current = self.remote_location.text().strip() or '.'
        parent = posixpath.dirname(current.rstrip('/')) or '/'
        if parent != current:
            self.remote_location.setText(parent); self.go_remote_folder()
    def go_remote_folder(self):
        self.remote_generation += 1; generation = self.remote_generation
        def loaded(value):
            if generation == self.remote_generation: self.show_remote(value)
        self.request('list', (self.remote_location.text().strip() or '.',), loaded)
    def refresh_folder(self):
        self.go_folder()
        if self.online: self.go_remote_folder()
    def show_remote(self, value):
        path, entries = value
        self.remote_folder = path; self.remote_location.setText(path)
        self.remote_tree.clear(); self.populate_remote(self.remote_tree.invisibleRootItem(), path, entries)

    def update_transfer_controls(self, *args):
        enabled = self.online and not self.transfer_busy
        upload = enabled and bool(self.local_tree.selectionModel().selectedRows(0))
        download = enabled and any(item.data(0, QtCore.Qt.ItemDataRole.UserRole) for item in self.remote_tree.selectedItems())
        self.upload_button.setEnabled(upload); self.download_button.setEnabled(download)
        for action, state in zip(getattr(self, 'transfer_actions', []), (upload, download)): action.setEnabled(state)

    def transfer_selected(self, direction):
        if not self.online: raise ValueError('Connect to an SSH server first.')
        if self.transfer_busy: return
        if direction == 'upload':
            sources = [self.local_model.filePath(i) for i in self.local_tree.selectionModel().selectedRows(0)]
            sources = [p for p in sources if not any(Path(q) in Path(p).parents for q in sources if q != p)]
            destination = self.remote_folder
        else:
            sources = [item.data(0, QtCore.Qt.ItemDataRole.UserRole)[0] for item in self.remote_tree.selectedItems() if item.data(0, QtCore.Qt.ItemDataRole.UserRole)]
            sources = [p for p in sources if not any(p.startswith(q.rstrip('/')+'/') for q in sources if q != p)]
            destination = self.folder
        if not sources: return
        self.transfer_busy = True; self.update_transfer_controls()
        self.show_transfer_status(f'{direction.title()} in progress: {len(sources)} item(s) → {destination}')
        def failed(message):
            self.transfer_busy = False; self.update_transfer_controls()
            self.transfer_progress.hide()
            self.show_transfer_status('Transfer interrupted: ' + message)
        def finished(result):
            self.transfer_busy = False; self.update_transfer_controls()
            self.transfer_progress.hide()
            self.show_transfer_status(f'{direction.title()}: {len(result["completed"])} completed, {len(result["errors"])} failed.')
            self.guard(self.refresh_folder)
            if result['errors']:
                box = W.QMessageBox(self.viewer); box.setWindowTitle('Transfer results')
                box.setText('Some items could not be copied. Existing destinations are never overwritten. Partially copied folders may remain.')
                box.setDetailedText('\n'.join(result['errors'])); box.exec()
        try: self.request(direction, (sources, destination), finished, failed)
        except Exception as exc: failed(str(exc))

    def populate_remote(self, parent, path, entries):
        for name, directory, size in entries:
            item = W.QTreeWidgetItem([name, '' if directory else f'{size:,}'])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, (posixpath.join(path, name), directory))
            item.setIcon(0, self.viewer.style().standardIcon(W.QStyle.StandardPixmap.SP_DirIcon if directory else W.QStyle.StandardPixmap.SP_FileIcon))
            parent.addChild(item)
            if directory: item.addChild(W.QTreeWidgetItem(['Loading…']))
    def expand_remote(self, item):
        entry = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not entry: return
        path, directory = entry
        if not directory: return
        def fill(value):
            import shiboken6
            if not shiboken6.isValid(item): return
            item.takeChildren(); self.populate_remote(item, value[0], value[1])
        self.request('list', (path,), fill)
    def remote_open(self, item):
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if data and not data[1]: self.open_remote(data[0])

    def open_dialog(self):
        path, _ = W.QFileDialog.getOpenFileName(self.viewer, 'Open text or structure', self.folder, 'All files (*);;QE files (*.pwin *.pwout *.in *.out *.pwi *.pwo)')
        if path: self.open_local(path)
    def open_local(self, path):
        path = str(Path(path).resolve())
        limit = STRUCTURE_LIMIT if Path(path).suffix.lower() in STRUCTURE_SUFFIXES | {'.cube', '.cub'} else EDITOR_LIMIT
        if Path(path).stat().st_size > limit: raise ValueError(f'File limit is {limit//1024//1024} MiB.')
        if Path(path).suffix.lower() in ('.cube', '.cub'):
            return self.open_cube(path, lambda: Path(path).read_bytes())
        self.open_document(path, Path(path).read_bytes())
    def open_remote(self, path):
        session = self.session.session_id if self.session else None
        if Path(path).suffix.lower() in ('.cube', '.cub'):
            for cube in self.cubes:
                if cube.path == path and cube.remote_session == session:
                    self.views.setCurrentWidget(cube); return
            cube = self.create_cube(path, session)
            def failed(message):
                cube.info.setText(message)
                self.remove_cube(cube)
                W.QMessageBox.warning(self.viewer, 'Open cube', message)
            try:
                self.request('geometry', (path,), lambda data: cube.load(data) if not cube.disposed else None, failed)
            except Exception:
                self.remove_cube(cube); raise
        else:
            operation = 'geometry' if Path(path).suffix.lower() in STRUCTURE_SUFFIXES else 'read'
            self.request(operation, (path,), lambda data: self.open_document(path, data, session))
    def create_cube(self, path, session=None):
        from cube_panel import CubePanel
        cube = CubePanel(self, path, session)
        self.cubes.append(cube); self.views.addTab(cube, self.title(cube)); self.views.setCurrentWidget(cube)
        cube.changed.connect(lambda: self.update_title(cube))
        return cube
    def open_cube(self, path, source, session=None):
        for cube in self.cubes:
            if cube.path == path and cube.remote_session == session:
                self.views.setCurrentWidget(cube); return cube
        cube = self.create_cube(path, session); cube.load(source); return cube
    def refresh_cube_operands(self):
        for cube in self.cubes: cube.refresh_operands()
    def remove_cube(self, cube):
        if cube not in self.cubes: return
        self.views.removeTab(self.views.indexOf(cube)); self.cubes.remove(cube)
        cube.dispose(); cube.deleteLater(); self.refresh_cube_operands()
    def save_cube(self, cube, copy=False):
        if cube.field is None: raise ValueError('Wait for the cube to load.')
        if cube.busy or cube.save_pending: raise ValueError('Wait for the current cube operation.')
        path = cube.path
        if copy:
            path, _ = W.QFileDialog.getSaveFileName(self.viewer, 'Save cube copy', str(Path(self.folder)/Path(path).name), 'Cube (*.cube *.cub)')
            if not path: return
        elif cube.remote_session and (not self.online or self.session.session_id != cube.remote_session):
            raise ValueError('This cube belongs to a disconnected session. Save a local copy or reopen it.')
        cube.save_pending = True
        def serialized(data):
            def saved(value):
                cube.save_pending = False
                cube.controls.setEnabled(True)
                if not copy: cube.expected = value; cube.modified = False; cube.changed.emit()
                cube.info.setText('Saved cube: ' + path)
            def failed(message):
                cube.save_pending = False; cube.controls.setEnabled(True)
                W.QMessageBox.warning(self.viewer, 'Save cube', message)
            try:
                if cube.remote_session and not copy:
                    self.request('save_cube', (path, data, cube.expected), saved, failed)
                else:
                    if not copy and (not Path(path).exists() or Path(path).stat().st_size > STRUCTURE_LIMIT or digest(Path(path).read_bytes()) != cube.expected):
                        raise ValueError('Local cube changed since opening. Save a copy instead.')
                    self.atomic_local(path, data); saved(digest(data))
            except Exception as exc: failed(str(exc))
        # Serialization is done off the GUI thread as cube files can be large.
        def serialize():
            return cube.field.payload()
        cube.run(serialize, serialized)
    def open_document(self, path, data, session=None):
        if Path(path).suffix.lower() in ('.cube', '.cub'): return self.open_cube(path, data, session)
        if len(data) > EDITOR_LIMIT:
            self.geometry_bytes(path, data); return
        for doc in self.documents:
            if doc.path == path and doc.remote_session == session:
                self.views.setCurrentWidget(doc); return doc
        try: doc = TextDocument(path, data, session)
        except UnicodeDecodeError:
            encoding, ok = W.QInputDialog.getItem(self.viewer, 'Text encoding', 'File is not UTF-8. Choose an encoding:', ['cp1252', 'latin-1', 'utf-16-le', 'utf-16-be'], 0, False)
            if not ok: return
            doc = TextDocument(path, data, session, encoding)
        self.documents.append(doc); self.views.addTab(doc, self.title(doc)); self.views.setCurrentWidget(doc)
        doc.save_requested.connect(lambda d=doc: self.guard(lambda: self.save_document(d)))
        doc.save_as_requested.connect(lambda d=doc: self.guard(lambda: self.save_copy(d, adopt=True)))
        if Path(path).suffix.lower() in STRUCTURE_SUFFIXES or Path(path).name.upper() in ('POSCAR', 'CONTCAR'):
            button = W.QPushButton('View this file in 3D • choose optimization step')
            button.clicked.connect(lambda: self.guard(lambda: self.geometry_bytes(doc.path, doc.payload())))
            doc.layout().insertWidget(1, button)
        doc.editor.document().modificationChanged.connect(lambda _: self.update_title(doc))
        # Geometry is loaded explicitly from this document, preserving the current 3D edits.
        return doc
    def title(self, doc): return ('● ' if doc.dirty() else '') + ('SSH: ' if doc.remote_session else '') + (Path(doc.path).name or 'Untitled')
    def update_title(self, doc):
        index = self.views.indexOf(doc)
        if index >= 0: self.views.setTabText(index, self.title(doc))
    def new_file(self):
        doc = TextDocument('', b''); self.documents.append(doc)
        doc.save_requested.connect(lambda d=doc: self.guard(lambda: self.save_document(d)))
        doc.save_as_requested.connect(lambda d=doc: self.guard(lambda: self.save_copy(d, adopt=True)))
        self.views.addTab(doc, 'Untitled'); self.views.setCurrentWidget(doc)
        doc.editor.document().modificationChanged.connect(lambda _: self.update_title(doc))

    def save_current(self):
        if self.views.currentWidget() in self.cubes: return self.save_cube(self.views.currentWidget())
        doc = self.current()
        if not doc: raise ValueError('Select a text editor tab to save. Use Export for structures.')
        self.save_document(doc)
    def save_document(self, doc):
        if doc.save_pending: raise ValueError('A save for this document is already in progress.')
        data = doc.payload()
        if len(data) > EDITOR_LIMIT: raise ValueError('Text editor limit is 20 MiB.')
        if doc.remote_session:
            if not self.online or self.session.session_id != doc.remote_session:
                raise ValueError('This file belongs to a disconnected SSH session. Save a local copy, or reopen it from the new session.')
            doc.save_pending = True
            def saved(value):
                doc.save_pending = False; doc.expected = value
                if doc.payload() == data: doc.editor.document().setModified(False)
                self.update_title(doc); self.viewer.statusBar().showMessage('Saved remote file: ' + doc.path)
            def failed(message):
                doc.save_pending = False; W.QMessageBox.warning(self.viewer, 'Remote save', message)
            self.request('save', (doc.path, data, doc.expected), saved, failed)
        else:
            if not doc.path: return self.save_copy(doc, adopt=True)
            path = Path(doc.path)
            if not path.exists() or path.stat().st_size > EDITOR_LIMIT or digest(path.read_bytes()) != doc.expected:
                raise ValueError('Local file changed or was removed since opening. Reload it or save a copy under a new name.')
            self.atomic_local(doc.path, data)
            doc.expected = digest(data); doc.editor.document().setModified(False); self.update_title(doc)
            self.viewer.statusBar().showMessage('Saved ' + doc.path)
    @staticmethod
    def atomic_local(path, data):
        output = QtCore.QSaveFile(path)
        if not output.open(QtCore.QIODevice.OpenModeFlag.WriteOnly): raise OSError(output.errorString())
        if output.write(data) != len(data): output.cancelWriting(); raise OSError(output.errorString())
        if not output.commit(): raise OSError(output.errorString())
    def save_copy(self, doc=None, adopt=False):
        if doc is None and self.views.currentWidget() in self.cubes: return self.save_cube(self.views.currentWidget(), copy=True)
        doc = doc or self.current()
        if not doc: raise ValueError('Select a text editor tab.')
        path, _ = W.QFileDialog.getSaveFileName(self.viewer, 'Save local text copy', str(Path(self.folder)/(Path(doc.path).name or 'untitled.pwin')), 'All files (*)')
        if not path: return
        data = doc.payload(); self.atomic_local(path, data)
        if adopt or not doc.remote_session:
            doc.path = str(Path(path).resolve()); doc.remote_session = None; doc.expected = digest(data)
            doc.path_label.setText(doc.path); doc.editor.document().setModified(False); self.update_title(doc)
        self.viewer.statusBar().showMessage('Saved local copy: ' + path)
    def reload_current(self):
        doc = self.current()
        if not doc: return
        if not self.allow_discard(doc): return
        path, session = doc.path, doc.remote_session
        if session:
            if not self.online or self.session.session_id != session: raise ValueError('Reconnect and reopen this file to reload it.')
            before = doc.payload()
            def loaded(data):
                if doc not in self.documents: return
                if doc.payload() != before: raise ValueError('Text changed while reloading. The current edits have been kept.')
                self.remove_document(doc); self.open_document(path, data, session)
            self.request('read', (path,), loaded)
        else:
            if Path(path).stat().st_size > EDITOR_LIMIT: raise ValueError('Text editor limit is 20 MiB.')
            data = Path(path).read_bytes(); self.remove_document(doc); self.open_document(path, data)
    def allow_discard(self, doc):
        if doc.save_pending:
            W.QMessageBox.information(self.viewer, 'Save in progress', 'Wait for this file to finish saving.'); return False
        if not doc.dirty(): return True
        return W.QMessageBox.question(self.viewer, 'Unsaved file', f'Discard unsaved edits in {Path(doc.path).name or "Untitled"}?', W.QMessageBox.StandardButton.Discard | W.QMessageBox.StandardButton.Cancel, W.QMessageBox.StandardButton.Cancel) == W.QMessageBox.StandardButton.Discard
    def close_tab(self, index):
        if index == 0: return
        doc = self.views.widget(index)
        if doc in self.cubes and doc.busy: return
        if self.allow_discard(doc):
            if doc in self.cubes: self.remove_cube(doc)
            else: self.remove_document(doc)
    def remove_document(self, doc):
        self.views.removeTab(self.views.indexOf(doc)); self.documents.remove(doc); doc.deleteLater()

    def geometry_bytes(self, name, data, show=True):
        if not self.viewer.confirm_discard(): return
        suffix = Path(name).suffix or ('.vasp' if Path(name).name.upper() in ('POSCAR', 'CONTCAR') else '.txt')
        path = Path(self.temp.name)/(digest(data)+suffix)
        path.write_bytes(data); self.viewer.load_path(path)
        self.views.setTabText(0, '3D: ' + Path(name).name)
        self.views.setTabToolTip(0, name)
        self.viewer.statusBar().showMessage('3D loaded from ' + name + ' — final coordinate frame selected.')
        if show: self.views.setCurrentIndex(0)
    def load_editor_geometry(self):
        doc = self.current()
        if not doc: raise ValueError('Open a QE or other structure file in a text tab first.')
        self.geometry_bytes(doc.path or 'untitled.pwin', doc.payload())
    def connect_dialog(self):
        if self.session and self.session.isRunning(): raise ValueError('Disconnect the current SSH session first.')
        dialog = ConnectDialog(self.viewer)
        try:
            if dialog.exec() != W.QDialog.DialogCode.Accepted: return
            config = dialog.config()
        finally:
            dialog.password.clear(); dialog.passphrase.clear(); dialog.deleteLater()
        self.connect_config(config)
    def connect_config(self, config, known_hosts=None):
        if self.session and self.session.isRunning(): raise ValueError('SSH session is already active.')
        if known_hosts is None:
            known_hosts = Path(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.AppDataLocation))/'StructureStudio'/'known_hosts'
        self.session = SSHSession(config, known_hosts, self)
        session = self.session
        self.endpoint = f'{config["user"]}@{config["host"]}:{config["port"]}'
        self.connection_label.setText('Connecting: ' + self.endpoint)
        self.session.host_key.connect(lambda *args: self.trust_host(*args) if self.session is session else session.trust(False))
        self.session.auth_prompt.connect(lambda *args: self.prompt_auth(session, *args))
        self.session.connected.connect(lambda *args: self.on_connected(*args) if self.session is session else None)
        self.session.disconnected.connect(lambda *args: self.on_disconnected(*args) if self.session is session else None)
        self.session.result.connect(lambda *args: self.on_result(*args) if self.session is session else None)
        self.session.failed.connect(lambda *args: self.on_failed(*args) if self.session is session else None)
        self.session.transfer_progress.connect(lambda token, name, copied, total: self.show_progress(name, copied, total))
        self.session.start()
    def prompt_auth(self, session, title, instructions, prompts):
        if session is not self.session or session.stopping.is_set():
            session.answer_auth(None); return
        saved_password = session.config.get('password', '')
        password_prompts = [bool(saved_password and not echo and 'password' in prompt.lower()) for prompt, echo in prompts]
        if prompts and all(password_prompts):
            session.answer_auth([saved_password] * len(prompts)); return
        dialog = W.QDialog(self.viewer); dialog.setWindowTitle('SSH login — ' + self.endpoint)
        dialog.setMinimumWidth(420); layout = W.QVBoxLayout(dialog)
        for text in (title, instructions):
            if text:
                label = W.QLabel(text); label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
                label.setWordWrap(True); layout.addWidget(label)
        form = W.QFormLayout(); layout.addLayout(form); fields = []
        for prompt_index, (prompt, echo) in enumerate(prompts):
            label = W.QLabel(prompt); label.setTextFormat(QtCore.Qt.TextFormat.PlainText); label.setWordWrap(True)
            field = W.QLineEdit()
            field.setEchoMode(W.QLineEdit.EchoMode.Normal if echo else W.QLineEdit.EchoMode.Password)
            if password_prompts[prompt_index]: field.setText(saved_password)
            form.addRow(label, field); fields.append(field)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Ok | W.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); layout.addWidget(buttons)
        session.disconnected.connect(dialog.reject)
        accepted = dialog.exec() == W.QDialog.DialogCode.Accepted
        session.disconnected.disconnect(dialog.reject)
        session.answer_auth([field.text() for field in fields] if accepted else None)
        for field in fields: field.clear()
        dialog.deleteLater()
    def trust_host(self, host, algorithm, fingerprint):
        answer = W.QMessageBox.question(self.viewer, 'Verify SSH host key', f'Unknown SSH host: {host}\nKey type: {algorithm}\nFingerprint: {fingerprint}\n\nCompare this fingerprint with your server administrator. Trust this key and save it for future connections?', W.QMessageBox.StandardButton.Yes | W.QMessageBox.StandardButton.No, W.QMessageBox.StandardButton.No)
        self.session.trust(answer == W.QMessageBox.StandardButton.Yes)
    def on_connected(self, folder):
        self.online = True; self.terminal.online = True; self.terminal.reset()
        self.terminal.reader = self.session.take_output
        cw, ch = self.terminal.metrics(); self.session.resize_terminal(max(20, self.terminal.width()//cw), max(4, self.terminal.height()//ch))
        self.connection_label.setText('SSH: ' + self.endpoint)
        self.remote_folder = folder; self.remote_location.setText(folder)
        self.remote_panel.setTitle('SSH • ' + self.endpoint); self.remote_panel.setEnabled(True)
        self.update_transfer_controls(); self.guard(self.go_remote_folder); self.terminal.setFocus()
    def disconnect(self):
        if self.session: self.session.stop()
        self.online = False; self.terminal.online = False
        self.remote_panel.setEnabled(False); self.remote_panel.setTitle('SSH • disconnected')
        self.remote_generation += 1; self.update_transfer_controls()
        self.remote_panel.setEnabled(False); self.remote_panel.setTitle('SSH • disconnected')
        self.remote_generation += 1; self.update_transfer_controls()
        self.connection_label.setText('SSH: disconnecting…' if self.session and self.session.isRunning() else 'SSH: disconnected')
    def on_disconnected(self, message):
        self.online = False; self.terminal.online = False
        self.remote_panel.setEnabled(False); self.remote_panel.setTitle('SSH • disconnected')
        self.remote_generation += 1; self.update_transfer_controls()
        self.remote_panel.setEnabled(False); self.remote_panel.setTitle('SSH • disconnected')
        self.remote_generation += 1; self.update_transfer_controls()
        self.connection_label.setText('SSH: disconnected — ' + message)
        self.terminal.reader = None
        self.terminal.feed(('\r\n[Disconnected: ' + message + ']\r\n').encode('utf-8'))
        for _, (_, failed) in list(self.pending.items()):
            if failed: failed('SSH connection ended; the operation was not confirmed. Check the destination before retrying.')
        self.pending.clear()
        for doc in self.documents + self.cubes:
            if doc.remote_session: doc.save_pending = False
    def request(self, operation, args, callback, failed=None):
        if not self.online or not self.session: raise ValueError('Connect to an SSH server first.')
        token = self.session.request(operation, *args); self.pending[token] = (callback, failed); return token
    def on_result(self, token, operation, value):
        callbacks = self.pending.pop(token, None)
        if callbacks: self.guard(lambda: callbacks[0](value))
    def on_failed(self, token, operation, message):
        callbacks = self.pending.pop(token, None)
        if callbacks and callbacks[1]: callbacks[1](message)
        else: W.QMessageBox.warning(self.viewer, 'SSH ' + operation, message)
    def can_close(self):
        if self.clipboard_worker is not None and self.clipboard_worker.isRunning():
            W.QMessageBox.information(self.viewer, 'File transfer', 'Wait for the local file transfer to finish.'); return False
        if any(cube.busy for cube in self.cubes):
            W.QMessageBox.information(self.viewer, 'Cube busy', 'Wait for the cube operation to finish.'); return False
        return all(self.allow_discard(doc) for doc in list(self.documents) + list(self.cubes))
    def shutdown(self):
        if hasattr(self, 'analysis_window'): self.analysis_window.dispose()
        for cube in self.cubes: cube.dispose()
        self.disconnect()
        return self.session is None or not self.session.isRunning()
