"""Local/SSH analysis workflow with embedded, editable Matplotlib figures."""
import copy
import json
import pickle
import posixpath
import sys
import tempfile
from pathlib import Path
from PySide6 import QtCore, QtWidgets as W
from analysis_controls import AnalysisControls, DEFINITIONS
from band_options import ColorPicker


class AnalysisWindow(W.QMainWindow):
    def __init__(self, workspace):
        super().__init__(workspace.viewer)
        self.workspace = workspace
        self.setWindowTitle('Analysis studio — Quantum ESPRESSO'); self.resize(1280, 850)
        self.storage = tempfile.TemporaryDirectory(prefix='qe-analysis-')
        self.inputs = []; self.generation = 0; self.process = None; self.loading_form = False
        self.figure_widgets = []
        self.ssh_directory = getattr(workspace, 'remote_folder', '.')
        root = W.QSplitter(); self.setCentralWidget(root)
        left = W.QWidget(); layout = W.QVBoxLayout(left); root.addWidget(left)
        self.choice = W.QComboBox(); self.choice.addItems(DEFINITIONS)
        layout.addWidget(W.QLabel('1. Choose analysis')); layout.addWidget(self.choice)
        self.help = W.QLabel(); self.help.setWordWrap(True); layout.addWidget(self.help)
        directories = W.QFormLayout(); layout.addLayout(directories)
        row = W.QHBoxLayout(); self.work_directory = W.QLineEdit(str(Path(workspace.folder).resolve()))
        self.work_directory.setToolTip('Local folder where analysis runs and relative output filenames are saved.')
        row.addWidget(self.work_directory); button = W.QPushButton('Browse…'); button.clicked.connect(self.browse_work_directory); row.addWidget(button)
        directories.addRow('Working / output folder (local)', row)
        row = W.QHBoxLayout(); self.ssh_location = W.QLineEdit(self.ssh_directory); self.ssh_location.setReadOnly(True)
        row.addWidget(self.ssh_location)
        directories.addRow('Analysis input folder (SSH)', row)
        layout.addWidget(W.QLabel('2. Input files'))
        self.files = W.QListWidget(); self.files.setMaximumHeight(125); layout.addWidget(self.files)
        buttons = W.QHBoxLayout(); layout.addLayout(buttons)
        for text, fn in [('Local…', self.add_local), ('Browse SSH files…', self.add_remote_path), ('View file', self.view_selected_file), ('Remove', self.remove_file)]:
            b = W.QPushButton(text); b.clicked.connect(fn); buttons.addWidget(b)
        layout.addWidget(W.QLabel('3. Class options'))
        self.controls = AnalysisControls(); layout.addWidget(self.controls, 1)
        self.controls.changed.connect(self.schedule)
        row = W.QHBoxLayout(); layout.addLayout(row)
        self.live = W.QCheckBox('Live preview'); self.live.setChecked(True); row.addWidget(self.live)
        self.watch = W.QCheckBox('Refresh inputs every 5 s'); row.addWidget(self.watch)
        b = W.QPushButton('Refresh / Run'); b.clicked.connect(self.refresh); row.addWidget(b)
        b = W.QPushButton('Stop'); b.clicked.connect(self.cancel); row.addWidget(b)
        right = W.QWidget(); rl = W.QVBoxLayout(right); root.addWidget(right)
        self.result_tabs = W.QTabWidget(); rl.addWidget(self.result_tabs, 1)
        self.figures = W.QTabWidget(); self.result_tabs.addTab(self.figures, 'Plots')
        self.file_tab = W.QWidget(); file_layout = W.QVBoxLayout(self.file_tab)
        self.file_choice = W.QComboBox(); file_layout.addWidget(self.file_choice)
        row = W.QHBoxLayout(); self.file_search = W.QLineEdit(); self.file_search.setPlaceholderText('Find text in file…'); row.addWidget(self.file_search)
        find = W.QPushButton('Find next'); row.addWidget(find); file_layout.addLayout(row)
        from text_editor import CodeEditor, QEHighlighter
        self.file_text = CodeEditor(); self.file_text.setReadOnly(True); self.file_highlighter = QEHighlighter(self.file_text.document()); file_layout.addWidget(self.file_text, 1)
        self.file_status = W.QLabel(); self.file_status.setWordWrap(True); self.file_status.setTextFormat(QtCore.Qt.TextFormat.PlainText); file_layout.addWidget(self.file_status)
        use_point = W.QPushButton('Add selected number as high-symmetry point'); use_point.clicked.connect(self.use_selected_point); file_layout.addWidget(use_point)
        self.result_tabs.addTab(self.file_tab, 'File contents'); self.preview_generation = 0
        self.file_choice.currentIndexChanged.connect(self.preview_file)
        self.result_tabs.currentChanged.connect(lambda: self.preview_file() if self.result_tabs.currentWidget() is self.file_tab else None)
        find.clicked.connect(self.find_file_text); self.file_search.returnPressed.connect(self.find_file_text)
        self.controls.band_options.view_file.connect(self.view_file)
        style = W.QHBoxLayout(); rl.addLayout(style)
        self.color = ColorPicker('blue'); self.override_color = W.QCheckBox('Override colors')
        self.width = W.QDoubleSpinBox(); self.width.setRange(0.1, 20); self.width.setValue(1.5)
        self.title_text = W.QLineEdit(); self.title_text.setPlaceholderText('Figure title')
        style.addWidget(self.override_color); style.addWidget(self.color); style.addWidget(W.QLabel('Line width')); style.addWidget(self.width); style.addWidget(self.title_text)
        b = W.QPushButton('Apply style'); b.clicked.connect(self.apply_style); style.addWidget(b)
        save = W.QHBoxLayout(); rl.addLayout(save)
        self.dpi = W.QSpinBox(); self.dpi.setRange(72, 2400); self.dpi.setValue(300)
        self.transparent = W.QCheckBox('Transparent'); save.addStretch(); save.addWidget(W.QLabel('Export DPI')); save.addWidget(self.dpi); save.addWidget(self.transparent)
        b = W.QPushButton('Save figure…'); b.clicked.connect(self.save_figure); save.addWidget(b)
        self.save_pdos_button = W.QPushButton('Save summed PDOS…'); self.save_pdos_button.clicked.connect(self.save_summed_pdos); save.addWidget(self.save_pdos_button); self.save_pdos_button.setEnabled(False)
        note = W.QLabel('Each figure toolbar provides zoom, pan, axes/curve editing and subplot layout. Class parameters control plot-specific options.'); note.setWordWrap(True); rl.addWidget(note)
        self.output = W.QPlainTextEdit(); self.output.setReadOnly(True); self.output.setMaximumHeight(180); rl.addWidget(self.output)
        root.setSizes([510, 770])
        self.debounce = QtCore.QTimer(self); self.debounce.setSingleShot(True); self.debounce.setInterval(650); self.debounce.timeout.connect(self.refresh)
        self.poll = QtCore.QTimer(self); self.poll.setInterval(5000); self.poll.timeout.connect(self.poll_inputs); self.poll.start()
        self.choice.currentTextChanged.connect(self.select_analysis)
        self.live.toggled.connect(lambda checked: self.schedule() if checked else self.debounce.stop())
        self.work_directory.textChanged.connect(lambda: self.schedule())
        self.select_analysis(self.choice.currentText())

    def message(self, text): self.statusBar().showMessage(text)
    def schedule(self):
        self.generation += 1  # Invalidate a fetch or calculation as soon as an option changes.
        if self.live.isChecked() and self.controls.automatic(): self.debounce.start()
        else: self.debounce.stop()

    def select_analysis(self, name):
        self.controls.configure(name)
        self.help.setText(DEFINITIONS[name][0] + ' — choose input files and adjust the class options below. Lists and ranges use Python notation; text needs no quotes.')
        if name in ('Band structure', 'Band gap'): self.help.setText('Select band data to find the global valence maximum and conduction minimum, with direct/indirect classification. Default: energies relative to Ef=0; optional SCF/manual reference for absolute energies.')
        if name == 'Projected bands':
            self.help.setText('ProjectedBandPlotter — select projected-band JSON data. The Qepy XML loader was not included in the supplied code.')
        if name in ('DOS / PDOS', 'Shifted PDOS'): self.help.setText('Select QE PDOS files together. They are sorted and grouped by element/orbital automatically. Choose atom or orbital components, or use the script’s two filename filters to calculate matching sums.')
        self.schedule()

    def add_input(self, path, remote=False):
        session = self.workspace.session.session_id if remote and self.workspace.online else None
        if remote and session is None: self.message('Connect SSH in the main window first.'); return
        if remote and not posixpath.isabs(path): path = posixpath.join(self.ssh_directory, path)
        if not remote:
            path = Path(path).expanduser()
            if not path.is_absolute(): path = self.working_directory() / path
            path = str(path.resolve())
        item = {'path': path, 'remote': remote, 'session': session}
        if item not in self.inputs: self.inputs.append(item)
        self.update_inputs()

    def update_inputs(self):
        self.files.clear()
        for i, item in enumerate(self.inputs): self.files.addItem(f'{i + 1}: {"SSH" if item["remote"] else "Local"} • {item["path"]}')
        self.controls.set_inputs(self.inputs)
        self.file_choice.blockSignals(True); previous = self.file_choice.currentData(); self.file_choice.clear()
        for i, item in enumerate(self.inputs): self.file_choice.addItem(item['path'], i)
        self.file_choice.setCurrentIndex(min(previous or 0, len(self.inputs)-1)); self.file_choice.blockSignals(False)
        self.preview_generation += 1
        if self.result_tabs.currentWidget() is self.file_tab: self.preview_file()
        self.schedule()

    def view_selected_file(self): self.view_file(self.files.currentRow())

    def view_file(self, index):
        if index is None or not 0 <= index < len(self.inputs): self.message('Select an input file to view.'); return
        self.result_tabs.blockSignals(True); self.result_tabs.setCurrentWidget(self.file_tab); self.result_tabs.blockSignals(False)
        self.file_choice.blockSignals(True); self.file_choice.setCurrentIndex(index); self.file_choice.blockSignals(False)
        self.preview_file()

    def preview_file(self, *args):
        self.preview_generation += 1; generation = self.preview_generation
        index = self.file_choice.currentData(); self.file_text.clear()
        if index is None: self.file_status.setText('Select a file to inspect.'); return
        item = dict(self.inputs[index]); self.file_status.setText('Loading ' + item['path'])
        def failed(message):
            if generation == self.preview_generation: self.file_status.setText(message)
        def loaded(data):
            if generation != self.preview_generation: return
            if item['remote'] and (not self.workspace.online or self.workspace.session.session_id != item['session']):
                failed('SSH connection changed. Select the file again.'); return
            from text_editor import decode_text
            try: text, _, _ = decode_text(data)
            except UnicodeDecodeError: text = data.decode('utf-8', errors='replace')
            except ValueError as exc: failed(str(exc)); return
            self.file_text.setPlainText(text); self.file_status.setText(item['path'] + ' — read-only; select a coordinate to add a symmetry point.')
        try:
            if item['remote']:
                if not self.workspace.online or self.workspace.session.session_id != item['session']: raise ValueError('Reconnect and select the SSH file again.')
                self.workspace.request('read', (item['path'],), loaded, failed)
            else:
                from ssh_session import EDITOR_LIMIT
                with open(item['path'], 'rb') as stream: data = stream.read(EDITOR_LIMIT + 1)
                if len(data) > EDITOR_LIMIT: raise ValueError('File preview supports text up to 20 MiB.')
                loaded(data)
        except Exception as exc: failed(str(exc))

    def find_file_text(self):
        text = self.file_search.text()
        if text and not self.file_text.find(text):
            cursor = self.file_text.textCursor(); cursor.movePosition(cursor.MoveOperation.Start); self.file_text.setTextCursor(cursor); self.file_text.find(text)

    def use_selected_point(self):
        try: value = float(self.file_text.textCursor().selectedText().strip().replace('D', 'E').replace('d', 'e'))
        except ValueError: self.message('Select one numeric k-path coordinate in the file first.'); return
        label, ok = W.QInputDialog.getText(self, 'High-symmetry point', 'Point label:', text='Γ')
        if ok and label.strip(): self.controls.band_options.add_point(label=label.strip(), coordinate=value)

    def add_local(self):
        paths, _ = W.QFileDialog.getOpenFileNames(self, 'Analysis input files', self.work_directory.text(), 'All files (*)')
        for path in paths: self.add_input(path)

    def add_remote_selection(self):
        for item in self.workspace.remote_tree.selectedItems():
            value = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if value and not value[1]: self.add_input(value[0], True)

    def add_remote_path(self):
        dialog = self.ssh_browser()
        if dialog is None: return
        if dialog.exec() == W.QDialog.DialogCode.Accepted:
            self.set_ssh_directory(dialog.folder)
            for path in dialog.paths: self.add_input(path, True)
        dialog.deleteLater()

    def ssh_browser(self, folders_only=False):
        if not self.workspace.online:
            self.message('Connect SSH in the main window first.'); return None
        from analysis_browser import SSHFileDialog
        return SSHFileDialog(self.workspace, self, self.ssh_directory, folders_only)

    def set_ssh_directory(self, path):
        self.ssh_directory = path; self.ssh_location.setText(path)

    def browse_ssh_directory(self):
        dialog = self.ssh_browser(folders_only=True)
        if dialog is None: return
        if dialog.exec() == W.QDialog.DialogCode.Accepted: self.set_ssh_directory(dialog.folder)
        dialog.deleteLater()

    def browse_work_directory(self):
        path = W.QFileDialog.getExistingDirectory(self, 'Analysis working and output folder', self.work_directory.text())
        if path: self.work_directory.setText(path)

    def working_directory(self):
        path = Path(self.work_directory.text().strip()).expanduser()
        if not self.work_directory.text().strip() or not path.is_absolute() or not path.is_dir():
            raise ValueError('Choose an existing absolute local working/output folder.')
        return path.resolve()

    def remove_file(self):
        index = self.files.currentRow()
        if index >= 0: self.inputs.pop(index); self.update_inputs()

    def poll_inputs(self):
        if self.controls.automatic() and self.watch.isChecked() and self.isVisible() and not self.debounce.isActive() and not getattr(self, 'fetching', False) and (self.process is None or self.process.state() == QtCore.QProcess.ProcessState.NotRunning): self.refresh()

    def cancel(self):
        self.generation += 1; self.fetching = False; self.debounce.stop()
        if self.process is not None and self.process.state() != QtCore.QProcess.ProcessState.NotRunning: self.process.kill()
        self.message('Stopped; last preview retained.')

    def refresh(self):
        try: self.working_directory()
        except ValueError as exc: self.message(str(exc)); return
        try: recipe = self.controls.build()
        except (ValueError, TypeError) as exc: self.message(str(exc)); return
        self.cancel(); self.generation += 1; generation = self.generation; self.fetching = True
        self.message('Loading analysis inputs…'); files = []
        inputs = copy.deepcopy(self.inputs)
        def failed(message):
            if generation == self.generation: self.fetching = False; self.message('Input error: ' + message)
        def next_file(index):
            if generation != self.generation: return
            if index == len(inputs):
                self.fetching = False; self.launch(recipe, files, generation); return
            item = inputs[index]
            if not item['remote']:
                if not Path(item['path']).is_file(): failed('File not found: ' + item['path']); return
                files.append(item['path']); next_file(index+1); return
            if not self.workspace.online or self.workspace.session.session_id != item['session']:
                failed('SSH session changed. Add the remote file again.'); return
            def loaded(data):
                if generation != self.generation: return
                directory = Path(self.storage.name) / f'input-{index}'; directory.mkdir(exist_ok=True)
                # Preserve the real basename for label inference and format detection.
                path = directory / Path(item['path']).name; path.write_bytes(data); files.append(str(path)); next_file(index+1)
            try: self.workspace.request('geometry', (item['path'],), loaded, failed)
            except Exception as exc: failed(str(exc))
        next_file(0)

    def launch(self, recipe, files, generation):
        request = Path(self.storage.name)/f'request-{generation}.json'; result = Path(self.storage.name)/f'result-{generation}.pickle'
        request.write_text(json.dumps({'recipe': recipe, 'files': files}), encoding='utf-8')
        process = QtCore.QProcess(self); self.process = process
        process.setWorkingDirectory(str(self.working_directory()))
        process.setProgram(sys.executable); process.setArguments([str(Path(__file__).with_name('analysis_engine.py')), str(request), str(result)])
        process.finished.connect(lambda code, status: self.finished(process, code, result, generation))
        process.errorOccurred.connect(lambda error: self.message(process.errorString()) if generation == self.generation else None)
        self.message('Calculating preview…'); process.start()

    def finished(self, process, code, result, generation):
        error = bytes(process.readAllStandardError()).decode('utf-8', errors='replace'); process.deleteLater()
        if self.process is process: self.process = None
        if generation != self.generation: return
        if code or not result.exists():
            self.output.setPlainText(error or 'Analysis process did not produce a result.'); self.message('Analysis failed — previous preview retained.'); return
        try:
            # Only deserialize output from our own child process in our private temporary directory.
            with result.open('rb') as stream: figures, output = pickle.load(stream)
            if figures and hasattr(figures[0], '_band_result'): self.controls.band_options.set_result(figures[0]._band_result)
            self.show_figures(figures); self.output.setPlainText(output or 'Analysis completed.'); self.message('Preview updated.')
        except Exception as exc: self.message(str(exc))

    def show_figures(self, figures):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        for widget in self.figure_widgets: widget.deleteLater()
        self.figures.clear(); self.figure_widgets = []
        for i, figure in enumerate(figures):
            widget = W.QWidget(); layout = W.QVBoxLayout(widget)
            canvas = FigureCanvasQTAgg(figure); toolbar = NavigationToolbar2QT(canvas, widget)
            layout.addWidget(toolbar); layout.addWidget(canvas, 1); widget.canvas = canvas
            self.figures.addTab(widget, f'Figure {i+1}'); self.figure_widgets.append(widget); canvas.draw_idle()
        self.save_pdos_button.setEnabled(any(hasattr(fig, '_pdos_tables') for fig in figures))
        if self.override_color.isChecked() or self.title_text.text(): self.apply_style()

    def apply_style(self):
        try:
            colors = [self.color.value()] if self.override_color.isChecked() else []
            for widget in self.figure_widgets:
                for ax in widget.canvas.figure.axes:
                    for i, line in enumerate(ax.lines):
                        line.set_linewidth(self.width.value())
                        if colors: line.set_color(colors[i % len(colors)])
                    for i, collection in enumerate(ax.collections):
                        if colors: collection.set_color(colors[i % len(colors)])
                    if ax.get_legend() is not None: ax.legend()
                    if self.title_text.text(): ax.set_title(self.title_text.text())
                widget.canvas.draw_idle()
        except Exception as exc: self.message(str(exc))

    def save_figure(self):
        widget = self.figures.currentWidget()
        if widget is None: self.message('Run a plotting method first.'); return
        path, _ = W.QFileDialog.getSaveFileName(self, 'Save figure', str(Path(self.work_directory.text())/'analysis.png'), 'PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tiff);;JPEG (*.jpg)')
        if path:
            try: widget.canvas.figure.savefig(path, dpi=self.dpi.value(), transparent=self.transparent.isChecked(), bbox_inches='tight'); self.message('Saved ' + path)
            except Exception as exc: self.message(str(exc))

    def save_summed_pdos(self):
        widget = self.figures.currentWidget()
        tables = getattr(widget.canvas.figure, '_pdos_tables', []) if widget else []
        if not tables: self.message('Run a PDOS plot first.'); return
        folder = W.QFileDialog.getExistingDirectory(self, 'Save summed PDOS data', self.work_directory.text())
        if not folder: return
        try:
            from pdos_data import export_groups
            paths = export_groups(tables, folder)
            self.message(f'Saved {len(paths)} summed PDOS file(s) to {folder}')
        except Exception as exc: self.message(str(exc))

    def closeEvent(self, event):
        self.preview_generation += 1
        self.cancel()
        for process in self.findChildren(QtCore.QProcess):
            if process.state() != QtCore.QProcess.ProcessState.NotRunning: process.kill(); process.waitForFinished(2000)
        event.accept()

    def dispose(self):
        self.close(); self.storage.cleanup()
