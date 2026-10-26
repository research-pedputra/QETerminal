"""Filename-based atom/orbital selection and grouped curve controls."""
from PySide6 import QtCore, QtWidgets as W
from pdos_data import MODES, identify, group_inputs, natural_key
from band_options import ColorPicker


class PdosOptions(W.QGroupBox):
    changed = QtCore.Signal()

    def __init__(self):
        super().__init__('PDOS selection and summation'); self.inputs = []; self.groups = []; self.saved = {}; self.error = ''
        layout = W.QVBoxLayout(self); form = W.QFormLayout(); layout.addLayout(form)
        self.mode = W.QComboBox(); self.mode.addItems(MODES); form.addRow('Group curves by', self.mode)
        self.element = W.QComboBox(); self.atom = W.QComboBox(); self.orbital = W.QComboBox()
        for label, combo in [('Element', self.element), ('Atom index', self.atom), ('Orbital', self.orbital)]: combo.addItem('All'); form.addRow(label, combo)
        self.word1 = W.QLineEdit(); self.word2 = W.QLineEdit(); self.word1.setPlaceholderText('(Cs)'); self.word2.setPlaceholderText('(d)')
        form.addRow('Filename contains (first)', self.word1); form.addRow('Then contains (second)', self.word2)
        self.layout_choice = W.QComboBox(); self.layout_choice.addItems(['Auto', 'Non-spin', 'Spin up/down']); form.addRow('Column layout', self.layout_choice)
        self.total_kind = W.QComboBox(); self.total_kind.addItems(['Projected DOS', 'Total DOS']); form.addRow('For pdos_tot files', self.total_kind)
        self.integral_min = W.QLineEdit(); self.integral_max = W.QLineEdit()
        for label, field in [('Integrate from (raw eV)', self.integral_min), ('Integrate to (raw eV)', self.integral_max)]:
            field.setPlaceholderText('File energy limit (automatic)'); form.addRow(label, field)
            field.setToolTip('Trapezoidal integral in states, using unshifted file energies. Blank uses the file limit; spin channels stay separate.')
            field.textChanged.connect(self.changed)
        self.table = W.QTableWidget(0, 4); self.table.setHorizontalHeaderLabels(['Plot / label', 'Color', 'Line style', 'Files'])
        self.table.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch); self.table.setMinimumHeight(140); layout.addWidget(self.table)
        self.status = W.QLabel(); self.status.setWordWrap(True); layout.addWidget(self.status)
        note = W.QLabel('Select multiple QE PDOS files above. Groups sum LDOS once, preserving spin channels. Double-click a color to choose it. Uncheck groups to exclude them.'); note.setWordWrap(True); layout.addWidget(note)
        for combo in (self.mode, self.element, self.atom, self.orbital): combo.currentTextChanged.connect(self.rebuild)
        for edit in (self.word1, self.word2): edit.textChanged.connect(self.rebuild)
        for combo in (self.layout_choice, self.total_kind): combo.currentTextChanged.connect(self.changed)
        self.table.itemChanged.connect(self.changed); self.table.cellDoubleClicked.connect(self.choose_color)

    def remember(self):
        for row, group in enumerate(self.groups):
            if row < self.table.rowCount():
                key = (group['label'], tuple(group['indices']), group.get('component'))
                self.saved[key] = ([self.table.item(row, j).text() for j in range(3)], self.table.item(row, 0).checkState())

    def set_inputs(self, inputs):
        self.remember(); self.inputs = list(inputs)
        meta = [identify(item['path']) for item in inputs]; meta = [m for m in meta if m]
        for combo, key in [(self.element, 'element'), (self.atom, 'atom'), (self.orbital, 'orbital')]:
            selected = combo.currentText(); combo.blockSignals(True); combo.clear(); combo.addItem('All')
            combo.addItems(sorted({str(m[key]) for m in meta}, key=natural_key)); combo.setCurrentText(selected if combo.findText(selected) >= 0 else 'All'); combo.blockSignals(False)
        self.rebuild()

    def rebuild(self, *args):
        self.remember(); self.error = ''
        try: self.groups = group_inputs(self.inputs, self.mode.currentText(), self.element.currentText(), self.atom.currentText(), self.orbital.currentText(), self.word1.text(), self.word2.text())
        except ValueError as exc: self.error = str(exc); self.groups = []
        self.table.blockSignals(True); self.table.setRowCount(len(self.groups))
        for row, group in enumerate(self.groups):
            key = (group['label'], tuple(group['indices']), group.get('component'))
            values, checked = self.saved.get(key, ([group['label'], f'C{row%10}', '-'], QtCore.Qt.CheckState.Checked))
            for col, value in enumerate(values + [str(len(group['indices']))]): self.table.setItem(row, col, W.QTableWidgetItem(value))
            item = self.table.item(row, 0); item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable); item.setCheckState(checked)
            item.setToolTip('\n'.join(self.inputs[i]['path'] for i in group['indices']))
            count = self.table.item(row, 3); count.setFlags(count.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        self.table.blockSignals(False)
        self.status.setText(self.error or f'{len(self.groups)} group(s), {len(set(i for g in self.groups for i in g["indices"]))} matched input file(s). Hover over a group for its sources.')
        self.changed.emit()

    def choose_color(self, row, column):
        if column != 1: return
        dialog = W.QDialog(self); dialog.setWindowTitle('PDOS curve color'); layout = W.QVBoxLayout(dialog); picker = ColorPicker('blue'); layout.addWidget(picker)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Ok | W.QDialogButtonBox.StandardButton.Cancel); layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        if dialog.exec() == W.QDialog.DialogCode.Accepted: self.table.item(row, column).setText(picker.value())
        dialog.deleteLater()

    def configuration(self):
        if self.error: raise ValueError(self.error)
        groups = []; labels = set()
        for row, group in enumerate(self.groups):
            if self.table.item(row, 0).checkState() != QtCore.Qt.CheckState.Checked: continue
            label, color, style = [self.table.item(row, j).text().strip() for j in range(3)]
            if not label or label in labels: raise ValueError('Use a unique label for each PDOS group.')
            labels.add(label); groups.append(dict(group, label=label, color=color, style=style))
        if not groups: raise ValueError('No PDOS groups selected. Add matching files or adjust the filters.')
        try:
            integration = {key: float(field.text()) if field.text().strip() else None for key, field in [('lower', self.integral_min), ('upper', self.integral_max)]}
        except ValueError:
            raise ValueError('Integration limits must be numbers in raw file energies (eV), or blank.')
        return dict(groups=groups, layout=self.layout_choice.currentText(), total_kind=self.total_kind.currentText(), integration=integration, source_names=[i['path'] for i in self.inputs])
