"""File associations, SCF-derived band edges, and ordinary plotting controls."""
import math
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets as W


class ColorPicker(W.QWidget):
    changed = QtCore.Signal()

    def __init__(self, color='red'):
        super().__init__()
        from matplotlib.colors import get_named_colors_mapping, to_hex
        self.colors = get_named_colors_mapping(); self.to_hex = to_hex
        row = W.QHBoxLayout(self); row.setContentsMargins(0, 0, 0, 0)
        self.combo = W.QComboBox()
        for name, value in sorted(self.colors.items()):
            pixmap = QtGui.QPixmap(18, 18); pixmap.fill(QtGui.QColor(to_hex(value)))
            self.combo.addItem(QtGui.QIcon(pixmap), name, to_hex(value))
        self.combo.setCurrentIndex(self.combo.findText(color)); row.addWidget(self.combo, 1)
        button = W.QPushButton('Custom…'); row.addWidget(button); button.clicked.connect(self.custom)
        self.combo.currentIndexChanged.connect(self.changed)

    def custom(self):
        color = W.QColorDialog.getColor(QtGui.QColor(self.value()), self, 'Choose plot color')
        if color.isValid():
            value = color.name(QtGui.QColor.NameFormat.HexRgb)
            self.combo.addItem('Custom ' + value, value); self.combo.setCurrentIndex(self.combo.count()-1)

    def value(self): return self.combo.currentData()


class BandPlotOptions(W.QGroupBox):
    changed = QtCore.Signal()
    view_file = QtCore.Signal(int)

    def __init__(self):
        super().__init__('Band structure')
        self.inputs = []; self.associations = {}; layout = W.QVBoxLayout(self); form = W.QFormLayout(); layout.addLayout(form)
        self.data_file = W.QComboBox(); self.scf_file = W.QComboBox()
        for label, combo in [('Band data (.gnu)', self.data_file), ('Associated SCF output', self.scf_file)]:
            row = W.QHBoxLayout(); row.addWidget(combo, 1)
            button = W.QPushButton('View file'); button.clicked.connect(lambda checked=False, c=combo: self.view_file.emit(c.currentData() if c.currentData() is not None else -1)); row.addWidget(button)
            form.addRow(label, row); combo.currentIndexChanged.connect(self.changed)
        self.auto_fermi = W.QCheckBox('Use optional SCF Fermi energy'); self.auto_fermi.setChecked(False)
        self.auto_edges = W.QCheckBox('Find global VBM / CBM across all bands'); self.auto_edges.setChecked(True)
        self.fermi = self.number(0); self.vbm = W.QSpinBox(); self.cbm = W.QSpinBox()
        for control, value in ((self.vbm, 1), (self.cbm, 2)): control.setRange(1, 100000); control.setValue(value)
        form.addRow(self.auto_fermi); form.addRow('Fermi energy (eV)', self.fermi)
        form.addRow(self.auto_edges); row = W.QHBoxLayout(); row.addWidget(W.QLabel('VBM band')); row.addWidget(self.vbm); row.addWidget(W.QLabel('CBM band')); row.addWidget(self.cbm); form.addRow(row)
        self.summary = W.QLabel('Choose band data. Default reference: energies relative to Ef = 0 eV.'); self.summary.setWordWrap(True); form.addRow(self.summary)
        self.highlight = W.QCheckBox('Highlight VBM/CBM and calculate CBM − VBM gap'); self.highlight.setChecked(True); form.addRow(self.highlight)
        note = W.QLabel('Plain .gnu files have no occupations. Default: energies are relative to Ef=0. For absolute energies, enter the reference or use SCF. Gaps and direct/indirect labels cover the sampled k path only.'); note.setWordWrap(True); form.addRow(note)
        self.points = W.QTableWidget(0, 2); self.points.setHorizontalHeaderLabels(['High-symmetry label', 'k-path distance'])
        self.points.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch); self.points.setMaximumHeight(150)
        layout.addWidget(self.points); row = W.QHBoxLayout(); layout.addLayout(row)
        for label, callback in [('Add point', self.add_point), ('Remove point', self.remove_point)]:
            button = W.QPushButton(label); button.clicked.connect(callback); row.addWidget(button)
        self.points.itemChanged.connect(self.changed)
        ranges = W.QFormLayout(); layout.addLayout(ranges)
        self.x_auto = W.QCheckBox('Auto X range'); self.x_auto.setChecked(True)
        self.y_auto = W.QCheckBox('Auto Y range'); self.y_auto.setChecked(False)
        self.xmin = self.number(0); self.xmax = self.number(1); self.ymin = self.number(-5); self.ymax = self.number(5)
        for check, low, high in [(self.x_auto, self.xmin, self.xmax), (self.y_auto, self.ymin, self.ymax)]:
            row = W.QHBoxLayout(); row.addWidget(check); row.addWidget(low); row.addWidget(high); ranges.addRow(row)
        self.band_color = ColorPicker('gray'); self.vbm_color = ColorPicker('blue'); self.cbm_color = ColorPicker('red')
        for label, control in [('Other bands', self.band_color), ('VBM color', self.vbm_color), ('CBM color', self.cbm_color)]: ranges.addRow(label, control); control.changed.connect(self.changed)
        self.linewidth = self.number(1.5); self.linewidth.setRange(0.1, 20); ranges.addRow('Band line width', self.linewidth)
        for checkbox in (self.auto_fermi, self.auto_edges, self.x_auto, self.y_auto): checkbox.toggled.connect(self.update_enabled)
        for widget in (self.auto_fermi, self.auto_edges, self.highlight, self.x_auto, self.y_auto): widget.toggled.connect(self.changed)
        for widget in (self.fermi, self.vbm, self.cbm, self.xmin, self.xmax, self.ymin, self.ymax, self.linewidth): widget.valueChanged.connect(self.changed)
        self.update_enabled()

    def number(self, value):
        control = W.QDoubleSpinBox(); control.setDecimals(6); control.setRange(-1e9, 1e9); control.setValue(value); return control

    def update_enabled(self):
        self.fermi.setEnabled(not self.auto_fermi.isChecked())
        for control in (self.vbm, self.cbm): control.setEnabled(not self.auto_edges.isChecked())
        for check, controls in [(self.x_auto, (self.xmin, self.xmax)), (self.y_auto, (self.ymin, self.ymax))]:
            for control in controls: control.setEnabled(not check.isChecked())

    def set_inputs(self, inputs):
        old = self.inputs; self.inputs = list(inputs)
        for role, combo in [('bands', self.data_file), ('scf', self.scf_file)]:
            index = combo.currentData()
            key = (old[index].get('session'), old[index]['path']) if index is not None and 0 <= index < len(old) else None
            combo.blockSignals(True); combo.clear(); combo.addItem('Choose associated file…', None)
            for i, item in enumerate(inputs): combo.addItem(f'{Path(item["path"]).name} — {"SSH" if item.get("remote") else "Local"}', i)
            found = next((i for i, item in enumerate(inputs) if (item.get('session'), item['path']) == key), None)
            if found is None and key is None:
                found = next((i for i, item in enumerate(inputs) if (Path(item['path']).suffix.lower() in ('.gnu', '.dat') if role == 'bands' else 'scf' in Path(item['path']).name.lower() and Path(item['path']).suffix.lower() in ('.out', '.pwo', '.pwout'))), None)
            combo.setCurrentIndex(combo.findData(found)); combo.blockSignals(False)

    def add_point(self, checked=False, label='', coordinate=''):
        self.points.blockSignals(True); row = self.points.rowCount(); self.points.insertRow(row)
        self.points.setItem(row, 0, W.QTableWidgetItem(label)); self.points.setItem(row, 1, W.QTableWidgetItem(str(coordinate)))
        self.points.blockSignals(False); self.changed.emit()

    def remove_point(self):
        if self.points.currentRow() >= 0: self.points.removeRow(self.points.currentRow()); self.changed.emit()

    def configuration(self):
        data = self.data_file.currentData(); scf = self.scf_file.currentData()
        if data is None: raise ValueError('Choose the band-data file.')
        if self.auto_fermi.isChecked() and scf is None:
            raise ValueError('Choose SCF output or switch off the optional SCF reference.')
        points = []
        for row in range(self.points.rowCount()):
            label = self.points.item(row, 0).text().strip(); text = self.points.item(row, 1).text().strip()
            if not label and not text: continue
            try: value = float(text)
            except ValueError: raise ValueError('Enter a numeric k-path distance for each high-symmetry point.')
            if not label or not math.isfinite(value): raise ValueError('Each symmetry point needs a label and finite distance.')
            points.append((label, value))
        result = dict(data=data, scf=scf, auto_fermi=self.auto_fermi.isChecked(), auto_edges=self.auto_edges.isChecked(), fermi=self.fermi.value(), vbm=self.vbm.value(), cbm=self.cbm.value(), highlight=self.highlight.isChecked(), points=points or None,
                      colors=[self.band_color.value(), self.vbm_color.value(), self.cbm_color.value()], linewidth=self.linewidth.value())
        for name, check, low, high in [('xlim', self.x_auto, self.xmin, self.xmax), ('ylim', self.y_auto, self.ymin, self.ymax)]:
            result[name] = None if check.isChecked() else [low.value(), high.value()]
            if result[name] and low.value() >= high.value(): raise ValueError('Axis minimum must be smaller than its maximum.')
        return result

    def set_result(self, result):
        for control, value, automatic in [(self.fermi, result['fermi'], self.auto_fermi.isChecked()), (self.vbm, result['vbm'], self.auto_edges.isChecked()), (self.cbm, result['cbm'], self.auto_edges.isChecked())]:
            if automatic: control.blockSignals(True); control.setValue(value); control.blockSignals(False)
        self.summary.setText(f'Gap = {result["gap"]:.6g} eV — {result.get("classification", "")}. VBM = {result.get("vbm_energy", 0):g} eV at k={result.get("vbm_k", [])}; CBM = {result.get("cbm_energy", 0):g} eV at k={result.get("cbm_k", [])}. Reference = {result["fermi"]:g} eV.')
