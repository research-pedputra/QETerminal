"""Direct controls for the classes in the user's analysis library."""
import ast
import copy
from pathlib import Path
from PySide6 import QtCore, QtWidgets as W
from analysis_engine import catalog
from analysis_presets import PRESETS, constructor, call


# Construction and loading are automatic; users choose an operation, not a sequence.
DEFINITIONS = {
    'Band structure': ('BandStructure', 'plot_band_structure'),
    'Band gap': ('BandGapAnalyzer', 'calculate'),
    'DOS / PDOS': ('PDOSPlotter', 'plot'),
    'Shifted PDOS': ('PDOSPlottershift', 'plot'),
    'Effective mass': ('EffectiveMass', 'calculate_average_mass'),
    'Effective mass with fitting': ('EffectiveMassAnalyzer', 'analyze'),
    'Charge transfer': ('ChargeTransfer', 'plot'),
    'Orbital center': ('OrbitalCenter', 'get_extreme_positions'),
    'Total energy': ('TotalEnergy', 'last_TotalEnergy'),
    'XRD': ('SimulatedXRD', 'plot'),
    'Atomic scattering': ('AtomicScatteringPlot', 'plot'),
    'Structure factors': ('FhklCalculator', 'calculate_many'),
    'Projected bands': ('ProjectedBandPlotter', 'plot'),
    'Interatomic distances': ('QeInputFile', 'print_average_distances'),
    'Output surface positions': ('Minimaout', 'find_highest_z_coordinates'),
    'Input surface positions': ('Minimain', 'find_highest_z_coordinates'),
    'Surface energy': ('SurfaceEnergyCalculator', 'surf'),
    'Adsorption energy': ('AdsorptionCalculator', 'Eadsorption'),
    'Structure manipulation': ('QEManipulator', 'get_cell'),
}

EXTRA = {
    'Effective mass with fitting': [constructor('BandStructure'), call('get_bandstructure', data='@data:0', fermi_energy=0, kpoints=None)],
    'Structure factors': [constructor('FhklCalculator'), call('set_lattice', a=3.0)],
    'Interatomic distances': [constructor('QeInputFile', file_paths='@files:all')],
    'Output surface positions': [constructor('Minimaout', file_path='@file:0')],
    'Input surface positions': [constructor('Minimain', file_path='@file:0')],
    'Surface energy': [constructor('SurfaceEnergyCalculator')],
    'Adsorption energy': [constructor('AdsorptionCalculator', Eadsorbate=0.0)],
    'Structure manipulation': [constructor('QEManipulator', input_file='@file:0')],
}


class ValueControl(W.QWidget):
    changed = QtCore.Signal()

    def __init__(self, name, default, inputs):
        super().__init__(); self.name = name; self.default = default; self.kind = 'text'
        if isinstance(default, int) and not isinstance(default, bool) and name in ('fermi_energy', 'Ef', 'linewidth', 'base_linewidth', 'max_linewidth', 'alpha', 'beta', 'gamma', 'y_offset', 'shift_value', 'x_min', 'x_max', 'a', 'b', 'c', 'wavelength'):
            default = float(default)
        layout = W.QHBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        if isinstance(default, str) and default.startswith('@') and default.split(':')[0] in ('@file', '@data', '@atoms', '@cube', '@band', '@projected'):
            self.kind = 'file'; self.prefix, index = default.split(':'); self.index = int(index)
            self.edit = W.QComboBox(); self.set_inputs(inputs)
            self.edit.currentIndexChanged.connect(self.changed)
        elif isinstance(default, bool):
            self.kind = 'bool'; self.edit = W.QCheckBox(); self.edit.setChecked(default); self.edit.toggled.connect(self.changed)
        elif isinstance(default, int):
            self.kind = 'number'; self.edit = W.QSpinBox(); self.edit.setRange(-1000000000, 1000000000); self.edit.setValue(default); self.edit.valueChanged.connect(self.changed)
        elif isinstance(default, float):
            self.kind = 'number'; self.edit = W.QDoubleSpinBox(); self.edit.setDecimals(12); self.edit.setRange(-1e100, 1e100); self.edit.setValue(default); self.edit.valueChanged.connect(self.changed)
        else:
            text = '' if default is None else (default if isinstance(default, str) else repr(default))
            self.edit = W.QLineEdit(text); self.edit.setPlaceholderText('Automatic / not set' if default is None else '')
            self.edit.textChanged.connect(self.changed)
            self.edit.setToolTip('Text needs no quotes. For lists, ranges, or mappings use Python notation, e.g. (-5, 5), [1, 2], or {"Fe": "red"}. Leave blank for the class default None.')
        layout.addWidget(self.edit)
        if 'color' in name and self.kind == 'text' and not isinstance(default, (list, dict, tuple)):
            button = W.QPushButton('Color…'); button.clicked.connect(self.pick_color); layout.addWidget(button)

    def pick_color(self):
        color = W.QColorDialog.getColor(parent=self)
        if color.isValid(): self.edit.setText(color.name())

    def set_inputs(self, inputs):
        if self.kind != 'file': return
        current = self.edit.currentData() if self.edit.count() else self.index
        self.edit.blockSignals(True); self.edit.clear()
        for index, item in enumerate(inputs): self.edit.addItem(f'{index + 1}. {Path(item["path"]).name}', index)
        if current is None: current = self.index
        if current >= len(inputs): self.edit.addItem('Choose an input file above', current)
        self.edit.setCurrentIndex(max(0, self.edit.findData(current))); self.edit.blockSignals(False)

    def value(self):
        if self.kind == 'file': return f'{self.prefix}:{self.edit.currentData()}'
        if self.kind == 'bool': return self.edit.isChecked()
        if self.kind == 'number': return self.edit.value()
        text = self.edit.text().strip()
        if not text: return None
        if isinstance(self.default, str): return text
        try: return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            if isinstance(self.default, (list, tuple, dict)) or text.startswith(('[', '(', '{')):
                raise ValueError(f'Check {self.name}: enter a valid list, range, or mapping.')
            return text


class AnalysisControls(W.QWidget):
    changed = QtCore.Signal()

    def __init__(self):
        super().__init__(); self.catalog = catalog(); self.inputs = []; self.controls = []; self.cache = {}
        layout = W.QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        row = W.QHBoxLayout(); row.addWidget(W.QLabel('Operation'))
        self.operation = W.QComboBox(); row.addWidget(self.operation, 1); layout.addLayout(row)
        scroll = W.QScrollArea(); scroll.setWidgetResizable(True); layout.addWidget(scroll, 1)
        self.host = W.QWidget(); self.form_layout = W.QVBoxLayout(self.host); scroll.setWidget(self.host)
        self.operation.currentTextChanged.connect(self.operation_changed)
        from band_options import BandPlotOptions
        self.band_options = BandPlotOptions(); self.band_options.setParent(self.host); self.band_options.hide(); self.band_options.changed.connect(self.changed)
        from pdos_options import PdosOptions
        self.pdos_options = PdosOptions(); self.pdos_options.setParent(self.host); self.pdos_options.hide(); self.pdos_options.changed.connect(self.changed)

    def configure(self, name):
        self.name = name; self.cls, default = DEFINITIONS[name]; self.cache = {}; self.controls = []; self.series_values = {}
        self.operation.blockSignals(True); self.operation.clear()
        methods = [m for m in self.catalog[self.cls] if m != '__init__']
        self.operation.addItems([default] + [m for m in methods if m != default]); self.operation.blockSignals(False)
        self.rebuild()

    def operation_changed(self):
        if hasattr(self, 'series'): self.pdos_options.remember()
        for key, control in self.controls:
            try: self.cache[key] = control.value()
            except ValueError: pass
        self.rebuild(); self.changed.emit()

    def set_inputs(self, inputs):
        self.inputs = inputs
        self.band_options.set_inputs(inputs)
        self.pdos_options.set_inputs(inputs)
        for _, control in self.controls: control.set_inputs(inputs)

    def group(self, title):
        box = W.QGroupBox(title); form = W.QFormLayout(box); self.form_layout.addWidget(box); return form

    def field(self, form, key, name, value):
        control = ValueControl(name, self.cache.get(key, value), self.inputs)
        control.changed.connect(self.changed); self.controls.append((key, control))
        form.addRow(name.replace('_', ' '), control)

    def parameters(self, step, key, title, cls):
        method = '__init__' if 'class' in step else step['method']
        info = self.catalog[cls][method]
        defaults = dict(info['params']); defaults.update(step.get('kwargs', {}))
        form = self.group(title); form.parentWidget().setToolTip(info['doc'])
        for name, value in defaults.items():
            if isinstance(value, str) and value.startswith(('@ref:', '@ax:', '@fig:', '@files:')): continue
            if name in ('files', 'data_dict') and self.cls in ('PDOSPlotter', 'PDOSPlottershift'): continue
            self.field(form, (key, name), name, value)
        if info['varargs'] and cls != 'SimulatedXRD':
            self.field(form, (key, '*args'), info['varargs'], step.get('args', []))

    def rebuild(self):
        self.band_mode = self.cls == 'BandGapAnalyzer' or (self.cls == 'BandStructure' and self.operation.currentText() in ('plot_band_structure', 'plot_band_number', 'plot_shifted_band_structure', 'shift_band', 'save_figure'))
        self.controls = []
        if hasattr(self, 'series'): del self.series
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item.widget():
                item.widget().hide()
                if item.widget() not in (self.band_options, self.pdos_options): item.widget().deleteLater()
        if self.band_mode:
            self.form_layout.addWidget(self.band_options); self.band_options.show()
        if self.cls == 'BandGapAnalyzer':
            self.setup = []; self.action = {}; self.form_layout.addStretch(); return
        preset = 'Projected bands (JSON)' if self.name == 'Projected bands' else self.name
        steps = copy.deepcopy(PRESETS.get(preset, EXTRA.get(self.name, [])))
        if self.name in ('Effective mass', 'Effective mass with fitting'):
            if self.name == 'Effective mass with fitting':
                steps[0]['id'] = 'bands'; steps[1]['object'] = 'bands'
                steps.append(constructor(self.cls, band_structure='@ref:bands'))
            else: steps = steps[:-1]
        elif self.name in PRESETS or self.name == 'Projected bands':
            # Remove final operations; retain only construction and necessary loading.
            keep = {'get_bandstructure', 'add_structure', 'add_atoms'}
            steps = [s for s in steps if 'class' in s or s.get('method') in keep]
        if self.cls == 'TotalEnergy': steps.append(call('extract'))
        if self.cls == 'BandStructure' and self.operation.currentText() in ('read_scf_info', 'effmass', 'get_band_gap'):
            steps = [s for s in steps if 'class' in s]
        if self.cls == 'SimulatedXRD':
            next(s for s in steps if s.get('method') == 'add_structure')['args'] = ['@files:all']
        self.setup = steps
        for i, step in enumerate(steps):
            if self.band_mode and step.get('method') == 'get_bandstructure': continue
            cls = step.get('class') or ('BandStructure' if step.get('object') == 'bands' else self.cls)
            title = 'Class options' if 'class' in step else {'get_bandstructure': 'Band data', 'add_structure': 'Structures', 'add_atoms': 'Elements', 'set_lattice': 'Lattice'}.get(step['method'], 'Input options')
            if self.catalog[cls]['__init__' if 'class' in step else step['method']]['params'] or step.get('args'):
                self.parameters(step, i, title, cls)
        if self.cls in ('PDOSPlotter', 'PDOSPlottershift'):
            self.form_layout.addWidget(self.pdos_options); self.pdos_options.show(); self.series = self.pdos_options.table
        if self.cls == 'FhklCalculator':
            form = self.group('Atomic sites')
            self.field(form, ('sites', 'rows'), 'sites (symbol, x, y, z, occupancy, B)', [['Fe', 0, 0, 0, 1, 0]])
        method = self.operation.currentText(); defaults = {}
        for s in PRESETS.get(preset, []):
            if s.get('method') == method: defaults.update(s.get('kwargs', {}))
        if self.cls == 'ProjectedBandPlotter' and method == 'plot': defaults.update(ax='@ax:')
        if self.cls == 'PDOSPlottershift' and method == 'plot': defaults.update(ax='@ax:', Ef=0.0)
        if self.cls == 'FhklCalculator' and method in ('calculate_many', 'print_table'): defaults['hkls'] = [[1, 0, 0], [1, 1, 0]]
        # File-dependent helper operations also use the normal input selector.
        for name in self.catalog[self.cls][method]['params']:
            if name in ('file', 'filepath', 'file_path', 'scf_file'): defaults.setdefault(name, '@file:0')
            if name == 'atoms': defaults.setdefault(name, '@atoms:0')
        self.action = call(method, **defaults)
        self.parameters(self.action, 'action', 'Plot / calculation options', self.cls)
        self.form_layout.addStretch()

    def automatic(self):
        return self.operation.currentText() not in ('save', 'save_plot', 'save_figure', 'strain_scan')

    def build(self):
        if self.cls == 'BandGapAnalyzer':
            return {'steps': [constructor('BandStructure'), call('get_bandstructure', data='@data:0', fermi_energy=0, kpoints=None), call('plot_band_structure')], 'band_options': self.band_options.configuration()}
        steps = copy.deepcopy(self.setup); action = copy.deepcopy(self.action); sites = []
        for (key, name), control in self.controls:
            value = control.value()
            if key == 'sites': sites = value; continue
            step = action if key == 'action' else steps[key]
            if name == '*args': step['args'] = value or []
            else: step.setdefault('kwargs', {})[name] = value
        if self.cls in ('PDOSPlotter', 'PDOSPlottershift') and action['method'] == 'plot':
            options = self.pdos_options.configuration()
            steps.append(action)
            return {'steps': steps, 'pdos_options': options}
        for row in sites:
            if len(row) < 4: raise ValueError('Each atomic site needs symbol, x, y, z.')
            steps.append(call('add_atom', **dict(zip(('symbol', 'x', 'y', 'z', 'occupancy', 'B', 'label', 'skip_duplicate'), row))))
        if self.cls == 'SimulatedXRD':
            for step in steps:
                if step.get('method') == 'add_structure': step['args'] = [f'@file:{i}' for i in range(len(self.inputs))]
        if action['method'].startswith('save') and self.cls in ('BandStructure', 'PDOSPlotter', 'SimulatedXRD', 'AtomicScatteringPlot'):
            steps.append(call('plot_band_structure' if self.cls == 'BandStructure' else 'plot'))
        steps.append(action)
        if self.cls == 'BandStructure' and action['method'] == 'shift_band': steps.append(call('plot_band_structure'))
        if self.cls == 'BandGapAnalyzer' and action['method'] == 'calculate': steps.append(call('print_results'))
        classes = {}
        def check_input(value):
            if isinstance(value, dict):
                for v in value.values(): check_input(v)
            elif isinstance(value, list):
                for v in value: check_input(v)
            elif isinstance(value, str) and value.startswith(('@file:', '@data:', '@atoms:', '@cube:', '@band:', '@projected:')):
                if int(value.split(':')[1]) >= len(self.inputs): raise ValueError('Choose the required input file above before running.')
            elif value == '@files:all' and not self.inputs: raise ValueError('Add at least one input file.')
        for step in steps:
            if 'class' in step: classes[step['id']] = step['class']
            cls = step.get('class') or classes[step.get('object', 'analysis')]
            method = '__init__' if 'class' in step else step['method']
            kwargs = step.get('kwargs', {})
            for name in self.catalog[cls][method]['required'][len(step.get('args', [])):]:
                if cls == 'BandStructure' and name == 'kpoints': continue
                if kwargs.get(name) is None: raise ValueError(f'Enter {name.replace("_", " ")} for {cls}.{method}.')
            check_input(kwargs); check_input(step.get('args', []))
        result = {'steps': steps}
        if self.band_mode: result['band_options'] = self.band_options.configuration()
        return result
