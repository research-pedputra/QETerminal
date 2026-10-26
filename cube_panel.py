"""Interactive cube surface/slice tab. Numeric and file work runs outside the GUI."""
from pathlib import Path
import numpy as np
import os
import csv
from PySide6 import QtCore, QtGui, QtWidgets as W
from cube_data import read_cube_bytes
from ssh_session import digest
from atom_scene import build_scene
from render_backends import PyVistaBackend


class TaskSignals(QtCore.QObject):
    done = QtCore.Signal(object, object)


class Task(QtCore.QRunnable):
    def __init__(self, function):
        super().__init__(); self.function = function; self.signals = TaskSignals()
    def run(self):
        try: self.signals.done.emit(self.function(), None)
        except Exception as exc: self.signals.done.emit(None, str(exc))


class CubePanel(W.QWidget):
    changed = QtCore.Signal()

    def __init__(self, workspace, path, session=None):
        super().__init__()
        self.workspace = workspace; self.path = path; self.remote_session = session
        self.field = self.original = self.previous = None
        self.expected = None; self.save_pending = False; self.busy = False; self.modified = False
        self.disposed = False; self.planes = []; self.task = None; self.backend = None
        self.operation_operands = []; self.phase_structures = []; self._slice_preview = None
        self.positive_color = '#4085d4'; self.negative_color = '#dd6056'
        self._redraw_timer = QtCore.QTimer(self); self._redraw_timer.setSingleShot(True); self._redraw_timer.setInterval(180)
        self._redraw_timer.timeout.connect(lambda: self.workspace.guard(self.redraw))
        layout = W.QVBoxLayout(self)
        self.path_label = W.QLabel(('SSH: ' if session else '') + path)
        self.path_label.setTextFormat(QtCore.Qt.TextFormat.PlainText); layout.addWidget(self.path_label)
        self.controls = W.QWidget(); tools = W.QVBoxLayout(self.controls); tools.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.controls)
        properties = W.QGroupBox('Properties'); properties_layout = W.QVBoxLayout(properties); tools.addWidget(properties)
        row = W.QHBoxLayout(); properties_layout.addLayout(row)
        self.iso = W.QLineEdit('0.02'); self.iso.setMaximumWidth(110)
        row.addWidget(W.QLabel('± Isovalue')); row.addWidget(self.iso)
        self.opacity = W.QDoubleSpinBox(); self.opacity.setRange(.05, 1); self.opacity.setSingleStep(.05); self.opacity.setValue(.55)
        row.addWidget(W.QLabel('Opacity')); row.addWidget(self.opacity)
        self.surfaces = W.QCheckBox('Surfaces'); self.surfaces.setChecked(True); row.addWidget(self.surfaces)
        self.atoms = W.QCheckBox('Atoms / bonds'); self.atoms.setChecked(True); row.addWidget(self.atoms)
        self.spectrum = [('Violet', '#6a00a8'), ('Blue', '#2455d6'), ('Cyan', '#00a6ca'), ('Green', '#22a884'), ('Yellow', '#f6d746'), ('Orange', '#f28e2b'), ('Red', '#d7191c')]
        row.addWidget(W.QLabel('+')); self.color_positive = self.spectrum_combo('#2455d6'); row.addWidget(self.color_positive)
        row.addWidget(W.QLabel('−')); self.color_negative = self.spectrum_combo('#d7191c'); row.addWidget(self.color_negative)
        self.unit_cell = W.QCheckBox('Unit cell'); self.unit_cell.setChecked(True); row.addWidget(self.unit_cell)
        self.button(row, 'Fit', lambda: self.redraw(True))
        row.addStretch()
        control_row = W.QHBoxLayout(); tools.addLayout(control_row)
        operations_box = W.QGroupBox('Operations'); operations_layout = W.QVBoxLayout(operations_box); control_row.addWidget(operations_box, 1)
        row = W.QHBoxLayout(); operations_layout.addLayout(row)
        self.operation = W.QComboBox(); self.operation.addItems(['+', '−', '×', '÷']); self.operation.setToolTip('Apply this operation to the original cube and each added cube in order.'); row.addWidget(self.operation)
        self.scalar = W.QLineEdit(); self.scalar.setPlaceholderText('scalar'); self.scalar.setMaximumWidth(85); row.addWidget(self.scalar)
        self.button(row, 'Add phase…', self.add_operand_cube)
        row.addStretch()
        row = W.QHBoxLayout(); operations_layout.addLayout(row)
        self.button(row, 'Undo / redo', self.undo); self.button(row, 'Reset field', self.reset_field)
        self.button(row, 'Save cube', lambda: workspace.save_cube(self))
        self.button(row, 'Save as…', lambda: workspace.save_cube(self, copy=True))
        row.addWidget(W.QLabel('Integrate along')); self.integration_axis = W.QComboBox(); self.integration_axis.addItems(['x', 'y', 'z']); row.addWidget(self.integration_axis)
        self.button(row, 'Integrate', self.integrate_axis)
        row.addStretch()
        row = W.QHBoxLayout(); operations_layout.addLayout(row)
        row.addWidget(W.QLabel('Surface top atom')); self.top_atom = W.QComboBox(); row.addWidget(self.top_atom, 1)
        row.addWidget(W.QLabel('Molecule bottom atom')); self.bottom_atom = W.QComboBox(); row.addWidget(self.bottom_atom, 1)
        self.boundary_label = W.QLabel('Boundary: unavailable'); row.addWidget(self.boundary_label)
        self.operand_names = W.QLabel('No added cubes'); operations_layout.addWidget(self.operand_names)
        slice_box = W.QGroupBox('Slices'); slice_layout = W.QVBoxLayout(slice_box); control_row.addWidget(slice_box, 1)
        row = W.QHBoxLayout(); slice_layout.addLayout(row); row.addWidget(W.QLabel('h k l'))
        self.hkl = []
        for val in (0, 0, 1):
            spin = W.QSpinBox(); spin.setRange(-99, 99); spin.setValue(val); spin.setMaximumWidth(65)
            self.hkl.append(spin); row.addWidget(spin)
        row.addWidget(W.QLabel('d (Å)')); self.distance = W.QDoubleSpinBox(); self.distance.setRange(-1e6, 1e6); self.distance.setDecimals(5); row.addWidget(self.distance)
        self.reference = W.QComboBox(); self.reference.addItems(['Cartesian origin', 'Cube grid origin']); row.addWidget(self.reference)
        self.button(row, 'Add slice', self.add_plane)
        row = W.QHBoxLayout(); slice_layout.addLayout(row); self.plane_list = W.QComboBox(); self.plane_list.setPlaceholderText('Select slice'); row.addWidget(self.plane_list, 1)
        self.button(row, 'Edit…', self.edit_plane)
        self.distance.setToolTip('Signed perpendicular distance: n · (r − reference) = d. n is the reciprocal-lattice (hkl) unit normal.')
        self.iso.textEdited.connect(lambda *_: self._redraw_timer.start())
        self.operation.currentIndexChanged.connect(lambda *_: self.recompute_field() if self.field is not None and self.operation_operands else None)
        self.scalar.editingFinished.connect(lambda: self.recompute_field() if self.scalar.text().strip() else None)
        self.integration_axis.currentIndexChanged.connect(lambda *_: self.refresh_boundary_atoms())
        self.top_atom.currentIndexChanged.connect(lambda *_: self.update_boundary_label())
        self.bottom_atom.currentIndexChanged.connect(lambda *_: self.update_boundary_label())
        for widget in (*self.hkl, self.distance):
            (widget.valueChanged).connect(lambda *_: self.update_slice_preview())
        self.reference.currentIndexChanged.connect(lambda *_: self.update_slice_preview())
        for widget in (self.opacity, self.surfaces, self.atoms, self.unit_cell):
            (widget.valueChanged if isinstance(widget, W.QDoubleSpinBox) else widget.toggled).connect(lambda *_: self._redraw_timer.start())
        self.color_positive.currentIndexChanged.connect(lambda *_: self._redraw_timer.start())
        self.color_negative.currentIndexChanged.connect(lambda *_: self._redraw_timer.start())
        self.info = W.QLabel('Loading cube…'); self.info.setWordWrap(True); layout.addWidget(self.info)
        self.backend = PyVistaBackend(self, lambda _: None); layout.addWidget(self.backend.widget, 1)
        self.controls.setEnabled(False)

    def button(self, layout, text, callback):
        button = W.QPushButton(text); layout.addWidget(button)
        button.clicked.connect(lambda: self.workspace.guard(callback)); return button

    def spectrum_combo(self, selected):
        combo = W.QComboBox()
        for name, color in self.spectrum:
            combo.addItem(name, color)
            combo.setItemData(combo.count()-1, QtGui.QColor(color), QtCore.Qt.ItemDataRole.DecorationRole)
        index = combo.findData(selected)
        if index >= 0: combo.setCurrentIndex(index)
        return combo

    def dirty(self): return self.modified

    def run(self, function, callback):
        if self.busy: raise ValueError('Wait for the current cube operation to finish.')
        self.busy = True; self.controls.setEnabled(False); self.info.setText('Working…')
        self.task = Task(function)
        def finished(value, error):
            self.busy = False
            if self.disposed: return
            self.controls.setEnabled(self.field is not None and not self.save_pending)
            if error:
                self.save_pending = False
                self.controls.setEnabled(self.field is not None)
                self.info.setText(error); self.workspace.guard(lambda: self.raise_error(error)); return
            self.workspace.guard(lambda: callback(value))
        self.task.signals.done.connect(finished)
        QtCore.QThreadPool.globalInstance().start(self.task)

    @staticmethod
    def raise_error(error): raise ValueError(error)

    def load(self, source):
        def read():
            if callable(source): data = source()
            else: data = source
            return read_cube_bytes(data), digest(data)
        def loaded(result):
            self.field, self.expected = result; self.original = self.field
            self.controls.setEnabled(True); self.auto_iso(); self.refresh_boundary_atoms(); self.redraw(True)
        self.run(read, loaded)

    def add_operand_cube(self):
        paths, _ = W.QFileDialog.getOpenFileNames(self, 'Add phase files', str(Path(self.path).parent), 'Phase files (*.cube *.cub *.cif);;Cube files (*.cube *.cub);;CIF files (*.cif)')
        cif_paths = [path for path in paths if Path(path).suffix.lower() == '.cif']
        cube_paths = [path for path in paths if Path(path).suffix.lower() in ('.cube', '.cub')]
        for path in cif_paths:
            try:
                from ase.io import read
                atoms = read(path, format='cif')
                atoms = self.edit_cif_coordinates(atoms, Path(path).name)
                if atoms is not None:
                    self.phase_structures.append((path, atoms)); self.refresh_boundary_atoms(); self.redraw(True)
            except Exception as exc: raise ValueError(f'Could not add CIF phase {Path(path).name}: {exc}')
        paths = cube_paths
        paths = [path for path in paths if path != self.path and all(path != name for name, _ in self.operation_operands)]
        if not paths: return
        base = self.original; operands = list(self.operation_operands); operation = {'+': '+', '−': '-', '×': '*', '÷': '/'}[self.operation.currentText()]
        scalar = self.scalar.text().strip()
        def calculate():
            from cube_data import read_cube_bytes
            added = [(path, read_cube_bytes(Path(path).read_bytes())) for path in paths]
            result = base
            if scalar: result = result.arithmetic(operation, float(scalar))
            for _, cube_field in operands + added: result = result.arithmetic(operation, cube_field)
            return added, result
        def done(value):
            added, result = value; self.operation_operands.extend(added)
            self.operand_names.setText('\n'.join(Path(name).name for name, _ in self.operation_operands) or 'No added cubes')
            self.refresh_boundary_atoms()
            self.set_field(result)
        self.run(calculate, done)

    def edit_cif_coordinates(self, atoms, name):
        from ase.data import atomic_numbers
        dialog = W.QDialog(self); dialog.setWindowTitle(f'CIF phase coordinates · {name}'); dialog.resize(620, 440)
        layout = W.QVBoxLayout(dialog); cell = W.QLabel('Cell (Å): ' + '  '.join(f'{v:.4g}' for v in atoms.cell.cellpar())); layout.addWidget(cell)
        table = W.QTableWidget(len(atoms), 4); table.setHorizontalHeaderLabels(['Element', 'X (Å)', 'Y (Å)', 'Z (Å)']); table.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch); layout.addWidget(table)
        def populate():
            table.setRowCount(len(atoms))
            for row, (symbol, position) in enumerate(zip(atoms.get_chemical_symbols(), atoms.positions)):
                table.setItem(row, 0, W.QTableWidgetItem(symbol))
                for col, value in enumerate(position, 1): table.setItem(row, col, W.QTableWidgetItem(f'{value:.8g}'))
        populate()
        controls = W.QHBoxLayout(); add = W.QPushButton('Add atom'); delete = W.QPushButton('Delete selected'); controls.addWidget(add); controls.addWidget(delete); controls.addStretch(); layout.addLayout(controls)
        def add_atom():
            from ase import Atom
            atoms.append(Atom('H')); populate()
        add.clicked.connect(add_atom)
        delete.clicked.connect(lambda: self.delete_cif_atoms(atoms, table, populate))
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Ok | W.QDialogButtonBox.StandardButton.Cancel); layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        if dialog.exec() != W.QDialog.DialogCode.Accepted: return None
        symbols, positions = [], []
        for row in range(table.rowCount()):
            symbol = table.item(row, 0).text().strip() if table.item(row, 0) else ''
            if symbol not in atomic_numbers: raise ValueError(f'Invalid element on row {row + 1}.')
            try: xyz = [float(table.item(row, col).text()) for col in (1, 2, 3)]
            except (AttributeError, ValueError): raise ValueError(f'Invalid coordinate on row {row + 1}.')
            if not np.isfinite(xyz).all(): raise ValueError(f'Nonfinite coordinate on row {row + 1}.')
            symbols.append(symbol); positions.append(xyz)
        if not symbols: raise ValueError('A CIF phase must contain at least one atom.')
        atoms = atoms.__class__(symbols, positions=positions, cell=atoms.cell, pbc=atoms.pbc)
        return atoms

    @staticmethod
    def delete_cif_atoms(atoms, table, populate):
        indices = sorted({item.row() for item in table.selectedItems()}, reverse=True)
        for index in indices:
            if 0 <= index < len(atoms): del atoms[index]
        populate()

    def refresh_operands(self):
        # Kept for workspace tab lifecycle callbacks; operands belong to this panel.
        return None

    def recompute_field(self):
        if self.busy or self.field is None: return
        operation = {'+': '+', '−': '-', '×': '*', '÷': '/'}[self.operation.currentText()]
        base = self.original; operands = list(self.operation_operands); scalar = self.scalar.text().strip()
        def calculate():
            result = base
            if scalar: result = result.arithmetic(operation, float(scalar))
            for _, cube_field in operands: result = result.arithmetic(operation, cube_field)
            return result
        self.run(calculate, self.set_field)

    def integrate_axis(self):
        if self.field is None: raise ValueError('Wait for the cube to load.')
        if self.busy: raise ValueError('Wait for the current cube operation to finish.')
        axis = self.integration_axis.currentIndex(); field = self.field
        source_path = self.path
        top_position = self.top_atom.currentData(); bottom_position = self.bottom_atom.currentData()
        # CubeData spacing vectors are stored in Å; QE pp.x charge density is e/Bohr³.
        from ase.units import Bohr
        def calculate():
            other = [i for i in range(3) if i != axis]
            d_area = float(np.linalg.norm(np.cross(field.spacing[other[0]], field.spacing[other[1]])))
            # Convert raw QE e/Bohr³ samples to e/Å³, then integrate each plane's area.
            profile = np.sum(field.values, axis=tuple(other), dtype=np.float64) * d_area / (Bohr ** 3)
            step = float(field.spacing[axis, axis])
            if not np.isclose(abs(step), np.linalg.norm(field.spacing[axis]), rtol=1e-5, atol=1e-8):
                step = float(np.linalg.norm(field.spacing[axis]))
            distance = field.origin[axis] + np.arange(profile.size) * step
            order = np.argsort(distance); distance = distance[order]; profile = profile[order]
            csv_path = self.write_profile_csv(source_path, 'xyz'[axis], distance, profile)
            transfer = None
            if top_position is not None and bottom_position is not None:
                boundary = (float(top_position) + float(bottom_position)) / 2
                mask = distance >= boundary
                if np.count_nonzero(mask) < 2:
                    transfer = {'error': f'Fewer than two grid planes lie above the {boundary:.6g} Å boundary.'}
                else:
                    z_above, rho_above = distance[mask], profile[mask]
                    positive = float(np.trapezoid(np.maximum(rho_above, 0), z_above))
                    negative = float(np.trapezoid(np.maximum(-rho_above, 0), z_above))
                    net = float(np.trapezoid(rho_above, z_above))
                    transfer = {'boundary': boundary, 'top': float(top_position), 'bottom': float(bottom_position),
                                'positive': positive, 'negative': negative, 'net': net,
                                'matched': min(positive, negative)}
            return axis, distance, profile, csv_path, transfer
        def report(result):
            axis_index, distance, profile, csv_path, transfer = result
            lines = [f'Planar charge density along {"xyz"[axis_index]} (current displayed field)',
                     f'Distance unit: Å; planar density unit: e/Å for QE pp.x charge-density fields.',
                     f'CSV saved without overwrite: {csv_path}',
                     'Distance (Å)       Planar charge density (e/Å)']
            for z_value, rho_value in zip(distance, profile):
                lines.append(f'{z_value:14.7f}  {rho_value:+.10g}')
            if transfer is None:
                lines.append('Charge transfer: select a surface top atom and a molecule bottom atom first.')
            elif 'error' in transfer:
                lines.append('Charge transfer: ' + transfer['error'])
            else:
                lines.extend([f'Boundary = (surface atom {transfer["top"]:.7f} Å + molecule atom {transfer["bottom"]:.7f} Å) / 2 = {transfer["boundary"]:.7f} Å.',
                              f'Positive charge above boundary: {transfer["positive"]:.8g} e.',
                              f'Negative charge above boundary: {transfer["negative"]:.8g} e (magnitude).',
                              f'Positive − negative = {transfer["net"]:+.8g} e; matched positive/negative transfer = {transfer["matched"]:.8g} e.'])
            self.workspace.terminal.feed(('\r\n' + '\r\n'.join(lines) + '\r\n').encode('utf-8'))
            self.info.setText(f'Planar density along {"xyz"[axis_index]} saved to CSV; charge transfer reported in terminal.')
        self.run(calculate, report)

    def write_profile_csv(self, source_path, axis_name, distance, profile):
        source = Path(source_path)
        if self.remote_session or not source.is_absolute() or not source.exists():
            source = Path(self.workspace.folder) / Path(source_path).name
        base = source.with_name(f'{source.stem}_integrated_{axis_name}')
        index = 1
        while True:
            candidate = base.with_name(f'{base.name}{"" if index == 1 else f"_{index}"}.csv')
            try:
                with candidate.open('x', encoding='utf-8', newline='') as output:
                    writer = csv.writer(output)
                    writer.writerow(['distance_A', *[f'{value:.12g}' for value in distance]])
                    writer.writerow(['planar_charge_density_e_per_A', *[f'{value:.12g}' for value in profile]])
                return str(candidate)
            except FileExistsError:
                index += 1

    def refresh_boundary_atoms(self):
        axis = self.integration_axis.currentIndex(); self.top_atom.clear(); self.bottom_atom.clear()
        if self.field is not None and len(self.field.atoms):
            symbols = self.field.atoms.get_chemical_symbols()
            positions = self.field.atoms.positions
            order = np.argsort(positions[:, axis])[::-1]
            for atom_index in order:
                value = float(positions[atom_index, axis])
                self.top_atom.addItem(f'{symbols[atom_index]}{atom_index+1} · {value:.5f} Å', value)
            self.top_atom.setCurrentIndex(0)
        phase_atoms = [(Path(path).name, atoms) for path, atoms in self.phase_structures if len(atoms)]
        phase_atoms.extend((Path(path).name, cube_field.atoms)
                           for path, cube_field in self.operation_operands if len(cube_field.atoms))
        entries = []
        for name, atoms in phase_atoms:
            for atom_index, (symbol, position) in enumerate(zip(atoms.get_chemical_symbols(), atoms.positions)):
                entries.append((float(position[axis]), f'{name}: {symbol}{atom_index+1} · {position[axis]:.5f} Å'))
        for value, label in sorted(entries): self.bottom_atom.addItem(label, value)
        if entries: self.bottom_atom.setCurrentIndex(0)
        else: self.bottom_atom.setPlaceholderText('Add a molecule CIF or cube phase')
        self.update_boundary_label()

    def update_boundary_label(self):
        top = self.top_atom.currentData(); bottom = self.bottom_atom.currentData()
        if top is None or bottom is None: self.boundary_label.setText('Boundary: unavailable')
        else: self.boundary_label.setText(f'Boundary: {(float(top)+float(bottom))/2:.4f} Å')

    def auto_iso(self):
        peak = float(np.max(np.abs(self.field.values)))
        self.iso.setText(f'{peak*.2:.8g}' if peak else '0.01')

    def set_field(self, field):
        if self.save_pending: raise ValueError('Wait for the cube save to finish.')
        self.previous, self.field = self.field, field
        self.modified = True; self.changed.emit(); self.auto_iso(); self.redraw()

    def undo(self):
        if self.busy or self.save_pending: raise ValueError('Wait for the cube operation to finish.')
        if self.previous is not None:
            self.previous, self.field = self.field, self.previous
            self.modified = True; self.changed.emit(); self.auto_iso(); self.redraw()

    def reset_field(self):
        if self.busy or self.save_pending: raise ValueError('Wait for the cube operation to finish.')
        self.operation_operands.clear(); self.scalar.clear(); self.operand_names.setText('No added cubes')
        if self.original is not None: self.set_field(self.original)

    def add_plane(self):
        hkl = tuple(spin.value() for spin in self.hkl); d = self.distance.value()
        grid_origin = self.reference.currentIndex() == 1
        normal, point = self.field.plane(hkl, d, grid_origin)
        if not self.grid.slice(normal=normal, origin=point).n_points:
            raise ValueError('This plane does not intersect the cube. Adjust d, its sign, or the reference origin.')
        plane = {'hkl': hkl, 'distance': d, 'grid_origin': grid_origin, 'color': '#57a6d8', 'opacity': .8}
        self.planes.append(plane); self.plane_list.addItem(f'{hkl} · d={d:g} Å'); self.plane_list.setCurrentIndex(len(self.planes)-1)
        self._slice_preview = None; self.redraw()

    def edit_plane(self):
        index = self.plane_list.currentIndex()
        if not 0 <= index < len(self.planes): return
        plane = self.planes[index]; dialog = W.QDialog(self); dialog.setWindowTitle('Edit slice'); form = W.QFormLayout(dialog)
        hkl = [W.QSpinBox() for _ in range(3)]
        for spin, value in zip(hkl, plane['hkl']): spin.setRange(-99, 99); spin.setValue(value)
        hrow = W.QHBoxLayout()
        for spin in hkl: hrow.addWidget(spin)
        form.addRow('h k l', hrow)
        distance = W.QDoubleSpinBox(); distance.setRange(-1e6, 1e6); distance.setDecimals(5); distance.setValue(plane['distance']); form.addRow('Distance (Å)', distance)
        reference = W.QComboBox(); reference.addItems(['Cartesian origin', 'Cube grid origin']); reference.setCurrentIndex(int(plane['grid_origin'])); form.addRow('Reference', reference)
        color = self.spectrum_combo(plane['color']); opacity = W.QDoubleSpinBox(); opacity.setRange(.05, 1); opacity.setSingleStep(.05); opacity.setValue(plane['opacity'])
        form.addRow('Color', color); form.addRow('Opacity', opacity)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Ok | W.QDialogButtonBox.StandardButton.Cancel); delete = buttons.addButton('Delete slice', W.QDialogButtonBox.ButtonRole.DestructiveRole); form.addRow(buttons)
        deleted = {'value': False}
        delete.clicked.connect(lambda: (deleted.__setitem__('value', True), dialog.accept()))
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        if dialog.exec() != W.QDialog.DialogCode.Accepted: return
        if deleted['value']:
            self.planes.pop(index); self.plane_list.removeItem(index)
            if self.planes: self.plane_list.setCurrentIndex(min(index, len(self.planes)-1))
        else:
            hkl_value = tuple(spin.value() for spin in hkl); grid_origin = reference.currentIndex() == 1
            normal, point = self.field.plane(hkl_value, distance.value(), grid_origin)
            if not self.grid.slice(normal=normal, origin=point).n_points: raise ValueError('This plane does not intersect the cube.')
            plane.update(hkl=hkl_value, distance=distance.value(), grid_origin=grid_origin, color=color.currentData(), opacity=opacity.value())
            self.plane_list.setItemText(index, f'{hkl_value} · d={distance.value():g} Å')
        self.redraw()

    def update_slice_preview(self):
        if self.field is None or not hasattr(self, 'grid'): return
        try:
            hkl = tuple(spin.value() for spin in self.hkl); distance = self.distance.value(); reference = self.reference.currentIndex() == 1
            normal, point = self.field.plane(hkl, distance, reference)
            preview = (hkl, distance, reference) if self.grid.slice(normal=normal, origin=point).n_points else None
        except (ValueError, np.linalg.LinAlgError): preview = None
        self._slice_preview = preview; self._redraw_timer.start()

    def redraw(self, fit=False):
        if self.field is None: return
        try: level = float(self.iso.text())
        except ValueError: return
        if not np.isfinite(level) or level <= 0: raise ValueError('Isovalue must be a finite positive number.')
        plotter = self.backend.plotter
        camera = self.backend.get_camera()
        plotter.clear(); plotter.enable_lightkit()
        self.grid = self.field.grid()
        if self.atoms.isChecked() and (len(self.field.atoms) or self.phase_structures):
            atoms = self.field.atoms.copy()
            for _, phase_atoms in self.phase_structures: atoms += phase_atoms
            scene = build_scene(atoms, boundary=False)
            self.backend.populate(plotter, scene)
        low, high = float(self.field.values.min()), float(self.field.values.max())
        if self.surfaces.isChecked():
            for value, color in [(level, self.color_positive.currentData()), (-level, self.color_negative.currentData())]:
                if low < value < high:
                    surface = self.grid.contour([value], scalars='Field')
                    if surface.n_points:
                        plotter.add_mesh(surface, color=color, opacity=self.opacity.value(), smooth_shading=True, specular=.3)
        if self.unit_cell.isChecked():
            plotter.add_mesh(self.grid.outline(), color='#b9c7d9', opacity=.75, line_width=1.5)
        peak = max(abs(low), abs(high), 1e-30)
        for plane_data in self.planes:
            hkl, distance, reference = plane_data['hkl'], plane_data['distance'], plane_data['grid_origin']
            normal, point = self.field.plane(hkl, distance, reference)
            plane = self.grid.slice(normal=normal, origin=point)
            if plane.n_points:
                plotter.add_mesh(plane, color=plane_data['color'], opacity=plane_data['opacity'], lighting=False)
        if self._slice_preview:
            hkl, distance, reference = self._slice_preview; normal, point = self.field.plane(hkl, distance, reference)
            preview = self.grid.slice(normal=normal, origin=point)
            if preview.n_points: plotter.add_mesh(preview, color='#f2c14e', opacity=.45, lighting=False)
        if fit:
            plotter.view_isometric(); plotter.reset_camera(bounds=self.grid.bounds)
            plotter.enable_parallel_projection()
        else: self.backend.set_camera(camera)
        plotter.reset_camera_clipping_range(); plotter.render()
        self.info.setText(f'Grid {self.field.values.shape} • range {low:.6g} … {high:.6g} • blue + / red − surfaces • {len(self.planes)} slices. '
                          f'Display: {self.grid.n_points:,} samples; arithmetic and saved cube retain all {self.field.values.size:,} samples.')

    def dispose(self):
        if self.disposed: return
        self.disposed = True
        if self.backend: self.backend.close()
