"""PySide6 / ASE structure editor. Run: python structure_viewer.py [structure]."""
import sys
import json
from pathlib import Path
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets as W
from ase import Atom
from ase.build import bulk, surface, make_supercell
from structure_io import load_structure, export_structure, require_cell
from atom_scene import build_scene, camera_view
from render_backends import BACKENDS
from renderer_animation import save_backend_gif
from structure_edit import transform, shift_termination


class Viewer(W.QMainWindow):
    def __init__(self, renderer='PyVista / VTK'):
        super().__init__()
        self.setWindowTitle('Quantum Structure Studio • ASE / PySide6')
        self.resize(1450, 920)
        self.atoms = bulk('Fe', 'bcc', a=2.87, cubic=True)
        self.original = self.atoms.copy()
        self.frames = [self.atoms.copy()]
        self.baseline_frames = [self.atoms.copy()]
        self.active_frame = 0
        self.undo_stack, self.redo_stack = [], []
        self.source_output = None
        self.dirty = False
        self.updating = False
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(400)
        self.timer.timeout.connect(self.next_frame)
        self.backend = None
        self.workspace = None
        self._close_confirmed = False
        self.pair_rules = {}
        self.poly_settings = dict(show_polyhedra=False, poly_centers='Ti', poly_neighbors='O',
                                  poly_cutoff=2.5, poly_color=(.25, .65, .9), poly_opacity=.45, poly_shine=.5)
        self.build_ui()
        try:
            self.switch_renderer(renderer)
        except Exception as exc:
            self.switch_renderer('Matplotlib (basic)')
            self.statusBar().showMessage(f'{renderer} unavailable: {exc}. Using Matplotlib; install requirements-renderers.txt.')
        self.refresh(True)
        from file_workspace import FileWorkspace
        self.workspace = FileWorkspace(self)

    def button(self, layout, text, callback):
        b = W.QPushButton(text)
        b.clicked.connect(lambda: self.safe(callback))
        layout.addWidget(b)
        return b

    def safe(self, callback):
        try:
            callback()
        except Exception as exc:
            W.QMessageBox.warning(self, 'Operation could not be completed', str(exc))
            try:
                self.refresh()
            except Exception:
                # Keep the last valid scene if pending display controls are invalid.
                pass

    def build_ui(self):
        file_menu = self.menuBar().addMenu('File')
        edit_menu = self.menuBar().addMenu('Edit')
        self.file_menu = file_menu; self.edit_menu = edit_menu
        animation_menu = self.menuBar().addMenu('Animation')
        gif_action = animation_menu.addAction('Save trajectory as GIF…')
        gif_action.triggered.connect(lambda: self.safe(self.gif_dialog))
        for name, callback, shortcut in [('Open', self.open_file, 'Ctrl+O'),
                                         ('Export', self.export_dialog, 'Ctrl+Shift+E'),
                                         ('Undo', self.undo, 'Ctrl+Z'), ('Redo', self.redo, 'Ctrl+Y'),
                                         ('Reset frame', self.reset, ''),
                                         ('Save original QE log', self.save_log, '')]:
            menu = file_menu if name in ('Open', 'Export', 'Save original QE log') else edit_menu
            action = menu.addAction(name)
            action.triggered.connect(lambda checked=False, cb=callback: self.safe(cb))
            if shortcut:
                action.setShortcut(shortcut)
        split = W.QSplitter()
        self.setCentralWidget(split)
        left = W.QWidget()
        ll = W.QVBoxLayout(left)
        self.info = W.QLabel()
        self.info.setWordWrap(True)
        ll.addWidget(self.info)
        self.tabs = W.QTabWidget()
        ll.addWidget(self.tabs)
        split.addWidget(left)
        self.setup_atoms()
        self.setup_transforms()
        self.setup_build()
        self.setup_qe()
        self.setup_display()
        right = W.QWidget()
        rl = W.QVBoxLayout(right)
        engine_row = W.QHBoxLayout()
        engine_row.addWidget(W.QLabel('3D renderer'))
        self.renderer_choice = W.QComboBox(); self.renderer_choice.addItems(BACKENDS)
        self.renderer_choice.currentTextChanged.connect(lambda name: self.safe(lambda: self.switch_renderer(name)))
        engine_row.addWidget(self.renderer_choice, 1)
        self.button(engine_row, 'Save image', self.save_image)
        rl.addLayout(engine_row)
        controls = W.QHBoxLayout()
        self.show_cell = W.QCheckBox('Unit cell'); self.show_cell.setChecked(False)
        self.show_bonds = W.QCheckBox('Bonds'); self.show_bonds.setChecked(True)
        view_menu = self.menuBar().addMenu('View')
        self.bonds_action = view_menu.addAction('Show bonds')
        self.bonds_action.setCheckable(True); self.bonds_action.setChecked(True)
        self.bonds_action.toggled.connect(self.show_bonds.setChecked)
        self.show_bonds.toggled.connect(self.bonds_action.setChecked)
        view_menu.addAction('Polyhedra settings…').triggered.connect(lambda: self.safe(self.polyhedra_dialog))
        file_menu.addAction('Save PNG image…').triggered.connect(lambda: self.safe(self.save_image))
        self.show_labels = W.QCheckBox('Atom IDs')
        for cb in (self.show_cell, self.show_bonds, self.show_labels):
            controls.addWidget(cb)
            cb.toggled.connect(lambda: self.draw())
        controls.addWidget(W.QLabel('Atom size'))
        self.size = W.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.size.setRange(10, 100); self.size.setValue(40)
        self.size.valueChanged.connect(lambda: self.draw())
        controls.addWidget(self.size)
        rl.addLayout(controls)
        views = W.QHBoxLayout()
        for name, elev, azim in [('Isometric', 25, -60), ('Front XY', 90, -90), ('Side XZ', 0, -90), ('Top YZ', 0, 0)]:
            self.button(views, name, lambda e=elev, a=azim: self.set_view(e, a))
        self.button(views, 'Fit', lambda: self.draw(True))
        rl.addLayout(views)
        self.viewport_host = W.QWidget()
        self.viewport_layout = W.QVBoxLayout(self.viewport_host)
        self.viewport_layout.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.viewport_host, 1)
        self.render_help = W.QLabel('')
        rl.addWidget(self.render_help)
        timeline = W.QHBoxLayout()
        timeline.addWidget(W.QLabel('Optimization step / frame'))
        self.frame = W.QSpinBox(); self.frame.setRange(1, 1)
        self.frame.valueChanged.connect(lambda n: self.safe(lambda: self.change_frame(n)))
        timeline.addWidget(self.frame)
        self.frame_count = W.QLabel('/ 1'); timeline.addWidget(self.frame_count)
        self.play = self.button(timeline, 'Play / pause', self.toggle_play)
        self.button(timeline, 'Last geometry', lambda: self.frame.setValue(len(self.frames)))
        self.results = W.QLabel(); timeline.addWidget(self.results, 1)
        rl.addLayout(timeline)
        split.addWidget(right); split.setSizes([470, 980])
        self.statusBar().clearMessage()
        self.setStyleSheet('QMainWindow {background:#eef2f6;} QTabWidget::pane {border:1px solid #ccd5e0;} QPushButton {padding:6px;} QTableWidget {gridline-color:#e0e6ed;}')

    def tab(self, name):
        widget = W.QWidget(); layout = W.QVBoxLayout(widget)
        self.tabs.addTab(widget, name)
        return layout

    def setup_atoms(self):
        layout = self.tab('Atoms')
        self.coords = W.QComboBox(); self.coords.addItems(['Cartesian (Å)', 'Crystal / fractional'])
        self.coords.currentIndexChanged.connect(lambda: self.safe(self.refresh))
        layout.addWidget(self.coords)
        self.table = W.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Element', 'X', 'Y', 'Z'])
        self.table.setSelectionBehavior(W.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(W.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(lambda item: self.safe(lambda: self.edit_atom(item)))
        self.table.itemSelectionChanged.connect(self.selection_changed)
        layout.addWidget(self.table)
        row = W.QHBoxLayout()
        self.button(row, 'Delete selected', self.delete_atoms)
        self.button(row, 'Distance', self.measure)
        layout.addLayout(row)
        form = W.QFormLayout()
        self.symbol = W.QLineEdit('H'); self.position = W.QLineEdit('0 0 0')
        self.shift = W.QLineEdit('0 0 0')
        form.addRow('New element', self.symbol); form.addRow('New position', self.position)
        form.addRow('Shift vector', self.shift); layout.addLayout(form)
        layout.addWidget(W.QLabel('Positions and shifts use the coordinate mode above.'))
        row = W.QHBoxLayout()
        self.button(row, 'Add atom', self.add_atom)
        self.button(row, 'Shift selected / all', self.shift_atoms)
        self.button(row, 'Wrap', self.wrap)
        layout.addLayout(row)

    def setup_transforms(self):
        self.transform_tab = W.QScrollArea(); self.transform_tab.setWidgetResizable(True)
        content = W.QWidget(); layout = W.QVBoxLayout(content)
        self.transform_tab.setWidget(content); self.tabs.addTab(self.transform_tab, 'Move / rotate')
        menu = self.menuBar().addMenu('Structure')
        for label in ('Translate atoms', 'Rotate atoms', 'Surface termination'):
            menu.addAction(label).triggered.connect(self.show_transforms)
        note = W.QLabel('Select atoms in the Atoms table (Ctrl/Shift for groups). Values below use Cartesian axes. Apply commits one undoable edit to the current frame.')
        note.setWordWrap(True); layout.addWidget(note)
        self.transform_scope = W.QComboBox(); self.transform_scope.addItems(['Selected atoms', 'All atoms'])
        layout.addWidget(self.transform_scope)
        def amount_row(form, label, maximum, initial, scale):
            widget = W.QWidget(); row = W.QHBoxLayout(widget); row.setContentsMargins(0, 0, 0, 0)
            spin = W.QDoubleSpinBox(); spin.setDecimals(4); spin.setRange(0, maximum); spin.setValue(initial)
            slider = W.QSlider(QtCore.Qt.Orientation.Horizontal); slider.setRange(0, int(maximum * scale)); slider.setValue(round(initial * scale))
            slider.valueChanged.connect(lambda value: spin.setValue(value / scale))
            def sync(value):
                blocker = QtCore.QSignalBlocker(slider)
                slider.setValue(round(value * scale))
            spin.valueChanged.connect(sync)
            row.addWidget(slider); row.addWidget(spin); form.addRow(label, widget)
            return spin
        translate_box = W.QGroupBox('Translation'); form = W.QFormLayout(translate_box)
        self.move_axis = W.QComboBox(); self.move_axis.addItems(['X', 'Y', 'Z', 'XYZ vector'])
        form.addRow('Axis', self.move_axis)
        self.move_amount = amount_row(form, 'Distance (Å)', 20, .1, 100)
        self.move_vector = W.QLineEdit('0 0 0'); form.addRow('XYZ displacement (Å)', self.move_vector)
        self.move_direction = W.QComboBox(); self.move_direction.addItems(['Up / +', 'Down / −']); form.addRow('Direction', self.move_direction)
        b = W.QPushButton('Apply translation'); b.clicked.connect(lambda: self.safe(self.apply_translation)); form.addRow(b)
        layout.addWidget(translate_box)
        rotation_box = W.QGroupBox('Rotation (right-hand rule)'); form = W.QFormLayout(rotation_box)
        self.rotate_axis = W.QComboBox(); self.rotate_axis.addItems(['X', 'Y', 'Z']); form.addRow('Axis', self.rotate_axis)
        self.rotate_amount = amount_row(form, 'Angle (degrees)', 360, 5, 10)
        self.rotate_direction = W.QComboBox(); self.rotate_direction.addItems(['Positive / +', 'Negative / −']); form.addRow('Direction', self.rotate_direction)
        self.rotate_pivot = W.QComboBox(); self.rotate_pivot.addItems(['Selection center', 'Origin', 'Custom XYZ (Å)']); form.addRow('Pivot', self.rotate_pivot)
        self.pivot_position = W.QLineEdit('0 0 0'); form.addRow('Custom pivot', self.pivot_position)
        b = W.QPushButton('Apply rotation'); b.clicked.connect(lambda: self.safe(self.apply_rotation)); form.addRow(b)
        layout.addWidget(rotation_box)
        box = W.QGroupBox('Periodic surface termination — complete layers'); form = W.QFormLayout(box)
        self.termination_plane = W.QComboBox(); self.termination_plane.addItems(['Lock XY → shift Z', 'Lock XZ → shift Y', 'Lock YZ → shift X']); form.addRow('Plane', self.termination_plane)
        self.layer_tolerance = W.QDoubleSpinBox(); self.layer_tolerance.setDecimals(4); self.layer_tolerance.setRange(.0001, 5); self.layer_tolerance.setValue(.05); form.addRow('Layer tolerance (Å)', self.layer_tolerance)
        self.layer_direction = W.QComboBox(); self.layer_direction.addItems(['Up: bottom layer → top', 'Down: top layer → bottom']); form.addRow('Direction', self.layer_direction)
        note = W.QLabel('Uses all atoms to identify the outer layer; ignores selection. Moves that layer by one full cell period, preserving periodic equivalence and atom count. Requires a periodic, axis-aligned bulk repeat; do this before adding vacuum. Cell stays fixed, so transferred atoms can lie outside it. Wrap reverses the termination choice.')
        note.setWordWrap(True); form.addRow(note)
        b = W.QPushButton('Shift one complete layer'); b.clicked.connect(lambda: self.safe(self.apply_termination)); form.addRow(b)
        layout.addWidget(box); layout.addStretch()

    def show_transforms(self):
        if self.workspace:
            self.workspace.views.setCurrentIndex(0)
        self.tabs.setCurrentWidget(self.transform_tab)

    def transform_ids(self):
        return list(range(len(self.atoms))) if self.transform_scope.currentIndex() else self.selected()

    def commit_transform(self, atoms, ids):
        self.timer.stop()
        self.wrap_display.setChecked(False)
        self.commit(atoms)
        selection = self.table.selectionModel()
        self.table.blockSignals(True); self.table.clearSelection()
        for index in ids:
            selection.select(self.table.model().index(int(index), 0), QtCore.QItemSelectionModel.SelectionFlag.Select | QtCore.QItemSelectionModel.SelectionFlag.Rows)
        self.table.blockSignals(False); self.draw()

    def apply_translation(self):
        axis = self.move_axis.currentIndex()
        vector = self.vector(self.move_vector.text()) if axis == 3 else np.eye(3)[axis] * self.move_amount.value()
        vector *= 1 if self.move_direction.currentIndex() == 0 else -1
        ids = self.transform_ids()
        self.commit_transform(transform(self.atoms, ids, vector=vector), ids)

    def apply_rotation(self):
        pivot_mode = self.rotate_pivot.currentIndex()
        pivot = None if pivot_mode == 0 else (np.zeros(3) if pivot_mode == 1 else self.vector(self.pivot_position.text()))
        angle = self.rotate_amount.value() * (1 if self.rotate_direction.currentIndex() == 0 else -1)
        ids = self.transform_ids()
        self.commit_transform(transform(self.atoms, ids, axis=self.rotate_axis.currentIndex(), angle=angle, pivot=pivot), ids)

    def apply_termination(self):
        atoms, ids = shift_termination(self.atoms, [2, 1, 0][self.termination_plane.currentIndex()], 1 if self.layer_direction.currentIndex() == 0 else -1, self.layer_tolerance.value())
        self.commit_transform(atoms, ids)
        self.statusBar().showMessage(f'Shifted {len(ids)} atoms by one period; total remains {len(atoms)}.')

    def setup_build(self):
        layout = self.tab('Cell / build')
        layout.addWidget(W.QLabel('Cell vectors (rows, Å)'))
        self.cell_table = W.QTableWidget(3, 3)
        self.cell_table.setMaximumHeight(130)
        self.cell_table.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.cell_table)
        self.scale_atoms = W.QCheckBox('Scale atoms with cell (preserve fractional positions)')
        layout.addWidget(self.scale_atoms)
        row = W.QHBoxLayout(); self.pbc = []
        for axis in 'XYZ':
            cb = W.QCheckBox('Periodic ' + axis); self.pbc.append(cb); row.addWidget(cb)
        layout.addLayout(row)
        self.button(layout, 'Apply cell / periodicity', self.apply_cell)
        layout.addWidget(W.QLabel('Integer supercell transformation (3 × 3)'))
        self.matrix = W.QTableWidget(3, 3); self.matrix.setMaximumHeight(130)
        self.matrix.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch)
        for i in range(3):
            for j in range(3):
                self.matrix.setItem(i, j, W.QTableWidgetItem(str(int(i == j))))
        layout.addWidget(self.matrix)
        self.button(layout, 'Create supercell', self.supercell)
        form = W.QFormLayout()
        self.hkl = W.QLineEdit('1 0 0'); self.layers = W.QSpinBox(); self.layers.setRange(1, 200); self.layers.setValue(3)
        self.vacuum = W.QDoubleSpinBox(); self.vacuum.setRange(0, 200); self.vacuum.setValue(10)
        form.addRow('Miller indices', self.hkl); form.addRow('Layers', self.layers); form.addRow('Vacuum on EACH side (Å)', self.vacuum)
        layout.addLayout(form)
        note = W.QLabel('Slab uses the current structure. ASE surface() expects a conventional bulk cell.'); note.setWordWrap(True); layout.addWidget(note)
        self.button(layout, 'Create slab', self.slab)
        layout.addStretch()

    def setup_qe(self):
        layout = self.tab('QE settings')
        note = W.QLabel('Editable ASE writer settings (JSON). Set real UPF filenames and suitable cutoffs before running QE. Imported namelists and automatic/gamma k-points are loaded here; review other cards manually. No calculation is run by this app.')
        note.setWordWrap(True); layout.addWidget(note)
        self.qe = W.QPlainTextEdit()
        self.qe.setPlainText(json.dumps({'input_data': {'control': {'calculation': 'scf', 'prefix': 'structure', 'pseudo_dir': './pseudo'}, 'system': {'ecutwfc': 60, 'ecutrho': 480}, 'electrons': {'conv_thr': 1e-8}}, 'pseudopotentials': {'Fe': 'Fe.UPF'}, 'kpts': [4, 4, 4], 'koffset': [0, 0, 0]}, indent=2))
        layout.addWidget(self.qe)
        self.button(layout, 'Preview QE input', self.preview_qe)

    def setup_display(self):
        layout = self.tab('Display / bonds')
        note = W.QLabel('Display copies complete crystal boundaries, like a crystallographic viewer. They do not add atoms to your exported structure.')
        note.setWordWrap(True); layout.addWidget(note)
        form = W.QFormLayout()
        self.representation = W.QComboBox(); self.representation.addItems(['Ball and stick', 'Space filling', 'Sticks'])
        form.addRow('Representation', self.representation)
        self.boundary = W.QCheckBox('Complete atoms at periodic cell boundaries'); self.boundary.setChecked(True)
        form.addRow(self.boundary)
        self.wrap_display = W.QCheckBox('Wrap displayed atoms into cell'); self.wrap_display.setChecked(True)
        form.addRow(self.wrap_display)
        repeats = W.QWidget(); row = W.QHBoxLayout(repeats); row.setContentsMargins(0, 0, 0, 0)
        self.display_repeat = []
        for axis in 'abc':
            row.addWidget(W.QLabel(axis)); spin = W.QSpinBox(); spin.setRange(1, 4)
            self.display_repeat.append(spin); row.addWidget(spin)
        form.addRow('Display repetitions', repeats)
        self.bond_scale = W.QDoubleSpinBox(); self.bond_scale.setRange(.5, 2); self.bond_scale.setSingleStep(.05); self.bond_scale.setValue(1.05)
        self.bond_radius = W.QDoubleSpinBox(); self.bond_radius.setRange(.02, .5); self.bond_radius.setSingleStep(.02); self.bond_radius.setValue(.12)
        form.addRow('Covalent radius cutoff factor', self.bond_scale)
        form.addRow('Bond cylinder radius (Å)', self.bond_radius)
        layout.addLayout(form)
        self.button(layout, 'Apply display settings', lambda: self.draw(True))
        self.button(layout, 'Element-pair bond distances…', self.bond_rules_dialog)
        self.button(layout, 'Polyhedra: color / transparency / shine…', self.polyhedra_dialog)
        self.display_summary = W.QLabel(); layout.addWidget(self.display_summary)
        help_text = W.QLabel('PyVista and OVITO use smooth spheres and bonds colored by their endpoint atoms. Bond distances are geometric rules, not calculated bond orders. Select PyVista for atom picking and labels; OVITO selection uses the atom table. Matplotlib is a basic fallback.')
        help_text.setWordWrap(True); layout.addWidget(help_text); layout.addStretch()

    def polyhedra_dialog(self):
        from atom_scene import suggest_octahedra
        dialog = W.QDialog(self); dialog.setWindowTitle('Coordination polyhedra')
        form = W.QFormLayout(dialog); settings = dict(self.poly_settings)
        present = set(self.atoms.get_chemical_symbols())
        if not set(settings['poly_centers'].split()) <= present or not set(settings['poly_neighbors'].split()) <= present:
            try:
                settings.update(suggest_octahedra(self.atoms)); settings['show_polyhedra'] = True
            except ValueError:
                pass
        available = W.QLabel('Elements in this structure: ' + ', '.join(sorted(present))); form.addRow(available)
        enabled = W.QCheckBox('Display coordination polyhedra'); enabled.setChecked(settings['show_polyhedra']); form.addRow(enabled)
        centers = W.QLineEdit(settings['poly_centers']); neighbors = W.QLineEdit(settings['poly_neighbors'])
        form.addRow('Center elements (e.g. Ti Fe)', centers); form.addRow('Neighbor elements (e.g. O F)', neighbors)
        cutoff = W.QDoubleSpinBox(); cutoff.setRange(.1, 10); cutoff.setDecimals(3); cutoff.setValue(settings['poly_cutoff']); form.addRow('Neighbor cutoff (Å)', cutoff)
        diagnostic = W.QLabel(); diagnostic.setWordWrap(True)
        def check_neighbors():
            try:
                style = self.render_style()
                style.update(show_polyhedra=True, poly_centers=centers.text(), poly_neighbors=neighbors.text(), poly_cutoff=cutoff.value())
                scene = build_scene(self.atoms, **style)
                from collections import Counter
                counts = Counter(len(points) for points, faces in scene.polyhedra)
                diagnostic.setText(('Found ' + str(len(scene.polyhedra)) + ' polyhedra: ' + ', '.join(f'{count} with {n} neighbors' for n, count in sorted(counts.items()))) if counts else 'No polyhedra: check elements and increase cutoff. Four non-coplanar neighbors are needed; an octahedron needs six.')
            except ValueError as exc:
                diagnostic.setText(str(exc))
        def detect():
            try:
                proposal = suggest_octahedra(self.atoms)
                centers.setText(proposal['poly_centers']); neighbors.setText(proposal['poly_neighbors']); cutoff.setValue(proposal['poly_cutoff'])
                enabled.setChecked(True); check_neighbors()
            except ValueError as exc:
                diagnostic.setText(str(exc))
        detect_button = W.QPushButton('Suggest octahedral coordination'); detect_button.clicked.connect(detect); form.addRow(detect_button)
        check_button = W.QPushButton('Check neighbors'); check_button.clicked.connect(check_neighbors); form.addRow(check_button); form.addRow(diagnostic)
        check_neighbors()
        color = QtGui.QColor.fromRgbF(*settings['poly_color'])
        color_button = W.QPushButton()
        def color_style():
            color_button.setText(color.name()); color_button.setStyleSheet(f'background-color: {color.name()}; color: {"black" if color.lightnessF() > .5 else "white"};')
        def choose_color():
            nonlocal color
            chosen = W.QColorDialog.getColor(color, dialog, 'Polyhedron face color')
            if chosen.isValid(): color = chosen; color_style()
        color_style(); color_button.clicked.connect(choose_color); form.addRow('Face color', color_button)
        transparency = W.QSpinBox(); transparency.setRange(0, 100); transparency.setSuffix(' %'); transparency.setValue(round((1-settings['poly_opacity'])*100))
        form.addRow('Transparency (0 = opaque)', transparency)
        shine = W.QSlider(QtCore.Qt.Orientation.Horizontal); shine.setRange(0, 100); shine.setValue(round(settings['poly_shine']*100))
        form.addRow('Shininess: matte → glossy', shine)
        note = W.QLabel('Polyhedra enclose neighboring atoms within the cutoff, including periodic images. At least four non-coplanar neighbors are required. Color and transparency work in all renderers; glossy highlights require PyVista / VTK. These are display surfaces and do not change the structure.')
        note.setWordWrap(True); form.addRow(note)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Ok | W.QDialogButtonBox.StandardButton.Cancel)
        def validate():
            if enabled.isChecked():
                try:
                    style = self.render_style(); style.update(show_polyhedra=True, poly_centers=centers.text(), poly_neighbors=neighbors.text(), poly_cutoff=cutoff.value())
                    if not build_scene(self.atoms, **style).polyhedra:
                        check_neighbors(); return
                    if transparency.value() == 100:
                        diagnostic.setText('100% transparency hides every face. Lower transparency to see the polyhedra.'); return
                except ValueError as exc:
                    diagnostic.setText(str(exc)); return
            dialog.accept()
        buttons.accepted.connect(validate); buttons.rejected.connect(dialog.reject); form.addRow(buttons)
        if dialog.exec() != W.QDialog.DialogCode.Accepted: return
        pending = dict(show_polyhedra=enabled.isChecked(), poly_centers=centers.text().strip(),
                       poly_neighbors=neighbors.text().strip(), poly_cutoff=cutoff.value(),
                       poly_color=(color.redF(), color.greenF(), color.blueF()),
                       poly_opacity=1-transparency.value()/100, poly_shine=shine.value()/100)
        style = self.render_style(); style.update(pending)
        scene = build_scene(self.atoms, **style)
        self.poly_settings = pending; self.draw(True)
        if enabled.isChecked():
            self.statusBar().showMessage(f'Displaying {len(scene.polyhedra)} polyhedra.' if scene.polyhedra else 'No polyhedra found: check center/neighbor elements and cutoff; four non-coplanar neighbors are required.')

    def bond_rules_dialog(self):
        from ase.data import atomic_numbers
        dialog = W.QDialog(self); dialog.setWindowTitle('Element-pair bond distances'); dialog.resize(520, 360)
        layout = W.QVBoxLayout(dialog)
        layout.addWidget(W.QLabel('One pair per line: Element Element minimum_Å maximum_Å\nExample: Fe O 0.5 2.4\nSet maximum to 0 to hide a pair. Unlisted pairs use the radius factor.'))
        editor = W.QPlainTextEdit('\n'.join(f'{a} {b} {lo:g} {hi:g}' for (a, b), (lo, hi) in self.pair_rules.items()))
        layout.addWidget(editor)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Ok | W.QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(buttons); buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        if dialog.exec() != W.QDialog.DialogCode.Accepted: return
        rules = {}
        for line in editor.toPlainText().splitlines():
            if not line.strip(): continue
            a, b, low, high = line.split(); low, high = float(low), float(high)
            if a not in atomic_numbers or b not in atomic_numbers or not np.isfinite([low, high]).all() or low < 0 or high < low:
                raise ValueError('Use valid elements and finite distances: 0 ≤ minimum ≤ maximum.')
            rules[tuple(sorted((a, b)))] = (low, high)
        self.pair_rules = rules; self.draw()

    def fractional(self):
        if self.coords.currentIndex() == 1:
            require_cell(self.atoms)
            return True
        return False

    @staticmethod
    def vector(text, integer=False):
        values = np.array([int(x) if integer else float(x) for x in text.replace(',', ' ').split()])
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError('Enter three finite numbers separated by spaces.')
        return values

    def selected(self):
        return sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})

    def commit(self, atoms):
        if not len(atoms) or len(atoms) > 20000:
            raise ValueError('Keep between 1 and 20,000 atoms for this desktop viewer.')
        if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
            raise ValueError('Coordinates and cell must be finite.')
        self.undo_stack.append(self.atoms.copy()); self.undo_stack = self.undo_stack[-40:]
        self.redo_stack.clear(); atoms.calc = None
        self.atoms = atoms; self.dirty = True; self.refresh(True)

    def undo(self):
        if self.workspace and self.workspace.views.currentWidget() in self.workspace.cubes:
            self.workspace.views.currentWidget().undo(); return
        if self.workspace and self.workspace.current():
            self.workspace.current().editor.undo(); return
        if self.undo_stack:
            self.redo_stack.append(self.atoms.copy()); self.atoms = self.undo_stack.pop(); self.dirty = True; self.refresh(True)

    def redo(self):
        if self.workspace and self.workspace.views.currentWidget() in self.workspace.cubes:
            self.workspace.views.currentWidget().undo(); return
        if self.workspace and self.workspace.current():
            self.workspace.current().editor.redo(); return
        if self.redo_stack:
            self.undo_stack.append(self.atoms.copy()); self.atoms = self.redo_stack.pop(); self.dirty = True; self.refresh(True)

    def reset(self):
        if self.workspace and self.workspace.views.currentWidget() in self.workspace.cubes:
            self.workspace.views.currentWidget().reset_field(); return
        self.commit(self.original.copy())

    def edit_atom(self, item):
        if self.updating:
            return
        a = self.atoms.copy(); row, col = item.row(), item.column()
        if col == 0:
            a[row].symbol = item.text().strip()
        else:
            xyz = a.get_scaled_positions(wrap=False) if self.fractional() else a.positions.copy()
            xyz[row, col - 1] = float(item.text())
            if self.fractional():
                a.set_scaled_positions(xyz)
            else:
                a.set_positions(xyz, apply_constraint=False)
        self.commit(a)

    def add_atom(self):
        a = self.atoms.copy(); v = self.vector(self.position.text())
        if self.fractional(): v = v @ a.cell.array
        a.append(Atom(self.symbol.text().strip(), v)); self.commit(a)

    def delete_atoms(self):
        indices = self.selected()
        if not indices: raise ValueError('Select atom rows first.')
        a = self.atoms.copy(); del a[indices]; self.commit(a)

    def shift_atoms(self):
        a = self.atoms.copy(); v = self.vector(self.shift.text())
        if self.fractional(): v = v @ a.cell.array
        a.positions[self.selected() or list(range(len(a)))] += v; self.commit(a)

    def wrap(self):
        require_cell(self.atoms); a = self.atoms.copy(); a.wrap(); self.commit(a)

    def apply_cell(self):
        cell = np.array([[float(self.cell_table.item(i, j).text()) for j in range(3)] for i in range(3)])
        if not np.isfinite(cell).all() or abs(np.linalg.det(cell)) < 1e-10:
            raise ValueError('Cell vectors must be finite and linearly independent.')
        a = self.atoms.copy(); a.set_cell(cell, scale_atoms=self.scale_atoms.isChecked()); a.pbc = [c.isChecked() for c in self.pbc]; self.commit(a)

    def supercell(self):
        require_cell(self.atoms)
        matrix = np.array([[int(self.matrix.item(i, j).text()) for j in range(3)] for i in range(3)])
        count = round(abs(np.linalg.det(matrix))) * len(self.atoms)
        if count < 1 or count > 20000: raise ValueError('Choose a nonsingular matrix producing at most 20,000 atoms.')
        self.commit(make_supercell(self.atoms, matrix))

    def slab(self):
        require_cell(self.atoms); hkl = self.vector(self.hkl.text(), True)
        if not hkl.any(): raise ValueError('Miller indices cannot all be zero.')
        if len(self.atoms) * self.layers.value() > 20000: raise ValueError('Too many atoms (limit 20,000).')
        a = surface(self.atoms, hkl, self.layers.value(), vacuum=self.vacuum.value()); self.commit(a)

    def measure(self):
        ids = self.selected()
        if len(ids) != 2: raise ValueError('Select exactly two atoms (Ctrl-click rows).')
        distance = self.atoms.get_distance(*ids, mic=bool(self.atoms.pbc.any()))
        W.QMessageBox.information(self, 'Distance', f'Atoms {ids[0]+1}–{ids[1]+1}: {distance:.6f} Å\nMinimum-image convention on periodic axes.')

    def refresh(self, fit=False):
        if self.coords.currentIndex() and abs(np.linalg.det(self.atoms.cell.array)) < 1e-10:
            self.coords.blockSignals(True); self.coords.setCurrentIndex(0); self.coords.blockSignals(False)
        self.updating = True; self.table.blockSignals(True)
        self.table.setRowCount(len(self.atoms))
        xyz = self.atoms.get_scaled_positions(wrap=False) if self.fractional() else self.atoms.positions
        for i, (sym, pos) in enumerate(zip(self.atoms.get_chemical_symbols(), xyz)):
            for j, val in enumerate([sym] + [f'{v:.10f}' for v in pos]):
                self.table.setItem(i, j, W.QTableWidgetItem(val))
        for i in range(3):
            self.pbc[i].setChecked(bool(self.atoms.pbc[i]))
            for j in range(3): self.cell_table.setItem(i, j, W.QTableWidgetItem(f'{self.atoms.cell[i,j]:.10f}'))
        self.table.blockSignals(False); self.updating = False
        self.info.setText(f'{self.atoms.get_chemical_formula()} · {len(self.atoms)} atoms\nCell lengths: ' + ', '.join(f'{x:.3f}' for x in self.atoms.cell.lengths()) + ' Å\nAngles: ' + ', '.join(f'{x:.2f}°' for x in self.atoms.cell.angles()))
        results = getattr(self.atoms.calc, 'results', {})
        energy = results.get('energy')
        self.results.setText(f'Energy: {energy:.6f} eV' if energy is not None else 'No energy for this geometry')
        self.draw(fit)

    def selection_changed(self):
        if not self.updating: self.draw()

    def switch_renderer(self, name):
        if self.backend is not None and self.backend.name == name: return
        old = self.backend
        camera = old.get_camera() if old is not None else None
        new = None
        try:
            new = BACKENDS[name](self.viewport_host, lambda index: self.table.selectRow(index))
            new.render(build_scene(self.atoms, **self.render_style(), selected=self.selected()), fit=True)
            if camera is not None: new.set_camera(camera)
        except Exception:
            if new is not None: new.close()
            if old is not None:
                self.renderer_choice.blockSignals(True); self.renderer_choice.setCurrentText(old.name); self.renderer_choice.blockSignals(False)
            raise
        if old is not None:
            self.viewport_layout.removeWidget(old.widget); old.close(); old.widget.deleteLater()
        self.backend = new
        self.viewport_layout.addWidget(new.widget)
        new.widget.show()
        self.renderer_choice.blockSignals(True); self.renderer_choice.setCurrentText(name); self.renderer_choice.blockSignals(False)
        self.render_help.setText('Drag to rotate • scroll to zoom • select atoms in the table' + (' or click spheres' if name != 'OVITO' else ' • OVITO native navigation'))
        self.show_labels.setEnabled(name != 'OVITO')
        self.show_labels.setToolTip('Atom labels are available in PyVista and Matplotlib.' if name == 'OVITO' else '')
        self.statusBar().clearMessage()

    def draw(self, fit=False):
        if self.backend is None: return
        scene = build_scene(self.atoms, **self.render_style(), selected=self.selected())
        self.backend.render(scene, fit=fit)
        self.display_summary.setText(f'{len(scene.positions)} displayed atoms · {len(scene.bonds)} bonds\n{len(self.atoms)} atoms in the structure file')

    def render_style(self):
        return dict(size=self.size.value(), show_cell=self.show_cell.isChecked(),
                    show_bonds=self.show_bonds.isChecked(), show_labels=self.show_labels.isChecked(),
                    repeat=tuple(spin.value() for spin in self.display_repeat), boundary=self.boundary.isChecked(),
                    bond_scale=self.bond_scale.value(), bond_radius=self.bond_radius.value(),
                    pair_rules=self.pair_rules, representation=self.representation.currentText(),
                    wrap_display=self.wrap_display.isChecked(), **self.poly_settings)

    def set_view(self, elev, azim):
        if self.backend is not None:
            self.backend.set_camera(camera_view(self.backend.get_camera(), elev, azim))

    def save_image(self):
        from PIL import Image
        dialog = W.QDialog(self); dialog.setWindowTitle('PNG image options')
        form = W.QFormLayout(dialog)
        width = W.QSpinBox(); width.setRange(64, 8192)
        height = W.QSpinBox(); height.setRange(64, 8192)
        previous = getattr(self, 'png_options', (1600, 1200, False))
        width.setValue(previous[0]); height.setValue(previous[1])
        transparent = W.QCheckBox('Transparent background (alpha channel)'); transparent.setChecked(previous[2])
        form.addRow('Width (pixels)', width); form.addRow('Height (pixels)', height)
        form.addRow(transparent)
        note = W.QLabel('Exports the current camera and bond visibility.\nUnchecked transparency uses a white background.')
        note.setWordWrap(True); form.addRow(note)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Save | W.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); form.addRow(buttons)
        if dialog.exec() != W.QDialog.DialogCode.Accepted: return
        path, _ = W.QFileDialog.getSaveFileName(self, 'Save 3D image', 'structure.png', 'PNG image (*.png)')
        if path:
            if not path.lower().endswith('.png'): path += '.png'
            pixels = self.backend.image(build_scene(self.atoms, **self.render_style()), self.backend.get_camera(), width.value(), height.value(), transparent=transparent.isChecked())
            Image.fromarray(pixels).save(path)
            self.png_options = (width.value(), height.value(), transparent.isChecked())
            self.statusBar().showMessage('Saved image: ' + path)

    def confirm_discard(self):
        return not self.dirty or W.QMessageBox.question(self, 'Unsaved edits', 'Discard unsaved structure edits?', W.QMessageBox.StandardButton.Discard | W.QMessageBox.StandardButton.Cancel) == W.QMessageBox.StandardButton.Discard

    def open_file(self):
        if self.workspace:
            self.workspace.guard(self.workspace.open_dialog); return
        path, selected = W.QFileDialog.getOpenFileName(self, 'Open structure or trajectory', '', 'Supported structures (*.in *.pwi *.out *.pwo *.pwout *.cif *.gz *.xyz *.extxyz *.vasp *.poscar POSCAR CONTCAR);;All files (*)')
        if not path or not self.confirm_discard(): return
        self.load_path(path)

    def load_path(self, path):
        frames, metadata, fmt = load_structure(path)
        if any(len(a) > 20000 for a in frames): raise ValueError('A frame exceeds the 20,000 atom display limit.')
        self.timer.stop(); self.frames = frames
        self.baseline_frames = [a.copy() for a in frames]
        self.source_output = Path(path).read_bytes() if fmt == 'espresso-out' else None
        if metadata: self.qe.setPlainText(json.dumps(metadata, indent=2))
        self.frame.blockSignals(True); self.frame.setRange(1, len(frames)); self.frame.setValue(len(frames)); self.frame.blockSignals(False)
        self.frame_count.setText(f'/ {len(frames)}'); self.dirty = False
        self.change_frame(len(frames))
        self.statusBar().showMessage(f'Loaded {path} — displaying final geometry ({len(frames)} of {len(frames)})')

    def change_frame(self, n):
        # Each trajectory frame retains its edits in memory when navigating.
        old = getattr(self, 'active_frame', None)
        if old is not None and old < len(self.frames) and self.dirty:
            self.frames[old] = self.atoms.copy()
            self.frames[old].calc = self.atoms.calc
        self.active_frame = n-1; self.atoms = self.frames[n-1].copy()
        self.atoms.calc = self.frames[n-1].calc
        self.original = self.baseline_frames[n-1].copy(); self.undo_stack.clear(); self.redo_stack.clear()
        self.refresh(True)

    def next_frame(self):
        self.frame.setValue(self.frame.value() % len(self.frames) + 1)

    def toggle_play(self):
        if self.timer.isActive(): self.timer.stop()
        elif len(self.frames) > 1: self.timer.start()

    def gif_dialog(self):
        self.timer.stop()
        dialog = W.QDialog(self); dialog.setWindowTitle('Save trajectory as GIF')
        layout = W.QVBoxLayout(dialog)
        note = W.QLabel(f'Animate the original imported coordinates ({len(self.baseline_frames)} frames).\nUses the current camera, atom size, bonds, labels, and unit-cell visibility.\nEdits in the structure table are not included.')
        note.setWordWrap(True); layout.addWidget(note)
        form = W.QFormLayout()
        start = W.QSpinBox(); start.setRange(1, len(self.baseline_frames))
        end = W.QSpinBox(); end.setRange(1, len(self.baseline_frames)); end.setValue(len(self.baseline_frames))
        step = W.QSpinBox(); step.setRange(1, len(self.baseline_frames))
        fps = W.QSpinBox(); fps.setRange(1, 50); fps.setValue(5)
        width = W.QSpinBox(); width.setRange(200, 1600); width.setSingleStep(100); width.setValue(800)
        loop = W.QCheckBox('Loop continuously'); loop.setChecked(True)
        for label, widget in [('First frame', start), ('Last frame', end), ('Frame step', step), ('Frames per second', fps), ('Image width (pixels)', width)]:
            form.addRow(label, widget)
        layout.addLayout(form); layout.addWidget(loop)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Save | W.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); layout.addWidget(buttons)
        if dialog.exec() != W.QDialog.DialogCode.Accepted: return
        if start.value() > end.value(): raise ValueError('First frame must not exceed last frame.')
        path, _ = W.QFileDialog.getSaveFileName(self, 'Save trajectory animation', 'trajectory.gif', 'GIF animation (*.gif)')
        if not path: return
        if not path.lower().endswith('.gif'): path += '.gif'
        indices = list(range(start.value()-1, end.value(), step.value()))
        # Always include the requested endpoint, including the final QE coordinates.
        if indices[-1] != end.value()-1: indices.append(end.value()-1)
        progress = W.QProgressDialog('Rendering trajectory…', 'Cancel', 0, len(indices), self)
        progress.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        progress.setAutoClose(False); progress.setAutoReset(False); progress.setMinimumDuration(0)
        def update(done, total):
            progress.setLabelText('Encoding GIF…' if done == total else f'Rendering frame {done+1} of {total}…')
            progress.setValue(done)
            W.QApplication.processEvents()
            return not progress.wasCanceled()
        self.centralWidget().setEnabled(False); self.file_toolbar.setEnabled(False); self.menuBar().setEnabled(False)
        try:
            saved = save_backend_gif(path, [self.baseline_frames[i] for i in indices], self.backend,
                                        frame_numbers=[i+1 for i in indices], fps=fps.value(),
                                        width=width.value(), loop=loop.isChecked(), progress=update,
                                        style=self.render_style())
            self.statusBar().showMessage('Saved GIF: ' + path if saved else 'GIF export canceled.')
        finally:
            progress.close(); self.centralWidget().setEnabled(True)
            self.file_toolbar.setEnabled(True); self.menuBar().setEnabled(True)

    def export_dialog(self):
        dialog = W.QDialog(self); dialog.setWindowTitle('Export current structure')
        layout = W.QVBoxLayout(dialog)
        formats = W.QComboBox()
        options = [('QE input (.in)', 'espresso-in', '.in'), ('QE geometry cards (.qe-cards)', 'qe-cards', '.qe-cards'), ('CIF (.cif)', 'cif', '.cif'), ('XYZ (.xyz)', 'xyz', '.xyz'), ('Extended XYZ (.extxyz)', 'extxyz', '.extxyz'), ('VASP / POSCAR (.vasp)', 'vasp', '.vasp')]
        formats.addItems([x[0] for x in options]); layout.addWidget(formats)
        mode = W.QComboBox(); mode.addItems(['Cartesian (Å)', 'Crystal / fractional']); mode.setCurrentIndex(self.coords.currentIndex()); layout.addWidget(mode)
        note = W.QLabel('QE geometry cards contain no calculated energies or forces.\nXYZ is Cartesian only. CIF uses a canonical cell orientation.\nExport saves the current frame.'); note.setWordWrap(True); layout.addWidget(note)
        def update_mode():
            xyz = options[formats.currentIndex()][1] in ('xyz', 'extxyz')
            if xyz: mode.setCurrentIndex(0)
            mode.setEnabled(not xyz)
        formats.currentIndexChanged.connect(update_mode)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Save | W.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); layout.addWidget(buttons)
        if dialog.exec() != W.QDialog.DialogCode.Accepted: return
        label, fmt, suffix = options[formats.currentIndex()]
        path, _ = W.QFileDialog.getSaveFileName(self, 'Save ' + label, 'structure' + suffix, f'{label} (*{suffix});;All files (*)')
        if path:
            if not Path(path).suffix: path += suffix
            qe = json.loads(self.qe.toPlainText()) if fmt == 'espresso-in' else None
            export_structure(path, self.atoms, fmt, mode.currentIndex() == 1, qe)
            self.statusBar().showMessage('Exported ' + path)
            # Export is per-frame and may be lossy; retain the document dirty flag.

    def preview_qe(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'preview.in'
            export_structure(path, self.atoms, 'espresso-in', self.fractional(), json.loads(self.qe.toPlainText()))
            text = path.read_text()
        dialog = W.QDialog(self); dialog.setWindowTitle('QE input preview'); dialog.resize(700, 650)
        layout = W.QVBoxLayout(dialog); editor = W.QPlainTextEdit(text); editor.setReadOnly(True); layout.addWidget(editor); dialog.exec()

    def save_log(self):
        if self.source_output is None: raise ValueError('Open a QE output first. This saves the original, unmodified calculation log; edited structures must be exported as input or geometry cards.')
        path, _ = W.QFileDialog.getSaveFileName(self, 'Copy original QE output (unmodified)', 'original.pwout', 'QE output (*.pwout *.out)')
        if path: Path(path).write_bytes(self.source_output)

    def closeEvent(self, event):
        if not self._close_confirmed:
            if not self.confirm_discard() or (self.workspace and not self.workspace.can_close()):
                event.ignore(); return
            self._close_confirmed = True
        self.timer.stop()
        if self.workspace and not self.workspace.shutdown():
            event.ignore()
            QtCore.QTimer.singleShot(100, self.close)
            return
        if self.workspace: self.workspace.temp.cleanup()
        if self.backend is not None: self.backend.close()
        event.accept()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Interactive QE / ASE structure editor')
    parser.add_argument('structure', nargs='?')
    parser.add_argument('--renderer', choices=['pyvista', 'ovito', 'matplotlib'], default='pyvista')
    args = parser.parse_args()
    app = W.QApplication(sys.argv)
    app.setOrganizationName('StructureStudio')
    app.setApplicationName('QuantumStructureStudio')
    app.setStyle('Fusion')
    if sys.platform == 'win32':
        app.setFont(QtGui.QFont('Segoe UI', 10))
    names = dict(pyvista='PyVista / VTK', ovito='OVITO', matplotlib='Matplotlib (basic)')
    window = Viewer(renderer=names[args.renderer]); window.show()
    if args.structure:
        window.safe(lambda: window.load_path(args.structure))
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
