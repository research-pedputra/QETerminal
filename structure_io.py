"""Structure IO; never manufacture a Quantum ESPRESSO calculation log."""
from pathlib import Path
import json
import numpy as np
from ase.io import read, write
from ase.io.espresso import read_fortran_namelist


def require_cell(atoms):
    if abs(np.linalg.det(atoms.cell.array)) < 1e-10:
        raise ValueError('This operation requires a nonsingular 3D cell. Set the cell first.')


def load_structure(path, format=None):
    path = Path(path)
    name = path.name.lower()
    if not format:
        if name.endswith(('.pwo', '.pwout', '.out')):
            format = 'espresso-out'
        elif name.endswith(('.pwi', '.pwin', '.in')):
            format = 'espresso-in'
        elif name in ('poscar', 'contcar') or name.endswith(('.vasp', '.poscar')):
            format = 'vasp'
    # Include final ATOMIC_POSITIONS even when no energy/SCF results follow it.
    kwargs = {'results_required': False} if format == 'espresso-out' else {}
    frames = read(str(path), index=':', format=format, **kwargs)
    if not isinstance(frames, list):
        frames = [frames]
    if not frames or any(len(a) == 0 for a in frames):
        raise ValueError('No nonempty structures found in this file.')
    metadata = {}
    if format == 'espresso-in':
        with path.open(encoding='utf-8') as f:
            namelist, cards = read_fortran_namelist(f)
        metadata['input_data'] = {k: dict(v) for k, v in namelist.items()}
        for i, line in enumerate(cards):
            if line.strip().upper().startswith('ATOMIC_SPECIES'):
                metadata['pseudopotentials'] = {
                    row.split()[0]: row.split()[2]
                    for row in cards[i + 1:i + 1 + int(namelist['system']['ntyp'])]
                }
            if line.strip().upper().startswith('K_POINTS'):
                if 'gamma' in line.lower():
                    metadata['kpts'] = None
                    metadata['koffset'] = [0, 0, 0]
                elif 'automatic' in line.lower():
                    values = [int(x) for x in cards[i + 1].split()[:6]]
                    metadata['kpts'], metadata['koffset'] = values[:3], values[3:]
    return frames, metadata, format


def geometry_cards(atoms, fractional):
    require_cell(atoms)
    lines = ['! Geometry only; NOT a pw.x calculation output.', 'CELL_PARAMETERS angstrom']
    lines += [' '.join(f'{x:.14f}' for x in row) for row in atoms.cell.array]
    lines += ['ATOMIC_POSITIONS ' + ('crystal' if fractional else 'angstrom')]
    xyz = atoms.get_scaled_positions(wrap=False) if fractional else atoms.positions
    lines += [s + ' ' + ' '.join(f'{x:.14f}' for x in p)
              for s, p in zip(atoms.get_chemical_symbols(), xyz)]
    return '\n'.join(lines) + '\n'


def export_structure(path, atoms, format, fractional=False, qe=None):
    if not len(atoms) or not np.isfinite(atoms.positions).all():
        raise ValueError('The structure must contain atoms with finite coordinates.')
    if format in ('espresso-in', 'qe-cards', 'vasp', 'cif') or fractional:
        require_cell(atoms)
    if format == 'qe-cards':
        Path(path).write_text(geometry_cards(atoms, fractional), encoding='utf-8')
    elif format == 'espresso-in':
        qe = json.loads(json.dumps(qe or {}))
        pseudo = qe.get('pseudopotentials', {})
        missing = set(atoms.get_chemical_symbols()) - set(pseudo)
        if missing or any(not str(v).strip() for v in pseudo.values()):
            raise ValueError('Set a UPF filename for every element in the QE settings. Missing: ' + ', '.join(sorted(missing)))
        data = qe.setdefault('input_data', {})
        system = data.setdefault('system', {})
        for key in list(system):
            if key.lower() in ('nat', 'ntyp', 'a', 'b', 'c', 'cosab', 'cosac', 'cosbc') or key.lower().startswith('celldm'):
                system.pop(key)
        system['ibrav'] = 0
        write(str(path), atoms, format='espresso-in', crystal_coordinates=fractional, **qe)
    elif format == 'vasp':
        write(str(path), atoms, format='vasp', direct=fractional, sort=True, vasp5=True)
    elif format == 'cif':
        # ASE normally writes fractional CIF. Supply valid Cartesian CIF explicitly.
        if fractional:
            write(str(path), atoms, format='cif')
        else:
            a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
            lines = ['data_structure']
            lines += [f'_cell_{key} {v:.14g}' for key, v in zip(
                ['length_a', 'length_b', 'length_c', 'angle_alpha', 'angle_beta', 'angle_gamma'],
                [a, b, c, alpha, beta, gamma])]
            lines += ["_symmetry_space_group_name_H-M 'P 1'", '_symmetry_Int_Tables_number 1',
                      'loop_', '_atom_site_label', '_atom_site_type_symbol',
                      '_atom_site_Cartn_x', '_atom_site_Cartn_y', '_atom_site_Cartn_z', '_atom_site_occupancy']
            # CIF cell parameters imply a canonical orientation; rotate positions into it.
            from ase.cell import Cell
            positions = atoms.get_scaled_positions(wrap=False) @ Cell.fromcellpar(atoms.cell.cellpar()).array
            lines += [f'{s}{i+1} {s} ' + ' '.join(f'{x:.14g}' for x in p) + ' 1'
                      for i, (s, p) in enumerate(zip(atoms.get_chemical_symbols(), positions))]
            Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    elif format in ('xyz', 'extxyz'):
        if fractional:
            raise ValueError('XYZ stores Cartesian coordinates only; use QE, CIF or VASP for fractional coordinates.')
        write(str(path), atoms, format=format, write_results=False) if format == 'extxyz' else write(str(path), atoms, format=format)
    else:
        raise ValueError('Unsupported export format: ' + format)
