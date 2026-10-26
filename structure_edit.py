"""Renderer-independent, atom-count-preserving geometry edits."""
import numpy as np
from structure_io import require_cell


def transform(atoms, indices, vector=None, axis=2, angle=0, pivot=None):
    ids = np.asarray(indices, dtype=int)
    if not len(ids):
        raise ValueError('Select one or more atoms first, or choose All atoms.')
    out = atoms.copy()
    if vector is not None:
        v = np.asarray(vector, dtype=float)
        if v.shape != (3,) or not np.isfinite(v).all():
            raise ValueError('Translation requires three finite values.')
        out.positions[ids] += v
    else:
        center = atoms.positions[ids].mean(axis=0) if pivot is None else np.asarray(pivot, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(angle):
            raise ValueError('Rotation requires a finite angle and pivot.')
        group = atoms[ids]
        group.rotate(angle, 'xyz'[axis], center=center, rotate_cell=False)
        out.positions[ids] = group.positions
    return out


def shift_termination(atoms, axis, direction, tolerance=.05):
    """Transfer a complete extreme layer to its opposite periodic image."""
    require_cell(atoms)
    if not atoms.pbc[axis]:
        raise ValueError('Enable periodicity on the shift axis first. A vacuum slab has no bulk repeat along its vacuum direction.')
    period = atoms.cell.array[axis]
    if not np.allclose(np.delete(period, axis), 0, atol=1e-8) or abs(period[axis]) < 1e-8:
        raise ValueError('Plane locking requires the corresponding lattice vector to be aligned with the Cartesian shift axis.')
    if direction not in (-1, 1) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Choose up/down and a positive layer tolerance.')
    positions = atoms.positions[:, axis]
    if np.ptp(positions) >= abs(period[axis]) - tolerance:
        raise ValueError('Atoms must occupy less than one period on this axis. Wrap the structure first and reduce the tolerance if needed.')
    edge = positions.min() if direction > 0 else positions.max()
    ids = np.flatnonzero(abs(positions - edge) <= tolerance)
    out = atoms.copy()
    out.positions[ids, axis] += direction * abs(period[axis])
    return out, ids
