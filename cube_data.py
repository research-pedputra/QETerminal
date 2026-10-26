"""Scalar Gaussian/QE cube fields, with explicit grid geometry and safe arithmetic."""
from dataclasses import dataclass, replace
import io
import warnings
import numpy as np
from ase import Atoms
from ase.units import Bohr

MAX_POINTS = 8_000_000
MAX_BYTES = 256 * 1024 * 1024


@dataclass
class CubeData:
    atoms: Atoms
    values: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray
    header: str

    @property
    def cell(self): return np.array(self.values.shape)[:, None] * self.spacing

    def arithmetic(self, operation, operand):
        if isinstance(operand, CubeData):
            if self.values.shape != operand.values.shape or not all(
                    np.allclose(a, b, rtol=1e-8, atol=1e-8) for a, b in
                    [(self.origin, operand.origin), (self.spacing, operand.spacing)]):
                raise ValueError('Cube grids differ: shape, origin and voxel vectors must match. No implicit resampling is performed.')
            operand = operand.values
        if not np.isfinite(operand).all(): raise ValueError('Operand must contain finite numbers.')
        if operation == '/' and np.any(np.asarray(operand) == 0):
            raise ValueError('Division by zero: the field has not been changed.')
        functions = {'+': np.add, '-': np.subtract, '*': np.multiply, '/': np.divide}
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            values = functions[operation](self.values, operand)
        if not np.isfinite(values).all(): raise ValueError('Result contains invalid values.')
        return replace(self, values=values)

    def plane(self, hkl, distance, grid_origin=False):
        hkl = np.asarray(hkl, dtype=float)
        if not np.any(hkl): raise ValueError('(0 0 0) does not define a plane.')
        normal = hkl @ np.linalg.inv(self.cell).T
        normal /= np.linalg.norm(normal)
        origin = self.origin if grid_origin else np.zeros(3)
        return normal, origin + float(distance)*normal

    def grid(self, max_points=250_000):
        """VTK uses x-fast ordering; cube text uses z-fast ordering. Retain endpoints."""
        import pyvista as pv
        counts = np.array(self.values.shape)
        while np.prod(counts) > max_points:
            axis = int(np.argmax(counts)); counts[axis] = max(2, (counts[axis]+1)//2)
        indices = [np.unique(np.linspace(0, n-1, num=int(count), dtype=int)) for n, count in zip(self.values.shape, counts)]
        ijk = np.stack(np.meshgrid(*indices, indexing='ij'), axis=-1)
        points = self.origin + ijk @ self.spacing
        grid = pv.StructuredGrid(*(points[..., k] for k in range(3)))
        grid.point_data['Field'] = self.values[np.ix_(*indices)].ravel(order='F')
        return grid

    def payload(self):
        stream = io.StringIO(); stream.write(self.header)
        flat = self.values.ravel()
        end = len(flat)//6*6
        np.savetxt(stream, flat[:end].reshape(-1, 6), fmt='%.12e')
        if end < len(flat): np.savetxt(stream, flat[end:][None, :], fmt='%.12e')
        data = stream.getvalue().encode('ascii')
        if len(data) > MAX_BYTES: raise ValueError('Export exceeds the 256 MiB cube limit.')
        return data


def read_cube_bytes(data):
    if len(data) > MAX_BYTES: raise ValueError('Cube limit is 256 MiB.')
    stream = io.StringIO(data.decode('ascii'))
    header = [stream.readline() for _ in range(6)]
    if not all(header): raise ValueError('Incomplete cube header.')
    comment = header[1].upper()
    if 'OUTER LOOP' in comment and [v[0] for v in comment.split()[2::3]] != ['X', 'Y', 'Z']:
        raise ValueError('Only X/Y/Z cube loop ordering is supported.')
    if 'CASTEP2CUBE' in comment: raise ValueError('CASTEP duplicate-boundary cubes are not supported by this QE cube editor.')
    fields = header[2].split(); natoms = int(fields[0])
    if natoms < 0 or (len(fields) > 4 and int(fields[4]) != 1):
        raise ValueError('Open a single scalar cube; multi-orbital/multi-field cubes are not supported.')
    if natoms > 20000: raise ValueError('Cube contains more than 20,000 atoms.')
    vectors = [line.split() for line in header[3:6]]
    counts = np.array([int(v[0]) for v in vectors])
    if not (np.all(counts > 0) or np.all(counts < 0)):
        raise ValueError('Cube axis counts must have the same nonzero sign.')
    unit = Bohr if np.all(counts > 0) else 1.0
    shape = np.abs(counts)
    if np.any(shape < 2) or np.prod(shape.astype(object)) > MAX_POINTS:
        raise ValueError('Cube requires at least 2 samples per axis and at most 8 million grid points.')
    number = lambda text: float(text.replace('D', 'E').replace('d', 'e'))
    origin = np.array([number(v) for v in fields[1:4]])*unit
    spacing = np.array([[number(v) for v in line[1:4]] for line in vectors])*unit
    if not np.isfinite(spacing).all() or abs(np.linalg.det(spacing)) < 1e-15:
        raise ValueError('Cube voxel vectors must define a finite, nonsingular cell.')
    numbers, positions = [], []
    for _ in range(natoms):
        line = stream.readline(); header.append(line); row = line.split()
        numbers.append(int(row[0])); positions.append([number(v)*unit for v in row[2:5]])
    if any(n < 0 or n > 118 for n in numbers): raise ValueError('Invalid cube atomic number.')
    if not np.isfinite(origin).all() or not np.isfinite(positions).all(): raise ValueError('Invalid atomic coordinates or origin.')
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        values = np.fromstring(stream.read().replace('D', 'E').replace('d', 'e'), sep=' ')
    if values.size != int(np.prod(shape)) or not np.isfinite(values).all():
        raise ValueError('Cube scalar count does not match its grid, or values are nonfinite.')
    atoms = Atoms(numbers=numbers, positions=np.array(positions).reshape(-1, 3), cell=shape[:, None]*spacing, pbc=False)
    return CubeData(atoms, values.reshape(tuple(shape)), origin, spacing, ''.join(header))
