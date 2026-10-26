"""Renderer-independent display geometry. Display images never modify the ASE document."""
from dataclasses import dataclass, field
import itertools
import numpy as np
from ase import Atoms
from ase.data import covalent_radii, vdw_radii
from ase.data.colors import jmol_colors
from ase.neighborlist import neighbor_list


@dataclass
class AtomScene:
    positions: np.ndarray
    numbers: np.ndarray
    source_ids: np.ndarray
    radii: np.ndarray
    colors: np.ndarray
    bonds: np.ndarray
    cell: np.ndarray
    show_cell: bool
    labels: bool
    bond_radius: float
    polyhedra: list = field(default_factory=list)
    poly_color: tuple = (.25, .65, .9)
    poly_opacity: float = .45
    poly_shine: float = .5

    def bounds(self):
        low = (self.positions-self.radii[:, None]).min(axis=0)
        high = (self.positions+self.radii[:, None]).max(axis=0)
        for vertices, faces in self.polyhedra:
            low = np.minimum(low, vertices.min(axis=0)); high = np.maximum(high, vertices.max(axis=0))
        if self.show_cell:
            corners = np.array(list(itertools.product((0, 1), repeat=3))) @ self.cell
            low = np.minimum(low, corners.min(axis=0)); high = np.maximum(high, corners.max(axis=0))
        return low, high


def build_scene(atoms, *, size=40, show_cell=False, show_bonds=True, show_labels=False,
                selected=(), repeat=(1, 1, 1), boundary=True, bond_scale=1.05,
                bond_radius=.12, pair_rules=None, representation='Ball and stick', wrap_display=True,
                show_polyhedra=False, poly_centers='', poly_neighbors='O', poly_cutoff=2.5,
                poly_color=(.25, .65, .9), poly_opacity=.45, poly_shine=.5):
    positions, numbers, ids = [], [], []
    repeat = np.asarray(repeat, dtype=int)
    if repeat.shape != (3,) or (repeat < 1).any() or (repeat > 4).any():
        raise ValueError('Display repetitions must be between 1 and 4 on each axis.')
    periodic = bool(atoms.pbc.any()) and atoms.cell.rank == 3
    if not periodic and (repeat != 1).any():
        raise ValueError('Display repetitions require a periodic 3D cell.')
    if len(atoms)*np.prod(repeat) > 20000:
        raise ValueError('Display repetitions would exceed 20,000 atoms.')
    if periodic:
        fractional = atoms.get_scaled_positions(wrap=wrap_display)
        # Wrap only periodic axes; keep molecule/slab coordinates on open axes.
        for source, (frac, number) in enumerate(zip(fractional, atoms.numbers)):
            choices = [range(repeat[k]+int(wrap_display and boundary and atoms.pbc[k] and abs(frac[k]) < 1e-7))
                       for k in range(3)]
            for shift in itertools.product(*choices):
                positions.append((frac + shift) @ atoms.cell.array)
                numbers.append(number); ids.append(source)
    else:
        positions = atoms.positions.copy(); numbers = atoms.numbers.copy(); ids = np.arange(len(atoms))
    positions, numbers, ids = np.asarray(positions), np.asarray(numbers, int), np.asarray(ids, int)
    if len(positions) > 20000: raise ValueError('Boundary copies exceed the 20,000 displayed atom limit.')
    colors = jmol_colors[numbers].copy()
    colors[np.isin(ids, list(selected))] = [1, .55, .05]
    radius = covalent_radii[numbers]*size/100
    if representation == 'Space filling':
        vdw = vdw_radii[numbers]
        radius = np.where(np.isfinite(vdw), vdw, covalent_radii[numbers]*1.5)*size/50
    elif representation == 'Sticks':
        radius = np.full(len(numbers), bond_radius*1.15)
    bonds = np.empty((0, 2), dtype=int)
    if show_bonds and representation != 'Space filling':
        display = Atoms(numbers=numbers, positions=positions)
        rules = pair_rules or {}
        search = max(2*covalent_radii[numbers].max()*bond_scale,
                     max((r[1] for r in rules.values()), default=0))
        i, j, distance = neighbor_list('ijd', display, search)
        keep = (i < j) & (distance > .1)
        symbols = display.get_chemical_symbols()
        cutoff = (covalent_radii[numbers[i]] + covalent_radii[numbers[j]])*bond_scale
        allowed = distance <= cutoff
        for k in np.flatnonzero(keep):
            rule = rules.get(tuple(sorted((symbols[i[k]], symbols[j[k]]))))
            if rule is not None: allowed[k] = rule[0] <= distance[k] <= rule[1]
        bonds = np.column_stack((i[keep & allowed], j[keep & allowed]))
        if len(bonds) > 60000: raise ValueError('Too many bonds. Reduce the cutoff or display repetitions.')
    polyhedra = []
    if show_polyhedra:
        from scipy.spatial import ConvexHull, QhullError
        from ase.data import atomic_numbers
        centers = set(poly_centers.replace(',', ' ').split())
        neighbors = set(poly_neighbors.replace(',', ' ').split())
        if not centers or not neighbors or (centers | neighbors) - atomic_numbers.keys():
            raise ValueError('Enter valid center and neighbor elements (for example Ti and O).')
        if not np.isfinite(poly_cutoff) or not 0 < poly_cutoff <= 10:
            raise ValueError('Polyhedron neighbor cutoff must be between 0 and 10 Å.')
        if not np.isfinite([*poly_color, poly_opacity, poly_shine]).all() or len(poly_color) != 3 or any(v < 0 or v > 1 for v in (*poly_color, poly_opacity, poly_shine)):
            raise ValueError('Polyhedron color, opacity and shininess must be between 0 and 1.')
        symbols = atoms.get_chemical_symbols()
        center_ids = [i for i, symbol in enumerate(symbols) if symbol in centers]
        if sum(symbols[i] in centers for i in ids) > 2000:
            raise ValueError('Limit polyhedra to 2,000 displayed centers. Reduce display repetitions or center elements.')
        # ASE supplies all periodic neighbor images, including across cell boundaries.
        i, j, offsets = neighbor_list('ijD', atoms, poly_cutoff)
        neighbor_mask = np.array([symbols[k] in neighbors for k in j], dtype=bool)
        hulls = {}
        for source in center_ids:
            mask = (i == source) & neighbor_mask
            vectors = np.unique(np.round(offsets[mask], 10), axis=0)
            vectors = vectors[np.linalg.norm(vectors, axis=1) > 1e-8]
            if len(vectors) < 4 or np.linalg.matrix_rank(vectors - vectors[0]) < 3:
                continue
            try:
                hull = ConvexHull(vectors)
            except QhullError:
                continue
            # Orient each triangle outward for consistent lighting.
            faces = hull.simplices.copy()
            for face, equation in zip(faces, hull.equations):
                a, b, c = vectors[face]
                if np.dot(np.cross(b-a, c-a), equation[:3]) < 0:
                    face[1], face[2] = face[2], face[1]
            hulls[source] = vectors, faces
        for position, source in zip(positions, ids):
            if source in hulls:
                vectors, faces = hulls[source]
                polyhedra.append((position + vectors, faces))
    return AtomScene(positions, numbers, ids, radius, colors, bonds,
                     repeat[:, None]*atoms.cell.array, show_cell, show_labels, bond_radius,
                     polyhedra, tuple(poly_color), poly_opacity, poly_shine)


def fit_camera(scene, direction=(1.5, -2.4, 1.2), up=(0, 0, 1)):
    low, high = scene.bounds()
    center = (low+high)/2
    radius = max(np.linalg.norm(high-low)/2, 1.)
    direction = np.asarray(direction, float); direction /= np.linalg.norm(direction)
    return dict(position=center+direction*radius*4, focal=center, up=np.asarray(up, float), scale=radius*1.12)


def suggest_octahedra(atoms):
    """Suggest an isolated six-neighbor shell, without inventing missing ligands."""
    symbols = np.asarray(atoms.get_chemical_symbols())
    elements = set(symbols)
    ligands = elements & {'O', 'F', 'Cl', 'Br', 'I', 'S', 'Se', 'N'}
    candidates = []
    i, j, distances, vectors = neighbor_list('ijdD', atoms, 10.)
    for ligand in sorted(ligands):
        for center in sorted(elements - {ligand}):
            sixth, seventh = [], []
            valid = True
            for source in np.flatnonzero(symbols == center):
                indices = np.flatnonzero((i == source) & (symbols[j] == ligand) & (distances > 1e-8))
                indices = indices[np.argsort(distances[indices])]
                if len(indices) < 6 or np.linalg.matrix_rank(vectors[indices[:6]] - vectors[indices[0]]) < 3:
                    valid = False; break
                sixth.append(distances[indices[5]])
                seventh.append(distances[indices[6]] if len(indices) > 6 else 10.)
            if valid and sixth:
                low, high = max(sixth), min(seventh)
                if high - low > .02:
                    candidates.append((low, center, ligand, min(low + .05, (low + high)/2)))
    if not candidates:
        raise ValueError('No isolated six-neighbor shell found within 10 Å. Choose center/neighbor elements and cutoff manually.')
    _, center, ligand, cutoff = min(candidates)
    return dict(poly_centers=center, poly_neighbors=ligand, poly_cutoff=float(np.ceil(cutoff*1000)/1000))


def camera_view(camera, elev, azim):
    e, a = np.radians([elev, azim])
    direction = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    result = dict(camera)
    result['position'] = camera['focal'] + direction*np.linalg.norm(camera['position']-camera['focal'])
    result['up'] = np.array([0, 1, 0]) if abs(elev) == 90 else np.array([0, 0, 1])
    return result
