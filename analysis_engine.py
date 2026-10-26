"""Run the supplied analysis library in an isolated plotting process."""
import ast
import contextlib
import inspect
import io
import json
import pickle
import sys
from pathlib import Path


class ProjectedData:
    """Portable JSON equivalent of the attributes used by ProjectedBandPlotter."""
    def __init__(self, data):
        import numpy as np
        self.spin_components = int(data.get('spin_components', 1))
        self.kpoints = np.asarray(data['kpoints'], dtype=float)
        self.nkpoints = len(self.kpoints); self.states = data['states']
        names = ('eigen1', 'eigen2') if self.spin_components == 2 else ('eigen',)
        for name in names: setattr(self, name, np.asarray(data[name], dtype=float))
        eigen = getattr(self, names[0]); self.nbands = eigen.shape[1]
        if eigen.shape[0] != self.nkpoints: raise ValueError('Eigenvalues must have shape [kpoint, band].')
        self.weights = np.asarray(data['weights'], dtype=float)
        shape = (len(self.states), self.nkpoints, self.nbands)
        if self.weights.shape != ((2,) + shape if self.spin_components == 2 else shape):
            raise ValueError('Weights need [state, kpoint, band], or [spin, state, kpoint, band] for collinear spin.')

    def get_weights(self, selected_orbitals):
        if self.spin_components == 2:
            return tuple(self.weights[spin, selected_orbitals].sum(axis=0) for spin in range(2))
        return self.weights[selected_orbitals].sum(axis=0)


def catalog():
    tree = ast.parse(Path(__file__).with_name('myqetools.py').read_text(encoding='utf-8'))
    result = {}
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef): continue
        methods = {}
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef) or (fn.name.startswith('_') and fn.name != '__init__'): continue
            params = {}
            args = fn.args.args
            defaults = [None] * (len(args) - len(fn.args.defaults)) + fn.args.defaults
            for arg, default in zip(args, defaults):
                if arg.arg in ('self', 'cls'): continue
                try: value = ast.literal_eval(default) if default is not None else None
                except (ValueError, TypeError): value = None
                params[arg.arg] = value
            methods[fn.name] = {'params': params, 'signature': ast.unparse(fn.args), 'doc': ast.get_docstring(fn) or '',
                                'required': [arg.arg for arg, default in zip(args, defaults) if default is None and arg.arg not in ('self', 'cls')],
                                'varargs': fn.args.vararg.arg if fn.args.vararg else None}
        result[cls.name] = methods
    return result


def run_recipe(recipe, files):
    import numpy as np
    import matplotlib.pyplot as plt
    import myqetools as library
    from ase.io import read
    from ase.io.cube import read_cube
    from types import SimpleNamespace
    objects = {}
    def resolve(value):
        if isinstance(value, list): return [resolve(v) for v in value]
        if isinstance(value, dict):
            if '$array' in value: return np.asarray(resolve(value['$array']))
            if '$object' in value: return SimpleNamespace(**{k: resolve(v) for k, v in value['$object'].items()})
            return {k: resolve(v) for k, v in value.items()}
        if not isinstance(value, str) or not value.startswith('@'): return value
        kind, _, key = value[1:].partition(':')
        if kind == 'ref':
            parts = key.split('.'); obj = objects[parts.pop(0)]
            for part in parts:
                obj = obj[int(part)] if part.isdigit() else (obj[part] if isinstance(obj, dict) else getattr(obj, part))
            return obj
        if kind == 'ax': return plt.gca()
        if kind == 'fig': return plt.gcf()
        if kind == 'files': return list(files)
        path = files[int(key)]
        if kind == 'file': return path
        if kind == 'data': return np.loadtxt(path)
        if kind == 'atoms':
            from structure_io import load_structure
            return load_structure(path)[0][-1]
        if kind == 'cube':
            with open(path) as handle: return read_cube(handle)['data']
        if kind == 'json': return resolve(json.loads(Path(path).read_text(encoding='utf-8')))
        if kind == 'projected': return ProjectedData(json.loads(Path(path).read_text(encoding='utf-8')))
        if kind == 'band':
            data = np.loadtxt(path); k = np.unique(data[:, 0])
            eigen = data[:, 1].reshape((-1, len(k))).T - float(recipe.get('fermi', 0))
            return SimpleNamespace(kpoints=np.column_stack([k, np.zeros((len(k), 2))]), nkpoints=len(k), spin_components=1, eigen=eigen)
        raise ValueError('Unknown input reference: ' + value)
    output = io.StringIO()
    original_show = plt.show
    plt.show = lambda *a, **kw: None
    try:
        with contextlib.redirect_stdout(output), plt.rc_context(recipe.get('rc', {})):
            if recipe.get('pdos_options'):
                from pdos_data import render_pdos
                figure = render_pdos(recipe, files)
                return [figure], output.getvalue()
            from band_analysis import prepare_band_analysis, decorate_bands
            recipe = prepare_band_analysis(recipe, files, library)
            for step in recipe['steps']:
                args = resolve(step.get('args', [])); kwargs = resolve(step.get('kwargs', {}))
                if 'class' in step:
                    cls = getattr(library, step['class'])
                    if step['class'] == 'ChargeTransfer':
                        fields = [kwargs.get(k) for k in ('system', 'substrate', 'adsorbate')]
                        atoms = [kwargs.get(k) for k in ('atoms', 'substrate_atoms', 'adsorbate_atoms')]
                        if any(f is None for f in fields) or len({f.shape for f in fields}) != 1:
                            raise ValueError('Charge-transfer density grids must have identical shapes.')
                        if any(a is None for a in atoms) or not all(np.allclose(atoms[0].cell, a.cell) for a in atoms[1:]):
                            raise ValueError('Charge-transfer cells must match.')
                    obj = cls(*args, **kwargs)
                else:
                    owner = objects[step.get('object', 'analysis')]
                    method = step['method']
                    if method.startswith('save') and recipe.get('band_options'):
                        decorate_bands(objects['analysis'], recipe['band_options'], library)
                    if method.startswith('_'): raise ValueError('Choose a public method.')
                    # A few supplied utility functions omit self and @staticmethod.
                    raw = inspect.getattr_static(type(owner), method)
                    parameters = list(inspect.signature(raw).parameters) if inspect.isfunction(raw) else ['self']
                    fn = raw if parameters and parameters[0] != 'self' else getattr(owner, method)
                    if type(owner).__name__ in ('PDOSPlotter', 'PDOSPlottershift') and method in ('read_pdos', 'load_pdos'):
                        from pdos_data import read_pdos
                        path = (args[0] if args else next(iter(kwargs.values())))
                        data = read_pdos(path)
                        if method == 'read_pdos': obj = (data.energy-owner.fermi, data.density)
                        else: obj = (data.energy, data.density[:, 0], -data.density[:, 1] if data.density.shape[1] == 2 else np.zeros_like(data.energy))
                    elif method == 'get_bandstructure' and recipe.get('_band_arrays') is not None:
                        owner.k, owner.bands = recipe['_band_arrays']
                        owner.bands = owner.bands.copy(); owner.data = args[0] if args else kwargs['data']
                        owner.fermi_energy = kwargs['fermi_energy']; owner.high_symmetry_points = kwargs.get('kpoints'); obj = None
                    else: obj = fn(*args, **kwargs)
                    if obj is not None: print(method + ':', obj)
                if step.get('id'): objects[step['id']] = obj
            if recipe.get('band_options'):
                result = decorate_bands(objects['analysis'], recipe['band_options'], library)
                print(f'Fermi energy = {result["fermi"]:g} eV; electrons = {result["electrons"]}')
                print(f'VBM band = {result["vbm"]}; CBM band = {result["cbm"]}; CBM − VBM gap = {result["gap"]:.6g} eV')
                print(f'{result["classification"]}; VBM energy = {result["vbm_energy"]:g} eV at k={result["vbm_k"]}; CBM energy = {result["cbm_energy"]:g} eV at k={result["cbm_k"]}')
                print(f'Minimum direct gap = {result["direct_gap"]} eV. Reference: {result["reference_source"]}. Sampled k path only.')
            figures = [plt.figure(n) for n in plt.get_fignums()]
            return figures, output.getvalue()
    finally:
        plt.show = original_show


def main():
    import matplotlib
    matplotlib.use('Agg')
    payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    try:
        figures, output = run_recipe(payload['recipe'], payload['files'])
        with open(sys.argv[2], 'wb') as stream: pickle.dump((figures, output), stream)
    except Exception:
        import traceback
        traceback.print_exc(); sys.exit(1)


if __name__ == '__main__': main()
