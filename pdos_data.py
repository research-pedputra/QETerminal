"""Direct QE PDOS grouping and numerical energy integration using NumPy."""
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np

ATOM = re.compile(r'^(?P<prefix>.*?)\.?pdos_atm#(?P<atom>\d+)\((?P<element>[^)]+)\)_wfc#(?P<wfc>\d+)\((?P<orbital>[spdfg])(?P<soc>[^)]*)\)', re.I)
COMPONENTS = {'s': ['s'], 'p': ['pz', 'px', 'py'], 'd': ['dz2', 'dzx', 'dzy', 'dx2-y2', 'dxy'], 'f': [f'f(m={m})' for m in range(1, 8)], 'g': [f'g(m={m})' for m in range(1, 10)]}
MODES = ['Element + orbital', 'Atom + orbital', 'Element', 'Atom', 'Orbital', 'Element + component', 'Atom + component', 'Each file', 'All matched files']


def identify(path):
    match = ATOM.search(Path(path).name)
    if not match: return None
    result = match.groupdict(); result['atom'] = int(result['atom']); result['wfc'] = int(result['wfc']); result['orbital'] = result['orbital'].lower()
    return result


def natural_key(text): return [int(p) if p.isdigit() else p.casefold() for p in re.split(r'(\d+)', str(text))]


def group_inputs(inputs, mode='Element + orbital', element='All', atom='All', orbital='All', word1='', word2=''):
    groups = {}; calculations = {(str(Path(i['path']).parent), identify(i['path'])['prefix']) for i in inputs if identify(i['path'])}
    for index, item in sorted(enumerate(inputs), key=lambda pair: natural_key(pair[1]['path'])):
        path = item['path']; name = Path(path).name; meta = identify(path)
        if not meta and Path(path).suffix.lower() in ('.out', '.in', '.gnu', '.cube', '.pwo', '.pwi'): continue
        start = name.find(word1)
        if start < 0 or name.find(word2, start+len(word1)) < 0: continue
        if element != 'All' and (not meta or meta['element'] != element): continue
        if atom != 'All' and (not meta or str(meta['atom']) != str(atom)): continue
        if orbital != 'All' and (not meta or meta['orbital'] != orbital): continue
        components = [(None, '')]
        if meta and 'component' in mode:
            if meta['soc']:
                raise ValueError('Named real-orbital components do not apply to spin-orbit j projections. Choose atom/orbital totals.')
            components = list(enumerate(COMPONENTS[meta['orbital']]))
        for component, component_label in components:
            if mode == 'All matched files': label = 'Selected PDOS sum'
            elif not meta or mode == 'Each file': label = name
            else:
                labels = {'Element + orbital': f'{meta["element"]} {meta["orbital"]}', 'Atom + orbital': f'{meta["element"]} #{meta["atom"]} {meta["orbital"]}',
                          'Element': meta['element'], 'Atom': f'{meta["element"]} #{meta["atom"]}', 'Orbital': meta['orbital'],
                          'Element + component': f'{meta["element"]} {component_label}', 'Atom + component': f'{meta["element"]} #{meta["atom"]} {component_label}'}
                label = labels[mode]
                if len(calculations) > 1: label = str(Path(path).parent) + '/' + meta['prefix'] + ' / ' + label
            key = (label, component)
            group = groups.setdefault(key, {'label': label, 'indices': [], 'component': component})
            group['indices'].append(index)
    return list(groups.values())


@dataclass
class Pdos:
    energy: np.ndarray
    density: np.ndarray
    components: np.ndarray
    path: str
    total: bool


def read_pdos(path, layout='Auto', total_kind='Projected DOS'):
    text = Path(path).read_text(encoding='utf-8-sig'); headers = []; rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        if line.startswith(('#', '!')): headers.append(line.lower()); continue
        try: rows.append([float(v.replace('D','E').replace('d','e')) for v in line.split()])
        except ValueError: raise ValueError(f'{Path(path).name}: non-numeric PDOS row.')
    header = ' '.join(headers)
    if re.search(r'#\s*ik\b', header): raise ValueError('k-resolved PDOS needs k-point weights; ordinary PDOS summation cannot combine it.')
    try: data = np.asarray(rows, dtype=float)
    except ValueError: raise ValueError(f'{Path(path).name}: inconsistent PDOS column counts.')
    if data.ndim != 2 or data.shape[1] < 2 or len(data) < 2 or not np.isfinite(data).all(): raise ValueError(f'{Path(path).name}: invalid PDOS table.')
    data = data[np.argsort(data[:, 0], kind='stable')]
    if np.any(np.diff(data[:, 0]) <= 0): raise ValueError('PDOS energy grid contains repeated points; check for k-resolved data.')
    meta = identify(path); total = 'pdos_tot' in Path(path).name.lower()
    if layout == 'Spin up/down': spin = 2
    elif layout == 'Non-spin': spin = 1
    elif re.search(r'(?:ldos|pdos|dos)up\b', header): spin = 2
    elif re.search(r'\b(?:ldos|pdos|dos)\s*(?:\(|\b)', header) or data.shape[1] == 2: spin = 1
    else: raise ValueError(f'{Path(path).name}: ambiguous spin columns. Choose Non-spin or Spin up/down explicitly.')
    start = 1
    if total:
        if total_kind == 'Projected DOS':
            # Noncollinear without SOC: E DOS PDOSup PDOSdw.
            start = 2 if spin == 2 and data.shape[1] == 4 else 1+spin
        elif spin == 2 and data.shape[1] == 4: spin = 1
    if data.shape[1] < start+spin: raise ValueError(f'{Path(path).name}: missing density columns for the chosen format.')
    density = data[:, start:start+spin]
    if meta:
        remaining = data[:, 1+spin:]
        if remaining.shape[1] % spin: raise ValueError('PDOS component columns do not match the spin format.')
        components = remaining.reshape((len(data), -1, spin))
        if not meta['soc'] and components.shape[1] != len(COMPONENTS[meta['orbital']]):
            raise ValueError(f'{Path(path).name}: expected {len(COMPONENTS[meta["orbital"]])} orbital components; check the header/spin setting.')
    else: components = np.empty((len(data), 0, spin))
    return Pdos(data[:, 0], density, components, str(path), total)


def sum_group(paths, component=None, layout='Auto', total_kind='Projected DOS'):
    if len({str(Path(p).resolve()) for p in paths}) != len(paths): raise ValueError('A PDOS input was selected twice.')
    sources = [read_pdos(p, layout, total_kind) for p in paths]
    if not sources: raise ValueError('No matching PDOS files.')
    if len(sources) > 1 and any(s.total for s in sources): raise ValueError('Do not sum pdos_tot with atomic files or another total; this would double-count DOS.')
    prefixes = {identify(p)['prefix'] for p in paths if identify(p)}
    if len(prefixes) > 1: raise ValueError('Select one calculation prefix when summing all matched files.')
    first = sources[0]; values = np.zeros_like(first.density)
    for source in sources:
        if source.energy.shape != first.energy.shape or not np.allclose(source.energy, first.energy, rtol=1e-9, atol=1e-8):
            raise ValueError('PDOS energy grids differ; no implicit interpolation or index-wise summation was performed.')
        if source.density.shape != values.shape: raise ValueError('Cannot sum different spin layouts.')
        if component is not None:
            if component >= source.components.shape[1]: raise ValueError('The selected orbital component is absent.')
            values += source.components[:, component, :]
        else: values += source.density  # LDOS already contains its orbital components.
    return np.column_stack((first.energy, values))


def integrate_pdos(table, lower=None, upper=None):
    """Integrate raw spin densities in states/eV; interpolate range endpoints."""
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 2 or not np.isfinite(table).all():
        raise ValueError('Integration requires a finite PDOS table with at least two energies.')
    energy = table[:, 0]
    if np.any(np.diff(energy) <= 0): raise ValueError('Integration energies must be strictly increasing.')
    lower = energy[0] if lower is None else float(lower)
    upper = energy[-1] if upper is None else float(upper)
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError('Integration minimum must be less than its maximum.')
    if lower < energy[0] or upper > energy[-1]:
        raise ValueError('Integration range must lie within the available PDOS energies.')
    grid = np.concatenate(([lower], energy[(energy > lower) & (energy < upper)], [upper]))
    density = np.column_stack([np.interp(grid, energy, column) for column in table[:, 1:].T])
    counts = np.sum(np.diff(grid)[:, None] * (density[:-1] + density[1:]) / 2, axis=0)
    return dict(lower=float(lower), upper=float(upper), states=counts.tolist())


def render_pdos(recipe, files):
    import matplotlib.pyplot as plt
    import myqetools
    from tempfile import TemporaryDirectory
    options = recipe['pdos_options']; groups = options['groups']
    if not groups: raise ValueError('Select at least one matching PDOS group.')
    tables = []; provenance = []
    for group in groups:
        calculations = {(str(Path(options['source_names'][i]).parent), identify(options['source_names'][i])['prefix']) for i in group['indices'] if identify(options['source_names'][i])}
        if len(calculations) > 1: raise ValueError('A summed group spans multiple calculation folders/prefixes. Choose separate groups or select one calculation.')
        sources = [files[i] for i in group['indices']]
        tables.append(sum_group(sources, group.get('component'), options['layout'], options['total_kind']))
        provenance.append([options['source_names'][i] for i in group['indices']])
    steps = recipe['steps']; init = next(s for s in steps if 'class' in s)
    plot = next(s for s in reversed(steps) if s.get('method') == 'plot')
    kwargs = dict(plot.get('kwargs', {})); kwargs.pop('files', None); kwargs.pop('data_dict', None); kwargs.pop('ax', None)
    with TemporaryDirectory(prefix='summed-pdos-') as directory:
        mapping = {}
        for i, (group, table) in enumerate(zip(groups, tables)):
            path = Path(directory)/f'curve-{i}.dat'
            np.savetxt(path, table, header='E(eV) ' + ('pdos(E)' if table.shape[1] == 2 else 'pdosup(E) pdosdw(E)'))
            mapping[group['label']] = [group['color'], group['style'], str(path)]
        if init['class'] == 'PDOSPlottershift':
            obj = myqetools.PDOSPlottershift(); fig, ax = plt.subplots(); obj.plot(ax=ax, data_dict=mapping, **kwargs)
        else:
            obj = myqetools.PDOSPlotter(**init.get('kwargs', {}))
            # Fix orientation and labels while retaining the uploaded class options.
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8,6)))
            orientation = kwargs.get('orientation') or obj.orientation
            for group, table in zip(groups, tables):
                energy = table[:, 0]-obj.fermi
                for spin in range(table.shape[1]-1):
                    density = table[:, spin+1] * (-1 if spin else 1)
                    label = group['label'] + ((' ↑' if spin == 0 else ' ↓') if table.shape[1] == 3 else '')
                    x, y = (energy, density) if orientation == 'vertical' else (density, energy)
                    if group['style'] == 'fill':
                        (ax.fill_between if orientation == 'vertical' else ax.fill_betweenx)(energy, density, color=group['color'], alpha=.5, label=label)
                    else: ax.plot(x, y, color=group['color'], linestyle=group['style'], label=label)
            if orientation == 'vertical': ax.set_xlabel('E − Ef (eV)'); ax.set_ylabel('PDOS (states/eV)'); ax.axvline(0, color='gray', ls='--')
            else: ax.set_ylabel('E − Ef (eV)'); ax.set_xlabel('PDOS (states/eV)'); ax.axhline(0, color='gray', ls='--')
            if kwargs.get('xlim'): (ax.set_xlim if orientation == 'vertical' else ax.set_ylim)(kwargs['xlim'])
            if kwargs.get('ylim'): (ax.set_ylim if orientation == 'vertical' else ax.set_xlim)(kwargs['ylim'])
            ax.legend(); fig.tight_layout()
    bounds = options.get('integration', {})
    integrals = [integrate_pdos(table, bounds.get('lower'), bounds.get('upper')) for table in tables]
    fig._pdos_tables = [{'label': group['label'], 'data': table, 'sources': names, 'integral': integral} for group, table, names, integral in zip(groups, tables, provenance, integrals)]
    print('PDOS groups (direct NumPy sums; trapezoidal energy integration):')
    for group, names, integral in zip(groups, provenance, integrals):
        channels = ', '.join(f'{value:.8g}' for value in integral['states'])
        print(f'{group["label"]}: {len(names)} file(s)\n  ' + '\n  '.join(names))
        print(f'  Integrated states [{integral["lower"]:.8g}, {integral["upper"]:.8g}] eV (raw energies): {channels}' + (' (up, down)' if len(integral['states']) == 2 else ''))
    return fig


def export_groups(tables, folder):
    paths = []
    for group in tables:
        name = 'pdos_' + re.sub(r'[^\w().+-]+', '_', group['label'])[:100].strip('.')
        path = Path(folder)/(name+'.dat'); number = 1
        while True:
            try:
                with path.open('x', encoding='utf-8') as output:
                    data = group['data']; columns = 'E(eV) ' + ('pdos(E)' if data.shape[1] == 2 else 'pdosup(E) pdosdw(E)')
                    integral = group.get('integral')
                    info = '' if integral is None else f'\nTrapezoidal integral over raw energies [{integral["lower"]}, {integral["upper"]}] eV; states by spin channel: {integral["states"]}'
                    np.savetxt(output, data, header=columns + info + '\nSummed sources (unshifted energies):\n' + '\n'.join(group['sources']))
                break
            except FileExistsError:
                number += 1; path = Path(folder)/f'{name}_{number}.dat'
        paths.append(str(path))
    return paths
