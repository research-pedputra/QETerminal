"""Editable recipes calling the original library's classes and methods."""

def create(name, **kwargs): return dict(id=name, **kwargs)
def call(method, **kwargs): return dict(object='analysis', method=method, kwargs=kwargs)
def constructor(cls, **kwargs): return {'id': 'analysis', 'class': cls, 'kwargs': kwargs}

PRESETS = {
    'Band structure': [constructor('BandStructure'), call('get_bandstructure', data='@data:0', fermi_energy=0, kpoints=None), call('plot_band_structure')],
    'Band gap': [constructor('BandGapAnalyzer', band='@band:0', n_valence_bands=1), call('calculate'), call('print_results')],
    'DOS / PDOS': [constructor('PDOSPlotter', fermi_energy=0), call('plot', files={'DOS': ['C0', '-', '@file:0']})],
    'Shifted PDOS': [constructor('PDOSPlottershift'), call('plot', ax='@ax:', data_dict={'DOS': ['C0', '-', '@file:0']}, Ef=0)],
    'Total energy': [constructor('TotalEnergy', filepath='@file:0'), call('extract'), call('last_TotalEnergy')],
    'Orbital center': [constructor('OrbitalCenter', file_path='@file:0'), call('get_extreme_positions')],
    'XRD': [constructor('SimulatedXRD'), dict(object='analysis', method='add_structure', args=['@file:0']), call('plot')],
    'Atomic scattering': [constructor('AtomicScatteringPlot'), dict(object='analysis', method='add_atoms', args=['Fe']), call('plot')],
    'Charge transfer': [constructor('ChargeTransfer', system='@cube:0', substrate='@cube:1', adsorbate='@cube:2', atoms='@atoms:0', substrate_atoms='@atoms:1', adsorbate_atoms='@atoms:2'), call('plot'), call('summary')],
    'Effective mass': [dict(id='bands', **{'class': 'BandStructure'}), dict(object='bands', method='get_bandstructure', kwargs={'data': '@data:0', 'fermi_energy': 0, 'kpoints': None}), constructor('EffectiveMass', bandstructure='@ref:bands'), call('calculate_average_mass', band_indices=[1], plot=True)],
    'Projected bands (JSON)': [constructor('ProjectedBandPlotter', band='@projected:0'), call('print_projected_states'), call('plot', ax='@ax:', selections={'All': {'atoms': ['all']}})],
}

HELP = {
    'Band structure': 'Input 0: QE bands.x two-column .gnu data (k distance, energy in eV), band by band. Set Fermi energy in get_bandstructure. kpoints: [["Γ", 0], ["X", 1.2]].',
    'Band gap': 'Input 0: two-column .gnu bands. Set the number of occupied bands (n_valence_bands). Gaps are sampled along the supplied path, which may miss extrema elsewhere in the Brillouin zone. @band uses path distance as its first coordinate, not physical k vectors.',
    'DOS / PDOS': 'Input 0: energy and DOS columns with a # header, as expected by the supplied reader. Add files to the plot files mapping to overlay curves. Set fermi_energy in the constructor.',
    'Charge transfer': 'Inputs 0, 1, 2: combined system, substrate, adsorbate cube files, on the same grid and cell. Units and cell assumptions are those of your uploaded class.',
    'Projected bands (JSON)': 'The Qepy XML loader was not included. Input 0: JSON with kpoints, eigen [kpoint,band] relative to Ef, states [{istate:1,atype:"Fe",wfc:1,l:2}], weights [state,kpoint,band], spin_components. For spin=2 use eigen1/eigen2 and weights [spin,state,kpoint,band].',
    'Effective mass': 'Input 0: two-column bands. The uploaded class’s indexing and units apply. Check k units and fit range before interpreting masses.',
}
