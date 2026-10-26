"""Band plotting adapter using global occupied/empty extrema."""
import copy
import math


def prepare_band_analysis(recipe, files, library):
    recipe = copy.deepcopy(recipe); options = recipe.get('band_options')
    if not options: return recipe
    if options['auto_fermi']:
        ef, electrons = library.BandStructure.read_scf_info(files[options['scf']])
        options['electrons'] = electrons
        if ef is None or not math.isfinite(ef): raise ValueError('The SCF file has no readable Fermi energy. Choose another SCF file or set Fermi energy manually.')
        options['fermi'] = ef
    from band_edges import load_band_table
    recipe['_band_arrays'] = load_band_table(files[options['data']])
    for step in recipe['steps']:
        if step.get('method') == 'get_bandstructure':
            step['kwargs'].update(data=f'@data:{options["data"]}', fermi_energy=options['fermi'], kpoints=options['points'])
    return recipe


def decorate_bands(bands, options, library):
    import numpy as np
    import matplotlib.pyplot as plt
    from band_edges import find_band_edges
    summary = find_band_edges(bands.bands, options['fermi'], manual=None if options['auto_edges'] else (options['vbm'], options['cbm']))
    summary.update(fermi=options['fermi'], electrons=options.get('electrons', 'not required'), reference_source='SCF Fermi energy' if options['auto_fermi'] else 'user-declared reference (default: energies relative to Ef=0)')
    summary['vbm_k'] = [float(bands.k[i]) for i in summary['vbm_kindices']]
    summary['cbm_k'] = [float(bands.k[i]) for i in summary['cbm_kindices']]
    vbm, cbm = summary['vbm'], summary['cbm']
    for number in plt.get_fignums():
        figure = plt.figure(number)
        if getattr(figure, '_band_decorated', False): continue
        for ax in figure.axes:
            for line in ax.lines:
                if np.array_equal(np.asarray(line.get_xdata()), bands.k):
                    line.set_color(options['colors'][0]); line.set_linewidth(options['linewidth'])
            if options['highlight']:
                for index, color, label in [(vbm, options['colors'][1], 'VBM'), (cbm, options['colors'][2], 'CBM')]:
                    ax.plot(bands.k, bands.bands[index-1] - options['fermi'], color=color, linewidth=options['linewidth'], label=f'{label} (band {index})')
                if not summary['metallic']:
                    for name, color in [('vbm', options['colors'][1]), ('cbm', options['colors'][2])]:
                        ax.scatter(summary[name+'_k'], [summary[name+'_energy']-options['fermi']]*len(summary[name+'_k']), color=color, marker='o', zorder=5)
                ax.legend()
            ax.set_xlim(options['xlim'] or [float(min(bands.k)), float(max(bands.k))])
            if options['ylim'] is not None: ax.set_ylim(options['ylim'])
            else:
                low, high = float(np.min(bands.bands) - options['fermi']), float(np.max(bands.bands) - options['fermi'])
                margin = max((high-low)*.05, .1); ax.set_ylim(low-margin, high+margin)
        figure._band_result = summary; figure._band_decorated = True
    return summary
