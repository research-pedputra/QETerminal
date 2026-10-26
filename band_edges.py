"""Validated band tables and extrema over the entire sampled k path."""
from pathlib import Path
import numpy as np


def load_band_table(path):
    groups = []; rows = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line:
            if rows: groups.append(rows); rows = []
            continue
        if line.startswith(('#', '!')): continue
        values = line.replace('D', 'E').replace('d', 'e').split()
        if len(values) != 2: raise ValueError('Band file must contain two columns: k-path distance and energy (eV), grouped by band.')
        rows.append([float(v) for v in values])
    if rows: groups.append(rows)
    if not groups: raise ValueError('No band data found.')
    if len(groups) == 1:
        data = np.asarray(groups[0]); starts = np.r_[0, np.where(np.diff(data[:, 0]) < 0)[0] + 1, len(data)]
        groups = [data[a:b] for a, b in zip(starts[:-1], starts[1:])]
    groups = [np.asarray(group, dtype=float) for group in groups]
    k = groups[0][:, 0]
    if len(k) < 2 or np.any(np.diff(k) < 0): raise ValueError('Each band needs an ordered k-path with at least two points.')
    for group in groups:
        if not np.isfinite(group).all(): raise ValueError('Band energies and coordinates must be finite.')
        if len(group) != len(k) or not np.allclose(group[:, 0], k, atol=1e-8, rtol=1e-8):
            raise ValueError('All bands must use the same k-path grid.')
    return k, np.asarray([group[:, 1] for group in groups])


def find_band_edges(energies, reference=0., n_valence=None, tolerance=1e-6, manual=None):
    """energies[band,k]; classify occupations using the declared reference/count."""
    energies = np.asarray(energies, dtype=float)
    if energies.ndim != 2 or not np.isfinite(energies).all(): raise ValueError('Invalid band energy matrix.')
    nband, nk = energies.shape
    crossing = False
    if manual is not None:
        v, c = manual
        if not (1 <= v < c <= nband): raise ValueError(f'Check the associated bands: 1 ≤ VBM < CBM ≤ {nband}.')
        occupied, empty = np.array([v-1]), np.array([c-1])
    elif n_valence is not None:
        if not 1 <= n_valence < nband: raise ValueError('Occupied-band count must leave both valence and conduction bands.')
        occupied, empty = np.arange(n_valence), np.arange(n_valence, nband)
    else:
        low, high = energies.min(axis=1), energies.max(axis=1)
        crossing = bool(np.any((low < reference-tolerance) & (high > reference+tolerance)))
        occupied = np.where(high <= reference+tolerance)[0]
        empty = np.where((low >= reference-tolerance) & (high > reference+tolerance))[0]
        if crossing:
            below = np.where(energies <= reference, energies, -np.inf)
            above = np.where(energies >= reference, energies, np.inf)
            vi = np.unravel_index(np.argmax(below), energies.shape); ci = np.unravel_index(np.argmin(above), energies.shape)
            return dict(gap=0., signed_gap=0., direct_gap=None, classification='metallic (Fermi crossing)', vbm=int(vi[0]+1), cbm=int(ci[0]+1), vbm_energy=float(energies[vi]), cbm_energy=float(energies[ci]), vbm_kindices=[int(vi[1])], cbm_kindices=[int(ci[1])], direct_kindices=[], metallic=True)
        if not len(occupied) or not len(empty):
            raise ValueError('Cannot identify occupied and empty bands around the energy reference. Plain .gnu files have no occupations: use energies relative to Ef=0, set the correct reference, or supply SCF Fermi energy.')
    valence = energies[occupied]; conduction = energies[empty]
    vbm = float(valence.max()); cbm = float(conduction.min()); signed = cbm-vbm
    vi = np.argwhere(np.isclose(valence, vbm, atol=tolerance, rtol=0)); ci = np.argwhere(np.isclose(conduction, cbm, atol=tolerance, rtol=0))
    vk = np.unique(vi[:, 1]); ck = np.unique(ci[:, 1]); common = np.intersect1d(vk, ck)
    vertical = conduction.min(axis=0)-valence.max(axis=0); direct = float(vertical.min())
    kind = 'metallic / overlapping' if signed < -tolerance else ('zero gap' if signed <= tolerance else ('direct' if len(common) else 'indirect'))
    return dict(gap=max(0., signed), signed_gap=signed, direct_gap=direct, classification=kind,
                vbm=int(occupied[vi[0, 0]]+1), cbm=int(empty[ci[0, 0]]+1), vbm_energy=vbm, cbm_energy=cbm,
                vbm_kindices=vk.tolist(), cbm_kindices=ck.tolist(), direct_kindices=np.where(np.isclose(vertical, direct, atol=tolerance, rtol=0))[0].tolist(), metallic=signed < -tolerance)
