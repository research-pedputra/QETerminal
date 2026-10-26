"""Shared, axis-free structure rendering for the viewer and GIF export."""
import itertools
import os
import tempfile
from pathlib import Path
import numpy as np
from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from ase.neighborlist import neighbor_list
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image


def cell_corners(atoms):
    return np.array(list(itertools.product((0, 1), repeat=3))) @ atoms.cell.array


def scene_limits(frames, show_cell=False):
    lower, upper = [], []
    for atoms in frames:
        points = np.vstack((atoms.positions, cell_corners(atoms))) if show_cell else atoms.positions
        padding = max(covalent_radii[atoms.numbers].max(), 1.0)
        lower.append(points.min(axis=0) - padding)
        upper.append(points.max(axis=0) + padding)
    low, high = np.min(lower, axis=0), np.max(upper, axis=0)
    center = (low + high) / 2
    half = max(float((high-low).max()) * .55, 1.0)
    return [(v-half, v+half) for v in center]


def render_scene(ax, atoms, *, size=40, show_cell=False, show_bonds=True,
                 show_labels=False, selected=(), limits=None, elev=25, azim=-60):
    ax.clear()
    ax.set_facecolor('white')
    ax.set_axis_off()
    ax.set_proj_type('ortho')
    positions, numbers = atoms.positions, atoms.numbers
    colors = jmol_colors[numbers]
    selected = set(selected)
    # Actual shaded sphere surfaces for ordinary structures; lightweight points for large ones.
    spheres = len(atoms) <= 500
    scatter = ax.scatter(*positions.T, s=(covalent_radii[numbers] * size) ** 2 / 3,
                         c=colors, alpha=0 if spheres else 1, depthshade=True,
                         edgecolors=['#ff8800' if i in selected else '#344054' for i in range(len(atoms))],
                         picker=5)
    if spheres:
        u = np.linspace(0, 2*np.pi, 18)
        v = np.linspace(0, np.pi, 12)
        unit = np.array([np.outer(np.cos(u), np.sin(v)),
                         np.outer(np.sin(u), np.sin(v)),
                         np.outer(np.ones_like(u), np.cos(v))])
        for i, (point, number, color) in enumerate(zip(positions, numbers, colors)):
            xyz = unit * covalent_radii[number] * size / 100 + point[:, None, None]
            ax.plot_surface(*xyz, color='#ff8800' if i in selected else color,
                            linewidth=0, antialiased=True, shade=True)
    if show_labels and len(atoms) <= 500:
        for i, p in enumerate(positions):
            ax.text(*p, f' {i+1}:{atoms[i].symbol}', fontsize=8)
    if show_cell:
        corners = np.array(list(itertools.product((0, 1), repeat=3)))
        points = corners @ atoms.cell.array
        edges = [[points[i], points[j]] for i in range(8) for j in range(i+1, 8)
                 if np.sum(corners[i] != corners[j]) == 1]
        ax.add_collection3d(Line3DCollection(edges, colors='#64748b', linewidths=1))
    if show_bonds and len(atoms) <= 3000:
        i, j, shifts = neighbor_list('ijS', atoms, 1.15 * covalent_radii[numbers])
        # Only connect displayed atoms; do not draw dangling bonds to invisible periodic images.
        keep = (i < j) & (shifts == 0).all(axis=1)
        segments = np.stack((positions[i[keep]], positions[j[keep]]), axis=1)
        ax.add_collection3d(Line3DCollection(segments[:30000], colors='#939ca9', linewidths=3))
    if limits is None:
        limits = scene_limits([atoms], show_cell)
    ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2])
    ax.set_box_aspect((1, 1, 1), zoom=1.25)
    ax.view_init(elev=elev, azim=azim)
    return scatter


def save_trajectory_gif(path, frames, *, fps=5, width=800, loop=True,
                        frame_numbers=None, progress=None, **style):
    """Render imported coordinates with fixed camera/bounds. Return False on cancellation.

    Write atomically, so cancellation or failure never damages an existing GIF.
    The progress callback returns False to cancel and is called before each frame.
    """
    if not frames or not 1 <= fps <= 50 or not 200 <= width <= 1600:
        raise ValueError('Choose frames, 1–50 fps, and an image width of 200–1600 pixels.')
    height = round(width * .75)
    if len(frames) * width * height > 150_000_000:
        raise ValueError('Animation is too large. Reduce the resolution or increase the frame step (150 megapixel total limit).')
    numbers = list(frame_numbers) if frame_numbers is not None else list(range(1, len(frames)+1))
    if len(numbers) != len(frames): raise ValueError('Frame numbers must match the frame count.')
    fig = Figure(figsize=(width/100, height/100), dpi=100, facecolor='white')
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    limits = scene_limits(frames, style.get('show_cell', False))
    images = []
    temporary = None
    try:
        for index, (atoms, number) in enumerate(zip(frames, numbers)):
            if progress and progress(index, len(frames)) is False: return False
            render_scene(ax, atoms, limits=limits, **style)
            # Frame number also keeps identical geometries as separate timed GIF frames.
            ax.text2D(.025, .96, f'Frame {number}', transform=ax.transAxes, color='#475569', fontsize=10)
            canvas.draw()
            image = Image.fromarray(np.asarray(canvas.buffer_rgba())[:, :, :3].copy())
            images.append(image.convert('P', palette=Image.Palette.ADAPTIVE, colors=256))
            image.close()
        if progress and progress(len(frames), len(frames)) is False: return False
        path = Path(path)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.gif', delete=False) as f:
            temporary = Path(f.name)
        options = {'loop': 0} if loop else {}
        images[0].save(temporary, format='GIF', save_all=True, append_images=images[1:],
                       duration=round(1000/fps/10)*10, disposal=2, optimize=False, **options)
        os.replace(temporary, path)
        temporary = None
        return True
    finally:
        for image in images: image.close()
        fig.clear()
        if temporary is not None: temporary.unlink(missing_ok=True)
