"""GIF export through the selected native renderer."""
import os
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from atom_scene import build_scene


def save_backend_gif(path, frames, backend, *, style, fps=5, width=800, loop=True,
                     frame_numbers=None, progress=None):
    if not frames or not 1 <= fps <= 50 or not 200 <= width <= 1600:
        raise ValueError('Choose frames, 1–50 fps, and a width of 200–1600 pixels.')
    height = round(width*.75)
    if len(frames)*width*height > 150_000_000:
        raise ValueError('Reduce resolution or increase the frame step (150 megapixel total limit).')
    numbers = frame_numbers if frame_numbers is not None else list(range(1, len(frames)+1))
    if len(numbers) != len(frames): raise ValueError('Frame numbers must match the frame count.')
    # One union bound prevents zoom breathing, and ensures all trajectory positions fit.
    lower, upper = [], []
    for a in frames:
        low, high = build_scene(a, **style).bounds(); lower.append(low); upper.append(high)
    low, high = np.min(lower, axis=0), np.max(upper, axis=0)
    center = (low+high)/2; radius = max(np.linalg.norm(high-low)/2, 1.)
    camera = backend.get_camera()
    direction = camera['position']-camera['focal']; direction /= np.linalg.norm(direction)
    camera = dict(camera, position=center+direction*radius*4, focal=center, scale=radius*1.12)
    images, temporary = [], None
    try:
        for i, (atoms, number) in enumerate(zip(frames, numbers)):
            if progress and progress(i, len(frames)) is False: return False
            pixels = backend.image(build_scene(atoms, **style), camera, width, height)
            image = Image.fromarray(np.asarray(pixels).astype(np.uint8)).convert('RGB')
            ImageDraw.Draw(image).text((12, 12), f'Frame {number}', fill='#475569')
            images.append(image.convert('P', palette=Image.Palette.ADAPTIVE, colors=256)); image.close()
        if progress and progress(len(frames), len(frames)) is False: return False
        path = Path(path)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.gif', delete=False) as f:
            temporary = Path(f.name)
        kwargs = {'loop': 0} if loop else {}
        images[0].save(temporary, format='GIF', save_all=True, append_images=images[1:],
                       duration=round(1000/fps/10)*10, disposal=2, optimize=False, **kwargs)
        os.replace(temporary, path); temporary = None
        return True
    finally:
        for image in images: image.close()
        if temporary is not None: temporary.unlink(missing_ok=True)
