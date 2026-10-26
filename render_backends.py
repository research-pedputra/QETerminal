"""Lazy-loaded, interchangeable native 3D renderers for the PySide6 app."""
import os
os.environ['QT_API'] = 'pyside6'
import itertools
import numpy as np
from PySide6 import QtWidgets as W
from atom_scene import fit_camera, camera_view


class PyVistaBackend:
    name = 'PyVista / VTK'

    def __init__(self, parent, picked):
        import pyvista as pv
        from pyvistaqt import QtInteractor
        self.pv = pv
        self.widget = QtInteractor(parent, auto_update=False)
        self.plotter = self.widget
        self.plotter.set_background('white')
        self.plotter.enable_parallel_projection()
        self.plotter.enable_anti_aliasing('ssaa')
        self.scene = None
        self.picked = picked
        self.plotter.enable_surface_point_picking(callback=self._pick, show_message=False,
                                                  show_point=False, left_clicking=True, picker='cell')

    def _pick(self, point):
        if self.scene is not None:
            distances = np.linalg.norm(self.scene.positions-point, axis=1)-self.scene.radii
            index = int(np.argmin(distances))
            if distances[index] < .1: self.picked(int(self.scene.source_ids[index]))

    def render(self, scene, fit=False):
        camera = self.get_camera() if self.scene is not None else fit_camera(scene)
        self.scene = scene
        self.plotter.clear()
        self.plotter.enable_lightkit()
        self.populate(self.plotter, scene)
        if fit:
            camera = fit_camera(scene, camera['position']-camera['focal'], camera['up'])
        self.set_camera(camera)

    def populate(self, plotter, scene):
        pv = self.pv
        if scene.polyhedra:
            plotter.enable_depth_peeling(number_of_peels=8)
            for index, (vertices, faces) in enumerate(scene.polyhedra):
                mesh = pv.PolyData(vertices, np.column_stack((np.full(len(faces), 3), faces)).ravel())
                plotter.add_mesh(mesh, color=scene.poly_color, opacity=scene.poly_opacity,
                                 smooth_shading=False, ambient=.18, diffuse=.75,
                                 specular=scene.poly_shine, specular_power=5 + 95*scene.poly_shine,
                                 pickable=False, name=f'polyhedron-{index}')
        # Sphere glyphs are merged into one mesh; avoid thousands of separate actors.
        points = pv.PolyData(scene.positions)
        points['radius'] = scene.radii
        points['color'] = (scene.colors*255).astype(np.uint8)
        sphere = pv.Sphere(theta_resolution=32, phi_resolution=24)
        # pv.Sphere's default radius is .5; create unit geometry for ASE radii.
        sphere.points *= 2
        glyphs = points.glyph(scale='radius', orient=False, geom=sphere)
        plotter.add_mesh(glyphs, scalars='color', rgb=True, smooth_shading=True,
                         ambient=.22, diffuse=.72, specular=.45, specular_power=35,
                         show_scalar_bar=False, name='atoms')
        if len(scene.bonds):
            starts = scene.positions[scene.bonds[:, 0]]; ends = scene.positions[scene.bonds[:, 1]]
            middle = (starts+ends)/2
            # Each half-cylinder is colored by its own atom.
            segments = np.concatenate((np.stack((starts, middle), axis=1), np.stack((middle, ends), axis=1)))
            pts = segments.reshape(-1, 3)
            lines = np.column_stack((np.full(len(segments), 2), np.arange(0, len(pts), 2), np.arange(1, len(pts), 2)))
            mesh = pv.PolyData(pts, lines=lines.ravel())
            mesh.cell_data['color'] = (np.concatenate((scene.colors[scene.bonds[:, 0]], scene.colors[scene.bonds[:, 1]]))*255).astype(np.uint8)
            tubes = mesh.tube(radius=scene.bond_radius, n_sides=16, capping=True)
            plotter.add_mesh(tubes, scalars='color', rgb=True, smooth_shading=True,
                             ambient=.2, specular=.3, show_scalar_bar=False, pickable=False, name='bonds')
        if scene.show_cell:
            corners = np.array(list(itertools.product((0, 1), repeat=3)))
            pts = corners @ scene.cell
            edges = [v for i in range(8) for j in range(i+1, 8) if np.sum(corners[i] != corners[j]) == 1 for v in (pts[i], pts[j])]
            plotter.add_lines(np.array(edges), color='#64748b', width=1, connected=False)
        if scene.labels and len(scene.positions) <= 500:
            from ase.data import chemical_symbols
            labels = [f'{chemical_symbols[n]}{i+1}' for n, i in zip(scene.numbers, scene.source_ids)]
            plotter.add_point_labels(scene.positions, labels, font_size=12, point_size=0,
                                     shape=None, always_visible=True)

    def get_camera(self):
        c = self.plotter.camera
        return dict(position=np.array(c.position), focal=np.array(c.focal_point), up=np.array(c.up), scale=c.parallel_scale)

    def set_camera(self, camera):
        c = self.plotter.camera
        c.position = camera['position']; c.focal_point = camera['focal']; c.up = camera['up']
        c.parallel_projection = True; c.parallel_scale = camera['scale']
        self.plotter.reset_camera_clipping_range(); self.plotter.render()

    def image(self, scene, camera, width, height, transparent=False):
        # Separate render window: exporting does not resize or disturb the live viewport.
        p = self.pv.Plotter(off_screen=True, window_size=(width, height))
        try:
            p.set_background('white'); p.enable_anti_aliasing('ssaa')
            self.populate(p, scene)
            p.camera_position = (camera['position'], camera['focal'], camera['up'])
            p.enable_parallel_projection(); p.camera.parallel_scale = camera['scale']
            p.reset_camera_clipping_range()
            return p.screenshot(return_img=True, transparent_background=transparent)
        finally:
            p.close()

    def close(self):
        self.widget.close()


class OvitoBackend:
    name = 'OVITO'

    def __init__(self, parent, picked):
        from ovito.gui import create_qwidget
        from ovito.vis import Viewport
        from ovito.pipeline import Pipeline, StaticSource
        from ovito.data import DataCollection
        self.pipeline = Pipeline(source=StaticSource(data=DataCollection()))
        self.pipeline.add_to_scene()
        self.viewport = Viewport(type=Viewport.Type.Ortho, camera_dir=(-1, 1, -.7))
        self.window = create_qwidget(self.viewport, parent=parent, show_orientation_indicator=False)
        self.widget = getattr(self.window, 'widget', self.window)
        self.scene = None

    def data(self, scene):
        from ovito.data import DataCollection, Particles, Bonds, SimulationCell
        from ovito.vis import BondsVis
        data = DataCollection()
        particles = Particles(count=len(scene.positions))
        particles.create_property('Position', data=scene.positions)
        particles.create_property('Color', data=scene.colors)
        particles.create_property('Radius', data=scene.radii)
        particles.create_property('Particle Identifier', data=scene.source_ids+1)
        data.objects.append(particles)
        if len(scene.bonds):
            bonds = Bonds()
            bonds.create_property('Topology', data=scene.bonds)
            bonds.vis.width = scene.bond_radius*2
            bonds.vis.coloring_mode = BondsVis.ColoringMode.ByParticle
            bonds.vis.flat_shading = False
            particles.bonds = bonds
        cell = SimulationCell(pbc=(False, False, False))
        cell[...] = np.column_stack((scene.cell.T, np.zeros(3)))
        cell.vis.enabled = scene.show_cell
        data.objects.append(cell)
        if scene.polyhedra:
            from ovito.data import TriangleMesh
            vertices, faces, offset = [], [], 0
            for points, triangles in scene.polyhedra:
                vertices.append(points); faces.append(triangles + offset); offset += len(points)
            mesh = TriangleMesh(identifier='coordination-polyhedra')
            mesh.set_vertices(np.concatenate(vertices)); mesh.set_faces(np.concatenate(faces))
            mesh.vis.color = scene.poly_color
            mesh.vis.transparency = 1 - scene.poly_opacity
            data.objects.append(mesh)
        return data

    def render(self, scene, fit=False):
        camera = self.get_camera() if self.scene is not None else fit_camera(scene)
        self.scene = scene
        self.pipeline.source.data = self.data(scene)
        if fit: camera = fit_camera(scene, camera['position']-camera['focal'], camera['up'])
        self.set_camera(camera)

    def get_camera(self):
        v = self.viewport
        direction = np.array(v.camera_dir); direction /= np.linalg.norm(direction)
        return dict(position=np.array(v.camera_pos), focal=np.array(v.camera_pos)+direction*self.camera_distance,
                    up=np.array(v.camera_up), scale=v.fov)

    def set_camera(self, camera):
        v = self.viewport
        self.camera_distance = np.linalg.norm(camera['focal']-camera['position'])
        v.camera_pos = camera['position']; v.camera_dir = camera['focal']-camera['position']
        v.camera_up = camera['up']; v.fov = camera['scale']

    def image(self, scene, camera, width, height, transparent=False):
        from ovito.vis import Viewport, TachyonRenderer
        from PySide6.QtGui import QImage
        original = self.pipeline.source.data
        try:
            self.pipeline.source.data = self.data(scene)
            v = Viewport(type=Viewport.Type.Ortho, camera_pos=camera['position'],
                         camera_dir=camera['focal']-camera['position'], camera_up=camera['up'], fov=camera['scale'])
            image = v.render_image(size=(width, height), background=(1, 1, 1), alpha=transparent, renderer=TachyonRenderer())
            image = image.convertToFormat(QImage.Format.Format_RGBA8888)
            pixels = np.frombuffer(image.bits(), np.uint8).reshape(height, image.bytesPerLine())[:, :width*4].reshape(height, width, 4)
            return pixels[:, :, :4 if transparent else 3].copy()
        finally:
            self.pipeline.source.data = original

    def close(self):
        self.pipeline.remove_from_scene()
        self.widget.close()


class MatplotlibBackend:
    name = 'Matplotlib (basic)'

    def __init__(self, parent, picked):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        self.figure = Figure(facecolor='white')
        self.widget = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_axes([0, 0, 1, 1], projection='3d')
        self.scene = None
        self.picked = picked
        self.widget.mpl_connect('pick_event', self.pick)
        self.widget.mpl_connect('scroll_event', self.zoom)

    def pick(self, event):
        if event.artist is self.scatter and len(event.ind):
            self.picked(int(self.scene.source_ids[event.ind[0]]))

    def zoom(self, event):
        if event.inaxes == self.ax:
            camera = self.get_camera(); camera['scale'] *= .85 if event.button == 'up' else 1/.85
            self.set_camera(camera)

    def render(self, scene, fit=False):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        camera = self.get_camera() if self.scene is not None else fit_camera(scene)
        self.scene = scene
        self.ax.clear(); self.ax.set_axis_off(); self.ax.set_facecolor('white'); self.ax.set_proj_type('ortho')
        self.ax.set_box_aspect((1, 1, 1), zoom=1.25)
        if scene.polyhedra:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            for vertices, faces in scene.polyhedra:
                self.ax.add_collection3d(Poly3DCollection(vertices[faces], facecolors=[scene.poly_color],
                                                        alpha=scene.poly_opacity, linewidths=0, shade=True))
        spheres = len(scene.positions) <= 500
        self.scatter = self.ax.scatter(*scene.positions.T, s=(scene.radii*100)**2/3,
                                       c=scene.colors, alpha=0 if spheres else 1, picker=5)
        if spheres:
            u, v = np.linspace(0, 2*np.pi, 24), np.linspace(0, np.pi, 16)
            unit = np.array([np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)), np.outer(np.ones_like(u), np.cos(v))])
            for position, radius, color in zip(scene.positions, scene.radii, scene.colors):
                self.ax.plot_surface(*(unit*radius+position[:, None, None]), color=color, linewidth=0, shade=True)
        if scene.show_cell:
            corners = np.array(list(itertools.product((0, 1), repeat=3)))
            points = corners @ scene.cell
            edges = [[points[i], points[j]] for i in range(8) for j in range(i+1, 8) if np.sum(corners[i] != corners[j]) == 1]
            self.ax.add_collection3d(Line3DCollection(edges, colors='#64748b', linewidths=1))
        if scene.labels and len(scene.positions) <= 500:
            from ase.data import chemical_symbols
            for position, number, source in zip(scene.positions, scene.numbers, scene.source_ids):
                self.ax.text(*position, f' {chemical_symbols[number]}{source+1}', fontsize=8)
        if len(scene.bonds):
            self.ax.add_collection3d(Line3DCollection(scene.positions[scene.bonds], colors='#939ca9', linewidths=3))
        if fit: camera = fit_camera(scene, camera['position']-camera['focal'], camera['up'])
        self.set_camera(camera)

    def get_camera(self):
        limits = np.array([self.ax.get_xlim(), self.ax.get_ylim(), self.ax.get_zlim()])
        center = limits.mean(axis=1)
        scale = (limits[0, 1]-limits[0, 0])/2
        return camera_view(dict(position=center+np.array([0, 0, getattr(self, 'camera_distance', 4*scale)]), focal=center,
                                up=np.array([0, 0, 1]), scale=scale), self.ax.elev, self.ax.azim)

    def set_camera(self, camera):
        self.camera_distance = np.linalg.norm(camera['position']-camera['focal'])
        d = camera['position']-camera['focal']; d = d/np.linalg.norm(d)
        self.ax.view_init(elev=np.degrees(np.arcsin(d[2])), azim=np.degrees(np.arctan2(d[1], d[0])))
        c, s = camera['focal'], camera['scale']
        self.ax.set(xlim=(c[0]-s, c[0]+s), ylim=(c[1]-s, c[1]+s), zlim=(c[2]-s, c[2]+s))
        self.widget.draw_idle()

    def image(self, scene, camera, width, height, transparent=False):
        old_scene, old_camera = self.scene, self.get_camera()
        inches = self.figure.get_size_inches().copy()
        facecolor = self.figure.get_facecolor()
        try:
            self.render(scene); self.set_camera(camera)
            if transparent:
                self.figure.set_facecolor('none'); self.ax.set_facecolor('none')
            self.figure.set_size_inches(width/self.figure.dpi, height/self.figure.dpi)
            self.widget.draw()
            return np.asarray(self.widget.buffer_rgba())[:, :, :4 if transparent else 3].copy()
        finally:
            self.figure.set_facecolor(facecolor)
            self.figure.set_size_inches(inches)
            self.render(old_scene); self.set_camera(old_camera)

    def close(self):
        self.widget.close()


BACKENDS = {'PyVista / VTK': PyVistaBackend, 'OVITO': OvitoBackend, 'Matplotlib (basic)': MatplotlibBackend}
