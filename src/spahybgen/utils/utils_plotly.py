import numpy as np
from plotly import graph_objects as go
import matplotlib


def plot_point_cloud(pts, color='blue', mode='markers', size=3):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode=mode,
        marker=dict(
            color=color,
            size=size
        ),
        hovertext=[str(hi) for hi in range(pts.shape[0])]
    )


def plot_volume(value, vexels=80j, min=0.5, max=1.0):
    X, Y, Z = np.mgrid[0:0.4:vexels, 0:0.4:vexels, 0:0.4:vexels]

    return go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=value.flatten(),
    isomin=min,
    isomax=max,
    opacity=0.2, # needs to be small to see through all surfaces
    surface_count=10, # needs to be a large number for good volume rendering
    )


def plot_vector_cone(pts, vec):
    return go.Cone(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], 
        u=vec[:, 0], v=vec[:, 1], w=vec[:, 2])


def plot_volume_cube(position, cube_size = 0.05, color = 'blue', opacity = 1):
    x= np.array([0, 1, 0, 1, 0, 1, 0, 1])*cube_size + position[0]
    y= np.array([0, 0, 1, 1, 0, 0, 1, 1])*cube_size + position[1]
    z= np.array([0, 0, 0, 0, 1, 1, 1, 1])*cube_size + position[2]
    cube = go.Mesh3d(
        x = x, y=y, z=z,
        i= [0, 3, 4, 7, 0, 6, 1, 7, 0, 5, 2, 7],
        j= [1, 2, 5, 6, 2, 4, 3, 5, 4, 1, 6, 3],
        k= [3, 0, 7, 4, 6, 0, 7, 1, 5, 0, 7, 2],
            opacity=opacity,
            color=color,
            flatshading = True
    )
    return cube


def plot_volume_mesh(volume, volume_length, alpha = 1.0, colormap='hsv', color_length=None, light=None):
    if color_length is None: color_length = len(volume)
    volume_color_indx = np.zeros_like(volume)
    for ind in range(len(volume_color_indx)): volume_color_indx[:, :, ind] = ind if ind < color_length else 0
    volume_vis = volume * volume_color_indx

    jet_12_colors = matplotlib.colormaps[colormap](np.linspace(0, 1, color_length)) # jet turbo rainbow gist_rainbow
    colors = (jet_12_colors*255).astype(int)
    colors[:, 3] = int(alpha * 255)
    data = create_voxel_figure(volume=volume_vis, light=light, colors=colors, 
                                length=volume_length/len(volume_color_indx))
    return data


def plot_volume_heat(volume, volume_length, alpha = 1.0, colormap='hsv', color_length = 100, light=None):
    """
    draw heat volume within range of [0, 1]
    """
    volume_vis = np.array(volume * color_length, dtype=int)
    jet_12_colors = matplotlib.colormaps[colormap](np.linspace(0, 1, color_length+1)) # jet turbo rainbow gist_rainbow
    colors = (jet_12_colors*255).astype(int)
    colors[:, 3] = int(alpha * 255)
    data = create_voxel_figure(volume=volume_vis, light=light, colors=colors, length=volume_length/len(volume))
    return data

##
## Below for plotly volume from git@github.com:Elenterius/python-voxel-plot.git
##

def mesh_each_voxel_as_cube(volume: np.ndarray):
    assert volume.ndim == 3
    cubes = []

    for x in range(volume.shape[0]):
        for y in range(volume.shape[1]):
            for z in range(volume.shape[2]):
                v = volume[x, y, z]
                if v > 0:
                    vertices, faces = _create_cube_mesh(x, y, z)
                    cubes.append([vertices, faces, v])

    return cubes

def mesh_greedy(volume: np.ndarray):
    assert volume.ndim == 3
    volume = volume.astype(dtype=int)
    dims = volume.shape
    mask = np.zeros(4096, dtype=int)
    vertices, faces = [], []

    for d in range(3):
        u, v = (d + 1) % 3, (d + 2) % 3
        xyz = [0, 0, 0]
        q = [0, 0, 0]
        q[d] = 1

        if len(mask) < dims[u] * dims[v]:
            mask = np.zeros(dims[u] * dims[v] + 1, dtype=int)

        xyz[d] = -1
        while xyz[d] < dims[d]:
            # Compute mask
            n = 0
            xyz[v] = 0
            while xyz[v] < dims[v]:
                xyz[u] = 0
                while xyz[u] < dims[u]:
                    a = volume[xyz[0], xyz[1], xyz[2]] if 0 <= xyz[d] else 0
                    b = volume[xyz[0] + q[0], xyz[1] + q[1], xyz[2] + q[2]] if xyz[d] < dims[d] - 1 else 0
                    if a == b:
                        mask[n] = 0
                    elif a != 0:
                        mask[n] = a
                    else:
                        mask[n] = -b
                    xyz[u] += 1
                    n += 1
                xyz[v] += 1

            xyz[d] += 1

            # generate mesh for mask using lexicographic ordering
            n = 0
            for j in range(dims[v]):
                i = 0
                while i < dims[u]:
                    c = mask[n]
                    if c != 0:
                        # compute width
                        width = 1
                        while c == mask[n + width] and i + width < dims[u]:
                            width += 1

                        # compute height
                        done = False
                        height = 1
                        while j + height < dims[v]:
                            for w in range(0, width):
                                if c != mask[n + w + height * dims[u]]:
                                    done = True
                                    break
                            if done:
                                break
                            height += 1

                        # add quad
                        xyz[u], xyz[v] = i, j
                        _add_quad(vertices, faces, xyz, width, height, u, v, c)

                        # zero-out mask
                        for h in range(0, height):
                            for w in range(0, width):
                                mask[n + w + h * dims[u]] = 0
                        i += width
                        n += width
                    else:
                        i += 1
                        n += 1
    return np.array(vertices), np.array(faces)


def _add_quad(vertices, faces, xyz, width, height, u, v, c):
    du = [0, 0, 0]
    dv = [0, 0, 0]
    if c > 0:
        dv[v] = height
        du[u] = width
    else:
        c = -c
        du[v] = height
        dv[u] = width

    vertex_count = len(vertices)
    vertices.append([xyz[0], xyz[1], xyz[2]])
    vertices.append([xyz[0] + du[0], xyz[1] + du[1], xyz[2] + du[2]])
    vertices.append([xyz[0] + du[0] + dv[0], xyz[1] + du[1] + dv[1], xyz[2] + du[2] + dv[2]])
    vertices.append([xyz[0] + dv[0], xyz[1] + dv[1], xyz[2] + dv[2]])
    faces.append([vertex_count, vertex_count + 1, vertex_count + 2, c])
    faces.append([vertex_count, vertex_count + 2, vertex_count + 3, c])


def _create_cube_mesh(x, y, z):
    vertices = []
    for i in range(8):
        vertices.append([x + i // 4, y + i // 2 % 2, z + i % 2])

    faces = np.array([
        [0, 1, 0, 1, 0, 6, 2, 7, 4, 7, 1, 7],
        [1, 2, 1, 4, 2, 2, 3, 3, 5, 5, 3, 3],
        [2, 3, 4, 5, 4, 4, 6, 6, 6, 6, 5, 5]
    ]).T

    return np.array(vertices), faces


# import numpy as np
import random
import plotly.graph_objects as go

# import voxel_mesher

class PastelColorUtil:
    """
    Modified version of the random pastel color script by Andreas Dewes
	original source: https://gist.github.com/adewes/5884820
    """

    @staticmethod
    def random_color(pastel_factor=0.5):
        return [(x + pastel_factor) / (1.0 + pastel_factor) for x in [random.uniform(0, 1.0) for _ in [1, 2, 3]]]

    @staticmethod
    def _color_distance(c1, c2):
        return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])

    @staticmethod
    def generate_color(existing_colors=None, pastel_factor=0.5, alpha: float = None):
        if existing_colors is None or len(existing_colors) == 0:
            color = PastelColorUtil.random_color(pastel_factor=pastel_factor)
            if alpha:
                color.append(alpha)
            return color

        max_distance = None
        best_color = None

        for i in range(0, 100):
            color = PastelColorUtil.random_color(pastel_factor=pastel_factor)
            best_distance = min([PastelColorUtil._color_distance(color, c) for c in existing_colors])
            if not max_distance or best_distance > max_distance:
                max_distance = best_distance
                best_color = color

        if alpha:
            best_color.append(alpha)
        return best_color

    @staticmethod
    def generate_colors(num_colors: int, pastel_factor=0.5, alpha: float = None):
        colors = []
        for i in range(0, num_colors):
            colors.append(PastelColorUtil.generate_color(colors, pastel_factor, alpha))
        return colors


def _create_voxel_mesh_figure(vertices: np.ndarray, faces: np.ndarray, face_colors: np.ndarray, light=None) -> go.Figure:
    x, y, z = vertices.T
    i, j, k = faces.T
    if light is not None: lighting_effects = light
    else: lighting_effects = dict(ambient=0.7, diffuse=0.8, roughness = 0.9, specular=0.6, fresnel=1.0)
    return go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=1,
            facecolor=face_colors,
            lighting=lighting_effects
        )


def create_voxel_figure(volume: np.ndarray, light: str, colors=None, length=1.0) -> go.Figure:
    assert volume.ndim == 3

    vertices, faces_ = mesh_greedy(volume)
    face_colors = faces_.copy()
    faces = faces_[:, :3]

    ids = np.unique(faces_[:, 3])
    if colors is None:
        colors = (np.array(PastelColorUtil.generate_colors(len(ids), 0.8, 0.5)) * 255).astype(int)
        for _ix, _id in enumerate(ids):
            face_colors[(faces_[:, 3] == _id)] = colors[_ix]
    else:
        for _id in (ids):
            face_colors[(faces_[:, 3] == _id)] = colors[_id]
    vertices = vertices*length
    return _create_voxel_mesh_figure(vertices, faces, face_colors, light)