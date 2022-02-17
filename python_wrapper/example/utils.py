# -*- coding: utf-8 -*-
"""
@author: alexander vieth, 2021
"""


def load_binary(path, dtype="<f"):
    import numpy as np
    with open(path, "rb") as file:
        # specify little endian float: dtype="<f"
        dat = np.fromfile(file, dtype=dtype)
    return dat


def write_binary(dataArray, path):
    with open(path, "wb") as file:
        dataArray.tofile(file)


def interpol_texture(im, coords, hexa=False):
    """

    :param im: [y,x]
    :param coords: [y,x]
    :param hexa: (optional, default=False)
    :return:
    """
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator
    from matplotlib.colors import to_hex
    y = np.arange(im.shape[0])
    x = np.arange(im.shape[1])

    # interpolate data points from the given image im
    interpol = RegularGridInterpolator((y, x), im)
    col = interpol(coords)

    if hexa is True:
        col /= 255
        t = np.array([to_hex(color) for color in col])
    else:
        col = col.astype('uint32')

    return col


def assign_embedding_colors(points, texPath, flipTexUD=False, flipTexLR=False, rot90=0):
    """

    :param points:
    :param texPath:
    :return:
    """
    import imageio
    import numpy as np
    # read texture
    tex = np.array(imageio.imread(texPath))

    if flipTexUD:
        tex = np.flipud(tex)

    if flipTexLR:
        tex = np.fliplr(tex)

    if rot90:
        tex = np.rot90(tex, rot90)

    # normalize data points to texture coordinate range
    coords = points + np.abs(np.min(points, axis=0))
    coords[:, 0] *= (tex.shape[0] - 1) / np.max(coords[:, 0])
    coords[:, 1] *= (tex.shape[1] - 1) / np.max(coords[:, 1])
    # just to be sure
    coords[:, 0] = np.clip(coords[:, 0], 0, tex.shape[0] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, tex.shape[1] - 1)

    # Interpolate values
    colors = interpol_texture(tex, coords, hexa=True)

    return colors
