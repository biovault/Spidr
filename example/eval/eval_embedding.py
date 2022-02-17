# -*- coding: utf-8 -*-
"""
@author: alexander vieth, 2021
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import imageio


def load_binary(path, dtype="<f"):
    with open(path, "rb") as file:
        # specify little endian float
        dat = np.fromfile(file, dtype=dtype)
    return dat


def interpol_texture(im, coords, hexa=False):
    """

    :param im: [y,x]
    :param coords: [y,x]
    :param hexa: (optional, default=False)
    :return:
    """
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

# Load data
loadPath = "../data/"
fileNameEmd = "CheckeredBoxes_2Ch_32_sp-tSNE_Chamfer.bin"
colorMapPath = './2d_Mittelstaed.png'

embedding = load_binary(loadPath + fileNameEmd)
numPoints = int(embedding.size / 2)
embedding = embedding.reshape((numPoints, 2))

# Image dimensions
numX = 32
numY = 32

# Map embedding positions to color
embeddingColors = assign_embedding_colors(embedding, colorMapPath, rot90=3)

# Map the embedding coloring to image space
coloredImg = embeddingColors.reshape((numY, numX, 3))

# Plot embedding
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
imgPos = plotOrder
emdPos = 1 - plotOrder
axs[imgPos].imshow(coloredImg)
axs[imgPos].invert_yaxis()
axs[imgPos].set_axis_off()
axs[emdPos].scatter(embedding[:, 0], embedding[:, 1], c=embeddingColors, s=5, alpha=0.5)
axs[emdPos].set_axis_off()
plt.tight_layout()
plt.show()

