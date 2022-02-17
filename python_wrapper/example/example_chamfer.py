import spidr
import numpy as np
import matplotlib.pyplot as plt
from os import getcwd
from os.path import dirname as up
from scipy.sparse import csr_matrix
from nptsne import TextureTsne as TSNE  # Texture refers to implementation details, not texture-aware DR
from umap import UMAP
from sklearn.manifold import MDS

from utils import load_binary, assign_embedding_colors

# load data
data_path = up(up(getcwd())) + "\\example\\data\\"
data_name = "CheckeredBoxes_2Ch_32.bin"
data_raw = load_binary(data_path + data_name)

# prep data
data = data_raw.reshape((-1, 2))
data_glob_ids = np.arange(data.shape[0])
imgWidth = int(np.sqrt(data.shape[0]))
imgHeight = imgWidth
numPoints = data.shape[0]
data_img = data.reshape((imgHeight, imgWidth, 2))

# settings
sp_metric = spidr.DistMetric.Bhattacharyya  # Chamfer_pc, QF_hist (define numHistBins),
sp_weight = spidr.WeightLoc.uniform
sp_neighborhoodSize = 1  # one neighbor in each direction, i.e. a 3x3 neighborhood

#################################
# spatially informed embeddings #
#################################

#########
# t-SNE #
#########
print("# Texture-aware t-SNE with HDILib (nptsne)")
# instantiate spidr wrapper
alg_spidr = spidr.SpidrAnalysis(distMetric=sp_metric, kernelType=sp_weight,  # numHistBins=5,
                                numLocNeighbors=sp_neighborhoodSize, aknnAlgType=spidr.KnnAlgorithm.hnsw)

# embed with t-SNE
emb_tsne = alg_spidr.fit_transform(X=data, pointIDsGlobal=data_glob_ids, imgWidth=imgWidth, imgHeight=imgHeight)


########
# UMAP #
########
print("# Texture-aware UMAP with umap-learn")
# instantiate spidr wrapper
alg_spidr = spidr.SpidrAnalysis(distMetric=sp_metric, kernelType=sp_weight,  # numHistBins=5,
                                numLocNeighbors=sp_neighborhoodSize, aknnAlgType=spidr.KnnAlgorithm.hnsw)
nn = alg_spidr.nn

# get knn dists to compute umap
knn_ind, knn_dists = alg_spidr.fit(X=data, pointIDsGlobal=data_glob_ids, imgWidth=imgWidth, imgHeight=imgHeight)

# create sparse matrix with scipy
knn_ind = np.array(knn_ind)  # .reshape((numPoints, nn))
knn_dists = np.array(knn_dists)  # .reshape((numPoints, nn))
knn_ind_row = np.repeat(np.arange(0, numPoints), nn)  # .reshape((numPoints, nn))

knn_csr = csr_matrix((knn_dists, (knn_ind_row, knn_ind)), shape=(numPoints, numPoints))

# embed with umap
alg_umap = UMAP()
emb_umap = alg_umap.fit_transform(knn_csr)


#######
# MDS #
#######
print("# Texture-aware MDS with scikit-learn")
# instantiate spidrlib
alg_spidr = spidr.SpidrAnalysis(distMetric=sp_metric, kernelType=sp_weight,  # numHistBins=5,
                                numLocNeighbors=sp_neighborhoodSize, aknnAlgType=spidr.KnnAlgorithm.full_dist_matrix)

# get full dist matrix to compute mds
_, knn_dists = alg_spidr.fit(X=data, pointIDsGlobal=data_glob_ids, imgWidth=imgWidth, imgHeight=imgHeight)

# create full distance matrix
knn_dists = np.array(knn_dists).reshape((numPoints, numPoints))

# embed with MDS
alg_mds = MDS(dissimilarity='precomputed', n_jobs=-1, random_state=1234)
emb_mds = alg_mds.fit_transform(knn_dists)


#######################
# standard embeddings #
#######################

# standard t-SNE
print("# Standard t-SNE with HDILib (nptsne)")
alg_tsne = TSNE()
emb_tsne_std = alg_tsne.fit_transform(data).reshape((numPoints, 2))

# standard UMAP
print("# Standard UMAP with umap-learn")
alg_umap = UMAP()
emb_umap_std = alg_umap.fit_transform(data)

# standard MDS
print("# Standard MDS with scikit-learn")
alg_mds = MDS(dissimilarity='euclidean', n_jobs=-1)
emb_mds_std = alg_mds.fit_transform(data)


#########
# Plots #
#########

## Plot the embeddings and data
fig, axs = plt.subplots(2, 4, figsize=(8, 5))
fig.suptitle('Data channels and embeddings')

axs[0, 0].title.set_text('Data, ch1')
axs[0, 0].imshow(data_img[:, :, 0], aspect="auto")
axs[1, 0].title.set_text('Data, ch2')
axs[1, 0].imshow(data_img[:, :, 1], aspect="auto")


def pltColEmb(col_n, title, emb, emb_std):
    axs[0, col_n].title.set_text(title + ' w/ chamfer')
    axs[0, col_n].scatter(emb[:, 0], emb[:, 1], alpha=0.5, s=0.5)
    axs[0, col_n].get_xaxis().set_visible(False)
    axs[0, col_n].get_yaxis().set_visible(False)
    axs[1, col_n].title.set_text(title + ' std')
    axs[1, col_n].scatter(emb_std[:, 0], emb_std[:, 1], alpha=0.5, s=0.5)
    axs[1, col_n].get_xaxis().set_visible(False)
    axs[1, col_n].get_yaxis().set_visible(False)


pltColEmb(1, 't-SNE', emb_tsne, emb_tsne_std)
pltColEmb(2, 'UMAP', emb_umap, emb_umap_std)
pltColEmb(3, 'MDS', emb_mds, emb_mds_std)

plt.tight_layout()
plt.show()

## Plot embeddings and recolored images
# map embedding positions to colors and then back to the image space
clm_path = '../../example/eval/2d_Mittelstaed.png'
emb_tsne_colors = assign_embedding_colors(emb_tsne, clm_path, rot90=3)
emb_tsne_std_colors = assign_embedding_colors(emb_tsne_std, clm_path, rot90=3)
emb_umap_colors = assign_embedding_colors(emb_umap, clm_path, rot90=3)
emb_umap_std_colors = assign_embedding_colors(emb_umap_std, clm_path, rot90=3)
emb_mds_colors = assign_embedding_colors(emb_mds, clm_path, rot90=3)
emb_mds_std_colors = assign_embedding_colors(emb_mds_std, clm_path, rot90=3)

# Plot embedding
fig, axs = plt.subplots(2, 6, figsize=(14, 5))
fig.suptitle('Embeddings and data colored based on embeddings')


def pltColProj(col_n, title, emb, emb_cols):
    axs[0, col_n].title.set_text(title)
    axs[0, col_n].scatter(emb[:, 0], emb[:, 1], c=emb_cols, s=5, alpha=0.5)
    axs[0, col_n].get_xaxis().set_visible(False)
    axs[0, col_n].get_yaxis().set_visible(False)
    axs[1, col_n].imshow(emb_cols.reshape((imgHeight, imgWidth, 3)), aspect="auto")
    axs[1, col_n].xaxis.tick_top()


pltColProj(0, 't-SNE w/ chamfer', emb_tsne, emb_tsne_colors)
pltColProj(1, 't-SNE std', emb_tsne_std, emb_tsne_std_colors)
pltColProj(2, 'UMAP w/ chamfer', emb_umap, emb_umap_colors)
pltColProj(3, 'UMAP std', emb_umap_std, emb_umap_std_colors)
pltColProj(4, 'MDS w/ chamfer', emb_mds, emb_mds_colors)
pltColProj(5, 'MDS std', emb_mds_std, emb_mds_std_colors)

plt.tight_layout()
plt.show()
