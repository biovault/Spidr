"""
Same as example.py but here we create embeddings for three metrics and plot them all together

"""
import spidr
import numpy as np
import matplotlib.pyplot as plt
from os import getcwd
from os.path import dirname as up
from os.path import join as concat_path
from scipy.sparse import csr_matrix
from nptsne import TextureTsne as TSNE  # Texture refers to implementation details, not texture-aware DR
from umap import UMAP
from sklearn.manifold import MDS

from utils import load_binary, assign_embedding_colors

print("example_multiple.py")
print("# Load data")

# load data
data_path = concat_path(up(up(up(__file__))), 'example', 'data')
data_name = "CheckeredBoxes_2Ch_64.bin"
data_raw = load_binary(concat_path(data_path, data_name))

# prep data
data = data_raw.reshape((-1, 2))
data_glob_ids = np.arange(data.shape[0])
imgWidth = int(np.sqrt(data.shape[0]))
imgHeight = imgWidth
numPoints = data.shape[0]
data_img = data.reshape((imgHeight, imgWidth, 2))

# settings
sp_metrics = [spidr.DistMetric.Bhattacharyya, spidr.DistMetric.QF_hist, spidr.DistMetric.Chamfer_pc]
sp_weight = spidr.WeightLoc.uniform
sp_neighborhoodSize = 1  # one neighbor in each direction, i.e. a 3x3 neighborhood

iterations = 2000

# histogram qf distance
numHistBins = 5

# only t-SNE
perplexity = 30

##################
# Plots the data #
##################
print("# Plot the data")

data_min = np.min(data_img)
data_max = np.max(data_img)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
#fig.suptitle('Data channels and embeddings')

axs[0].title.set_text('Channel 1')
im1 = axs[0].imshow(data_img[:, :, 0], aspect="auto", vmin=data_min, vmax=data_max)
axs[1].title.set_text('Channel 2')
im2 = axs[1].imshow(data_img[:, :, 1], aspect="auto", vmin=data_min, vmax=data_max)

fig.colorbar(im2, ax=axs.ravel().tolist())

plt.show()


#################################
# spatially informed embeddings #
#################################
print("# Begin computing spatially informed embeddings")

embs_tsne_sp = {}
embs_umap_sp = {}
embs_mds_sp = {}

embs = {"tsne": embs_tsne_sp, "umap": embs_umap_sp, "mds": embs_mds_sp}

for sp_metric in sp_metrics:
    print(f"Metric: {sp_metric}")
    #########
    # t-SNE #
    #########
    print("# Texture-aware t-SNE with HDILib (nptsne)")
    # instantiate spidr wrapper
    alg_spidr = spidr.SpidrAnalysis(distMetric=sp_metric, kernelType=sp_weight, perplexity=perplexity, numHistBins=numHistBins,
                                    numLocNeighbors=sp_neighborhoodSize, aknnAlgType=spidr.KnnAlgorithm.hnsw,
                                    numIterations=iterations)

    # embed with t-SNE
    embs["tsne"][sp_metric] = alg_spidr.fit_transform(X=data, pointIDsGlobal=data_glob_ids, imgWidth=imgWidth, imgHeight=imgHeight)


    ########
    # UMAP #
    ########
    print("# Texture-aware UMAP with umap-learn")
    print("# Texture-aware UMAP with umap-learn: compute knn")
    # instantiate spidrlib
    alg_spidr = spidr.SpidrAnalysis(distMetric=sp_metric, kernelType=sp_weight, numHistBins=numHistBins,
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
    print("# Texture-aware UMAP with umap-learn: compute transformation")
    seed_umap = 123
    alg_umap = UMAP(random_state=seed_umap, n_epochs=iterations)
    embs["umap"][sp_metric] = alg_umap.fit_transform(knn_csr)


    #######
    # MDS #
    #######
    print("# Texture-aware MDS with scikit-learn")
    print("# Texture-aware MDS with scikit-learn: compute full distance matrix")
    # instantiate spidrlib
    alg_spidr = spidr.SpidrAnalysis(distMetric=sp_metric, kernelType=sp_weight, numHistBins=numHistBins,
                                    numLocNeighbors=sp_neighborhoodSize, aknnAlgType=spidr.KnnAlgorithm.full_dist_matrix)

    # get full dist matrix to compute mds
    _, knn_dists = alg_spidr.fit(X=data, pointIDsGlobal=data_glob_ids, imgWidth=imgWidth, imgHeight=imgHeight)

    # create full distance matrix
    knn_dists = np.array(knn_dists).reshape((numPoints, numPoints))

    # check if symmetric
    if not np.allclose(knn_dists, knn_dists.T, atol=1E-10):
        # This condition must be met for the scipy MDS implementation
        # make sure, that the asymmetry is due to numeric issues
        for tol in [1E-3, 1E-4, 1E-5, 1E-8, 1E-7, 1E-8, 1E-9, 1E-10]:
            if not np.allclose(knn_dists, knn_dists.T, atol=tol):
                break
        if tol >= 1E-3:
            raise ValueError("knn_dists is not symmetric")
        else:
            print(f"# Texture-aware MDS with scikit-learn: knn_dists is only symmetric up to a tolerance of {tol} due "
                  "to numeric issues. Automatically making it symmetric.")
            knn_dists = (knn_dists + knn_dists.T) / 2

    # embed with MDS
    print("# Texture-aware MDS with scikit-learn: compute transformation")
    seed_mds = 123456
    alg_mds = MDS(dissimilarity='precomputed', n_jobs=-1, random_state=seed_mds, max_iter=iterations)
    embs["mds"][sp_metric] = alg_mds.fit_transform(knn_dists)

print("# Finished computing spatially informed embeddings")

#######################
# standard embeddings #
#######################
print("# Begin computing standard embeddings")
# standard t-SNE
print("# Standard t-SNE with HDILib (nptsne)")
#alg_tsne = TSNE(perplexity=perplexity, iterations=iterations)
#emb_tsne_std = alg_tsne.fit_transform(data).reshape((numPoints, 2))  # there might be a bug in nptsne causing problems on linux. you might as well go ahead here with emb_tsne_std = data.copy()
emb_tsne_std = data.copy()

# standard UMAP
print("# Standard UMAP with umap-learn")
alg_umap = UMAP(n_epochs=iterations)
emb_umap_std = alg_umap.fit_transform(data)

# standard MDS
print("# Standard MDS with scikit-learn")
alg_mds = MDS(dissimilarity='euclidean', n_jobs=-1, max_iter=iterations)
emb_mds_std = alg_mds.fit_transform(data)

print("# Finished computing standard embeddings")

####################
# Plots embeddings #
####################
print("# Plot embeddings")

# map embedding positions to colors and then back to the image space
clm_path = '../../example/eval/2D_Mittelstaed.png'

embs_tsne_sp_colors = {}
embs_umap_sp_colors = {}
embs_mds_sp_colors = {}

for sp_metric in sp_metrics:
    embs_tsne_sp_colors[sp_metric] = assign_embedding_colors(embs["tsne"][sp_metric], clm_path, rot90=3)
    embs_umap_sp_colors[sp_metric] = assign_embedding_colors(embs["umap"][sp_metric], clm_path, rot90=3)
    embs_mds_sp_colors[sp_metric] = assign_embedding_colors(embs["mds"][sp_metric], clm_path, rot90=3)

emb_umap_std_colors = assign_embedding_colors(emb_umap_std, clm_path, rot90=3)
emb_tsne_std_colors = assign_embedding_colors(emb_tsne_std, clm_path, rot90=3)
emb_mds_std_colors = assign_embedding_colors(emb_mds_std, clm_path, rot90=3)

# Plot embedding
fig, axs = plt.subplots(3, 8, figsize=(15, 5))
#fig.suptitle('Embeddings and data colored based on embeddings')

def pltColProj(row_n, col_n, title, emb, emb_cols):
    # emb scatter
    axs[row_n, col_n].scatter(emb[:, 0], emb[:, 1], c=emb_cols, s=5, alpha=0.5)
    #axs[row_n, col_n].title.set_text(title)

    axs[row_n, col_n].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axs[row_n, col_n].tick_params(axis='y', which='both', left=False, labelleft=False)

    # img re-colored
    axs[row_n, col_n+1].imshow(emb_cols.reshape((imgHeight, imgWidth, 3)), aspect="auto")
    axs[row_n, col_n+1].xaxis.tick_top()
    axs[row_n, col_n+1].invert_yaxis()
    axs[row_n, col_n+1].get_xaxis().set_visible(False)
    axs[row_n, col_n+1].get_yaxis().set_visible(False)


for metric_id, sp_metric in enumerate(sp_metrics):
    col_id = 2 + 2*metric_id
    pltColProj(0, col_id, f't-SNE w/ {sp_metric.name}', embs["tsne"][sp_metric], embs_tsne_sp_colors[sp_metric])
    pltColProj(1, col_id, f'UMAP w/ {sp_metric.name}', embs["umap"][sp_metric], embs_umap_sp_colors[sp_metric])
    pltColProj(2, col_id, f'MDS w/ {sp_metric.name}', embs["mds"][sp_metric], embs_mds_sp_colors[sp_metric])

pltColProj(0, 0, 't-SNE std', emb_tsne_std, emb_tsne_std_colors)
pltColProj(1, 0, 'UMAP std', emb_umap_std, emb_umap_std_colors)
pltColProj(2, 0, 'MDS std', emb_mds_std, emb_mds_std_colors)

# label rows
# xytext depends on scatterplot ranges, automating this would be better
pad = 5
axs[0, 0].set_ylabel("t-SNE")
axs[1, 0].set_ylabel("UMAP")
axs[2, 0].set_ylabel("MDS")

# label columns
height_lab_c = 0.91
plt.figtext(0.22, height_lab_c, "Standard", va="center", ha="center", size=10)
plt.figtext(0.41, height_lab_c, "Chamfer point cloud", va="center", ha="center", size=10)
plt.figtext(0.61, height_lab_c, "Histograms", va="center", ha="center", size=10)
plt.figtext(0.81, height_lab_c, "Bhattacharyya", va="center", ha="center", size=10)

#plt.tight_layout()
plt.show()
#plt.savefig("example_several_DR_and_metrics.pdf", format="pdf", bbox_inches="tight")


###################
# Save embeddings #
###################
print("# Save embeddings")

from utils import write_binary
save_path = concat_path(up(__file__), "embeddings")

print(f"# Saving all embeddings to: {save_path}")
# spatially informed embeddings
for sp_metric in sp_metrics:
    for embs_name, embs_dict in embs.items():
        perp_str = f"_P{perplexity}" if embs_name == "tsne" else ""
        save_name = f"{data_name.split('.')[0]}_sp-{embs_name}_emb{perp_str}_I{iterations}_k1_{sp_metric.name}.bin"
        write_binary(embs_dict[sp_metric].flatten().astype(np.float32), save_path + save_name)

# standard embeddings
write_binary(emb_tsne_std.flatten().astype(np.float32), save_path + f"{data_name.split('.')[0]}_std-tsne_emb_I{iterations}_P{perplexity}.bin")
write_binary(emb_umap_std.flatten().astype(np.float32), save_path + f"{data_name.split('.')[0]}_std-umap_emb_I{iterations}.bin")
write_binary(emb_mds_std.flatten().astype(np.float32), save_path + f"{data_name.split('.')[0]}_std-mds_emb_I{iterations}.bin")

