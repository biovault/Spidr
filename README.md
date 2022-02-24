# Spatial Information in Dimensionality Reduction (Spidr)

[![DOI](https://zenodo.org/badge/460389824.svg)](https://zenodo.org/badge/latestdoi/460389824)

Introduces spatial neighborhood information in dimensionality reduction for high-dimensional images. 
Extends t-SNE such that similarities are based on a point's spatial neighborhood instead of only the high-dimensional point itself.

To clone the repo and its external submodules (hnswlib, glfw, spdlog):

```git clone --recurse-submodule https://github.com/biovault/Spidr/```

Currently, tested on Windows with Visual Studio 2019. Use cmake for setting up the project:
```
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
```

The standard cpp implementation uses the A-tSNE implementation from the [HDILib](https://github.com/biovault/HDILib) and [Hnswlib](https://github.com/nmslib/hnswlib) for approximated nearest neighbor search. Other DR techniques might also be used, as shown in the python example below.

## Usage

See `example/SpidrExample.cpp` for a example on how to use the library in cpp.

See `python_wrapper` for install intructions and a usage example on how to use the library in python. The example showcases spatially informed t-SNE, UMAP and MDS embeddings.

## Dependencies
Not all dependencies are included in this repo (see submodules in `external/`), some need to be downloaded/installed by yourself. 
Make sure to adjust your system variables respectively:
- [HDILibSlim](https://github.com/alxvth/HDILib-slim) (build and install the library and define the system variable `HDILIBSLIM_ROOT` pointing to the install DIR for cmake to automatically find the library.)
- OpenMP

## References
This library implements the methods presented in **Incorporating Texture Information into Dimensionality Reduction for High-Dimensional Images** (2022), to be published at PacificVis 2022. A preprint is available on arxiv [2202.09179](https://arxiv.org/abs/2202.09179).

```
@inproceedings { VVLEH22,
  author       = "Vieth, Alexander and Vilanova, Anna and Lelieveldt, Boudewijn P.F. and Eisemann, Elmar and H\öllt, Thomas",
  title        = "Incorporating Texture Information into Dimensionality Reduction for High-Dimensional Images",
  booktitle    = "2022 15th IEEE Pacific Visualization Symposium (PacificVis)",
  year         = "2022",
  keywords     = "Mathematics of computing, Dimensionality reduction,  Human-centered computing, Visualization techniques, Human-centered computing, Visual analytics",
  url          = "https://arxiv.org/abs/2202.09179https://arxiv.org/abs/2202.09179"
}

@software{alexander_vieth_2022_6120880,
  author       = {Alexander Vieth},
  title        = {biovault/Spidr: PacificVis 2022},
  month        = feb,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v0.1},
  doi          = {10.5281/zenodo.6120880},
  url          = {https://doi.org/10.5281/zenodo.6120880}
}
```

