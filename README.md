# Spatial Information in Dimensionality Reduction (Spidr)

[![DOI](https://zenodo.org/badge/460389824.svg)](https://zenodo.org/badge/latestdoi/460389824)

Introduces spatial neighborhood information in dimensionality reduction methods for high-dimensional images. 
Extends t-SNE such that similarities are based on a point's spatial neighborhood instead of only the high-dimensional point itself (see `python_wrapper\example\example_multiple.py` for examples with UMAP and MDS).

To clone the repo and its external submodules (vcpkg, hnswlib, pybind11, HDILib):

```git clone --recurse-submodule https://github.com/biovault/Spidr/```

Currently, tested on Windows 10 with Visual Studio 2019. Use cmake for setting up the project:
```
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
```

The standard cpp implementation uses the t-SNE implementation from the [HDILib](https://github.com/biovault/HDILib) and [Hnswlib](https://github.com/nmslib/hnswlib) for approximated nearest neighbor search. Other DR techniques might also be used, as shown in the python example below.

The HDILib dependency is currently handled a little different depending on which OS you want to build the Spidr library.
On Linux, the HDILib is build with this project. vcpkg will handle all it's dependencies.
On Windows, you can do the same, but when inlcuding this project into another, the resulting library might have some dll dependencies on lz4 (a flann dependency, which in turn in used by HDILib). This is easier to handle with the [HDILibSlim](https://github.com/alxvth/HDILibSlim), a version of the HDILib without dependencies. Build it, set `BUILD_HDILIB` OFF and `USE_HDILIBSLIM` ON, and provide the `HDILIBSLIM_ROOT` to the cmake folder in the HDILibSlim\lib installation directory. 

On linux systems you need to make sure that some dependencies are installed, e.g. on Ubuntu:
```
sudo apt install curl zip unzip tar pkg-config libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev
```

## Usage
By default, the cmake project creates two targets in addition to the library: 
- See `example/SpidrExample.cpp` for an example on how to use the library in cpp. Set the cmake option `CREATE_EXAMPLE=OFF` to not create this target.
- See `python_wrapper` for install intructions and an example on how to use the library in python. The example showcases spatially informed t-SNE, UMAP and MDS embeddings. Set cmake option `CREATE_PYTHON_WRAPPER=OFF` to not create this target.

## Dependencies
All depedencies are either managed by vcpkg or are submodules of this repo. If you want to build the [HDILib](https://github.com/biovault/HDILib) yourselft, set the cmake option `BUILD_HDILIB=OFF` and provide the variables HDILIB_ROOT, FLANN_ROOT, LZ4_ROOT. See the HDILib documentation for more detail on the HDILib and it's dependencies.

## References
This library implements the methods presented in **Incorporating Texture Information into Dimensionality Reduction for High-Dimensional Images** (2022), published at [PacificVis 2022](https://doi.org/10.1109/PacificVis53943.2022.00010). A preprint is available on arXiv [2202.09179](https://arxiv.org/abs/2202.09179), the conference talk recording and other supplemental material are available [here](http://graphics.tudelft.nl/Publications-new/2022/VVLEH22/).

```
@InProceedings { VVLEH22,
  author       = "Vieth, Alexander and Vilanova, Anna and Lelieveldt, Boudewijn P.F. and Eisemann, Elmar and H\öllt, Thomas",
  title        = "Incorporating Texture Information into Dimensionality Reduction for High-Dimensional Images",
  booktitle    = "2022 15th IEEE Pacific Visualization Symposium (PacificVis)",
  pages        = "11-20",
  year         = "2022",
  doi          = "10.1109/PacificVis53943.2022.00010",
  keywords     = "Mathematics of computing, Dimensionality reduction,  Human-centered computing, Visualization techniques, Human-centered computing, Visual analytics",
  url          = "http://graphics.tudelft.nl/Publications-new/2022/VVLEH22"
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

