# Spidr wrapper

Creates python bindings for the SpidrLib, based on [this](https://github.com/pybind/cmake_example) pybind11 example.

## Installation
Make sure, that the external pybind11 submodule is present. That should be the case if you cloned the entire repo with the `--recurse-submodule` option. To build and install the python wrapper use (after navigating into this folder in a python shell):

```bash
pip install . --use-feature=in-tree-build
```

To enable AXV instructions, before building the package set an environment variable `USE_AVX=ON`. For example, assuming you use a conda shell and a virtual environment:
```conda
conda activate ENV_NAME
conda env config vars set USE_AVX=ON
conda activate ENV_NAME                 // reload conda env to make use of variabel change

conda env config vars unset USE_AVX     // un-set variable to fall back to SEE instructions
conda activate ENV_NAME
```

## Usage

See `example/example_chamfer.py` and `example/example_multiple.py` for t-SNE, UMAP and MDS examples with a synthetic data set.

These emebddings are based on the data in  `../../example/data/CheckeredBoxes_2Ch_64_tiff/` and recolor the image based on the emebddings and a 2D colormaps, as done in `example/example_multiple.py`.
![Example: embeddings and re-colored data](example/example_multiple_embs.png)