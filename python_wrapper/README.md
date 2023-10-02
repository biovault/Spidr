# Spidr Python Wrapper

You can use the spidr library from python, using this wrapper based on [this](https://github.com/pybind/cmake_example) pybind11 example.

## Installation
Make sure, that the external pybind11 submodule is present. That should be the case if you cloned the entire repo with the `--recurse-submodule` option. The wrapper is tested with python 3.9 since one some dependencies like nptsne were not available for newer python versions with pip at the time of testing. To build and install the python wrapper, open a terminal at the root dir of this repo - one folder up- and use (for pip < 22.1 I recommend the option ``--use-feature=in-tree-build``):

```bash
pip install .
```

This will handle the build based on the CMakeList.txt.

To enable AXV instructions, before building the package set an environment variable `USE_AVX=ON`. For example, assuming you use a conda shell and a virtual environment:
```conda
conda activate ENV_NAME
conda env config vars set USE_AVX=ON
conda activate ENV_NAME                 // reload conda env to make use of variabel change

conda env config vars unset USE_AVX     // un-set variable to fall back to SEE instructions
conda activate ENV_NAME
```

## Usage

See `example/example_chamfer.py` and `example/example_multiple.py` for t-SNE, UMAP and MDS examples with a synthetic data set. The dependencies of the examples are listed in `requirements.txt`.

These emebddings are based on the data in  `../../example/data/CheckeredBoxes_2Ch_64_tiff/` and recolor the image based on the embeddings and a 2D colormaps, as done in `example/example_multiple.py`.
![Example: embeddings and re-colored data](example/example_multiple_embs.png)

If you are using Anaconda to set up python, be aware of [this error and its solution](https://stackoverflow.com/a/72427700) that you might run into.
If you run into problems using nptsne in the examples, you could try [openTSNE](https://opentsne.readthedocs.io/en/latest/) instead.
