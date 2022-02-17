# Spidr example

Stand alone example: Loads an image (in binary), computes a spatially informed embedding and saves it to disk.

The example image is provided as .tiff files (`data/CheckeredBoxes_2Ch_32_tiff/*.`), one for each of the two channels. To keep the example code small, it reads the image from a binary file provided (`data/CheckeredBoxes_2Ch_32.bin`. 
A short python script (`eval/eval_embedding.py`) plots the embedding for easy inspection.
