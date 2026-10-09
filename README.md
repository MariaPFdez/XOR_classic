[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23262316.svg)](https://doi.org/10.5281/zenodo.23262316)

# XOR_classic

This repository contains code to train neural networks for input reconstruction tasks. The project supports both fully connected and convolutional autoencoders, which can be trained on multiple datasets: MNIST, EMNIST, and a synthetic Random dataset.

### Overview

The goal of this project is to obtain accurate reconstructions of input data using biologically inspired learning mechanisms.
To this end, we employ a signed XOR-based loss function, a biologically plausible basic motif introduced in [1], and demonstrate that it performs effectively for the specified reconstruction tasks.

### Features

Training of fully connected and convolutional autoencoders

Support for multiple datasets: MNIST, EMNIST, and Random

Biologically inspired XOR-based loss function with sign

Training of an additional linear classifier on the learned representations

Tools for post-training analysis, including:

- Visualization of the latent space using t-SNE

- Visualization of input reconstructions

### Usage

The Jupyter Notebook included in this repository provides a tutorial-style walkthrough of all the available functions. In addition, the scripts can also be accessed and executed directly.

The following section describes how to use them in detail.

#### Training a reconstruction model

The main training script is train.py, which provides the train_net function to train an autoencoder with the desired configuration.

Example usage:

```python
train_net(train_data = 'MNIST', network = 'CNN', save_vars = True)
```

#### Training a classifier

A linear classifier can be trained on the learned latent representations using:

```python
train_class(classif_data = 'MNIST', train_data='MNIST', network='CNN', save_vars = True)
```

#### Analysis

After training, several analysis functions are provided to represent the results, appearing in analysis.py. Here there are examples to use them:

```python
model_accuracy(test_data='MNIST', train_data='MNIST', network='CNN', classif = True)

plot_tsne(test_data='MNIST', train_data='MNIST', network='CNN', save_fig = False)

plot_reconstructions(test_data='MNIST', train_data='MNIST', network='CNN', save_fig = False)

plot_classifications(test_data='MNIST', train_data='MNIST', network='Fully', save_fig = False)
```

#### License

This project is released under the Apache 2.0 License.


#### References
<a id="1">[1]</a> 
Peña M., Marco J., and Lloret L. (2024). 
Implementing engrams from a machine learning perspective: XOR as a basic motif
doi:10.48550/arXiv:2406.09940


#### Citation
If you use this software in your research, please cite:
Peña M., Lloret L. and Marco J. *Training Neural Networks with an XOR-Based Loss Function*. Version 1.0.0. Zenodo. https://doi.org/10.5281/zenodo.23262316

