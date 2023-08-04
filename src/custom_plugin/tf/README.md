# tf-lib

This repository provides a script and recipe to build the tensorflow custom op.

### Requirements

- CMake >= 3.8
- CUDA >= 10.0
- Python 3.6
- Tensorflow 1.13、1.14、1.15

## Quick Start Guide

### Build the tf op

1. Build the project.

 Tensorflow mode:

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TF=ON -DCUDA_VERSION=10.0 -DTF_PATH=/home/gulixin/anaconda3/lib/python3.6/site-packages/tensorflow .. # Tensorflow mode
make
```

Tensorflow nn-predictor mode:

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TF_CC=ON -DCUDA_VERSION=10.0 -DTF_PATH=/home/gulixin/workspace/nn_predictor/nn-predictor/build/tensorflow_cc .. # Tensorflow nn-predictor mode
make
```

2. 如果自己不编译的话，可以从wget ftp://m7-model-gpu08/packages/tf_lib_so.tar.gz下载，解压之后放到tf-lib根目录下
