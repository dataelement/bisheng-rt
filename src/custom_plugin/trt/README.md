# trt-lib

This repository provides a script and recipe to build the tensorrt custom op.

### Requirements

- CMake >= 3.13
- CUDA >= 11.6
- Python 3.6
- nvidia tensorflow 1.15
- TensorRT 8.4.2.4 (cuda11.7 cudnn8.5)

## Quick Start Guide

### Build the tensorrt custom op

1. Build the project.
修改build.sh中TRT_RELEASE路径，编译

```bash
sh build.sh
```

### convert trt from pb
```bash
python pb2trt.py
```

### 模型和chaset对应关系
transformer-v2.8-gamma <-> charset_6409.txt
transformer-hand-v1.16 <-> charset_6409.txt
transformer-rare-v1.3 <-> charset_rare_11517.txt
transformer-blank-v0.2 <-> charset_blank_6410.txt
others <-> charset_6410.txt