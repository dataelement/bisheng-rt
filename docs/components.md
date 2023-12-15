
## 基础依赖组件说明

镜像环境： tritonserver:22.10 cuda11.8

更新时间：12.13.2023

- common: r22.10 -> r23.10 (cb62c7 -> cf617c)
- third_party r22.10 -> r23.10 (c5fd70 -> 0d09c0)

- core: r22.10 -> r23.10 (4080df -> 958a24)，更新最新基线代码 + 增量合并历史逻辑
- server: r22.10 -> main (86427d -> 6dfa3e7 -> f2cd99) 直接合并最新基线代码

- backend: r22.10->r23.11 (ebb4aa -> )
- backends/python_backend: r22.08 -> r23.10 (0d24fd -> 67ca86)


更新时间：12.03.2022

- tensorflow r22.08->r22.10 ( fba9a45d404a199e5a76fd4ff00b3ef78040a648 -> b7995ce42cba010d8c633844ef50c121e85106a5
- onnxruntime
- pytorch r22.08->r22.10 00bf7ea157951c12800286a7757d3df5c1ed9aa5 -> 935f4a5afbbece6d79dd9114eff0bf06f2c849f4


更新时间：10.25.2022

- server: r22.08->r22.10, 98ee6a->86427d
- core: r22.08->r22.10, c9cd66->4080df
- backend: r22.08->r22.10, 83fc70->ebb4aa
- common: r22.08->r22.10, d5c561->cb62c7
- tensorrt r22.08->r22.10 73b8f3->ddd62b
- tensorflow
- onnxruntime
- openvino r22.08->r22.10 e3336d->92eaea
- pytorch


更新时间： 09.28.2022

- backend, 83fc70545c040c4e2a3b8c899e90c3ae94263d47, r22.08
- common, d5c561841e9bd0818c40e5153bdb88e98725ee79, r22.08
- core, c9cd6630ecb04bb26e2110cd65a37f23aec8153b, r22.08
- server, 98ee6a10ff896d6bde431ec2fb602bfd624864a7, r22.08
- third_party, c5fd70e8c3fdcb471a326c64f72eb78c4d45d9ee, r22.08
- dali_backend, e3a6d1a63a2df0130ab7ce61517e0e2006697f65, r22.08
- onnxruntime_backend, 4cddb165a302a368fb3e5ccd277816a4b0c9fbc9, r22.08
- openvino_backend, e3336dbb6138e1625547bb35cb68077775bf1f22, r22.08
- paddlepaddle_backend, 6c1c935e9978f5606082cfb372cc74948bcb6e7a, r22.08
- python_backend, 0d24fda0d2459536e1f4e8d5c368aaeda8ce838f, r22.08
- pytorch_packend, 00bf7ea157951c12800286a7757d3df5c1ed9aa5, r22.08
- tensorrt_backend, 73b8f381fa71a2c20cb4cc61fef2529bc0807ba4, r22.08
- client, b4f10a4650a6c3acd0065f063fd1b9c258f10b73, r22.08

## 开发镜像制作

step 1. 执行build.py获得编译`tritonserver_build:latest`

```
export HTTP_PROXY=192.168.106.8:1081
export HTTPS_PROXY=192.168.106.8:1081
export http_proxy=192.168.106.8:1081
export https_proxy=192.168.106.8:1081

./build.py -v --enable-gpu \
    --version 2.25.0 --container-version 22.08 --upstream-container-version 22.08 \
    --backend ensemble --backend tensorrt --backend tensorflow1 --backend pytorch --backend dali --backend python \
    --enable-logging --enable-stats --enable-metrics --enable-gpu-metrics \
    --enable-tracing --enable-nvtx \
    --endpoint grpc --endpoint http

```

step 2. 打tag

- `docker tag tritonserver:latest tritonserver:22.08`
- `docker rmi tritonserver:latest`

step 3. 启动编译容器

`docker run --gpus=all --shm-size 2g --rm --net=host -itd --name hf_2208_dev -v /home/hanfeng:/home/hanfeng tritonserver_buildbase:latest bash`


step 4. udpate cuda 11.6
`update-alternatives --config cuda`
