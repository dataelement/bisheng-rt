build_image() {
  # build basic image
  tag=11.1-cudnn8-devel-ubuntu16.04
  docker build -t trt/dev:${tag} -f "ubuntu.Dockerfile" .
}

export_image() {
  tag=11.1-cudnn8-devel-ubuntu16.04
  docker save -o trt_dev_${tag}.tar trt/dev:${tag}
  scp trt_dev_${tag}.tar public@m7-model-gpu08:/home/public/ftp/release/images/
}

run_container() {
    echo "run docker image"
    DOCKER_IMAGE="trt/dev:11.1-cudnn8-devel-ubuntu16.04"
    DEV_MOUNT="-v /home/gulixin:/home/gulixin"

    container_name="tensorrt_glx_dev-ubuntu"
    docker run --pid=host --name ${container_name} -p 8902:8902 ${DEV_MOUNT} -d -it --rm --gpus=all $DOCKER_IMAGE bash
    if [ $? -eq 0 ]; then
        echo "run  docker image success"
    else
        echo "run docker image failed!"
        exit 1
    fi
}

# build_image
# run_container
export_image