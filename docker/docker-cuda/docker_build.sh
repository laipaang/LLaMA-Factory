docker build -f ./docker/docker-cuda/Dockerfile \
    --build-arg BASE_IMAGE=$base_image \
    --build-arg INSTALL_VLLM=true \
    --build-arg INSTALL_DEEPSPEED=true \
    --build-arg INSTALL_FLASHATTN=true \
    --build-arg PIP_INDEX=$pip_index \
    --build-arg HTTP_PROXY=$http_proxy \
    -t llamafactory:0.9.3.dev0-$version .

