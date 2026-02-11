NVIDIA_TRITON_SERVER_VERSION="nvcr.io/nvidia/tritonserver:24.12-py3"

source .env


sudo docker run -d \
           --name triton-server \
           --ipc=host \
           --shm-size=2g \
           -p8000:8000 \
           -p8001:8001 \
           -p8002:8002 \
           -v $(pwd):/apps \
           -w /apps \
           $NVIDIA_TRITON_SERVER_VERSION \
           tritonserver --model-repository=/apps/models_repository


NVIDIA_TRITON_SERVER_VERSION=${NVIDIA_TRITON_SERVER_VERSION} docker compose up -d