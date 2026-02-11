ARG NVIDIA_TRITON_SERVER_VERSION

FROM ${NVIDIA_TRITON_SERVER_VERSION}

# Install dependencies
RUN pip install opencv-python torchvision && \
    apt update && \
    apt install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*


COPY . /workspace/triton
WORKDIR /workspace/triton
RUN pip install -e .

# Set working directory back
WORKDIR /workspace

CMD ["tritonserver", "--model-repository=/models_repository"]