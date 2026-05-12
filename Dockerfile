FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /workspace/unZipro

RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    aria2 \
    bzip2 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Create conda environment
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda create -n unZipro python=3.9 -y

SHELL ["conda", "run", "-n", "unZipro", "/bin/bash", "-c"]

COPY . /workspace/unZipro

# Install dependencies
RUN pip install numpy==1.26.4 --force-reinstall && \
    pip install pandas biotite requests tqdm && \
    pip install torch==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124

ENV PATH=/opt/conda/envs/unZipro/bin:$PATH
ENV CONDA_DEFAULT_ENV=unZipro
CMD ["/bin/bash"]