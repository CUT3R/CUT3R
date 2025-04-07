FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04 
SHELL ["/bin/bash", "-c"]

# Get rid of annoying tz popup
ENV TZ=America/New_York
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y tzdata
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && dpkg-reconfigure -f noninteractive tzdata

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    git \
    sudo \
    cmake \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx

# Install Miniforge as root in /root's home directory
ENV CONDA_DIR=/root/miniforge3
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p ${CONDA_DIR} && \
    rm Miniforge3-$(uname)-$(uname -m).sh

# Set environment path for root
ENV PATH="${CONDA_DIR}/bin:$PATH"

# Initialize conda for bash and disable auto activation of base
RUN conda init bash && \
    echo "conda config --set auto_activate_base false" >> ~/.bashrc

WORKDIR /CUT3R

# Clone the repository directly to the working directory
RUN git clone https://github.com/CUT3R/CUT3R.git .

# Create the conda environment with Python 3.11 as specified in the README
RUN conda create -n cut3r python=3.11 cmake=3.14.0 -y

# Set up shell to use the conda environment
SHELL ["conda", "run", "-n", "cut3r", "/bin/bash", "-c"]

# Install PyTorch with CUDA 12.1 support
RUN conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies from requirements.txt
RUN pip install -r requirements.txt

# Install additional dependencies mentioned in the README
RUN conda install 'llvm-openmp<16'
RUN pip install evo
RUN pip install open3d
RUN pip install gdown

# Install gsplat with explicit CUDA architecture flags
# Set TORCH_CUDA_ARCH_LIST based on CUDA 12.1 compatibility
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
RUN pip install ninja && \
    pip install git+https://github.com/nerfstudio-project/gsplat.git

# Add conda environment activation to .bashrc
RUN echo "conda activate cut3r" >> ~/.bashrc

# Set environment variable to enable device-side assertions for debugging
ENV TORCH_USE_CUDA_DSA=1
ENV CUDA_LAUNCH_BLOCKING=1

# Compile the cuda kernels for RoPE (as in CroCo v2)
RUN cd /CUT3R/src/croco/models/curope/ && \
    python setup.py build_ext --inplace && \
    python setup.py install && \
    cd ../../../../

# Set the default command to activate the conda environment and start a bash shell
CMD ["bash", "-c", "conda activate cut3r && exec /bin/bash"]
