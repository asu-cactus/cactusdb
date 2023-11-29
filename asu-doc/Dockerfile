FROM amd64/ubuntu:22.04
WORKDIR /home
ENV HOME=/home

RUN apt-get update -y
RUN apt-get install -y libopenblas-dev unzip 
RUN apt-get install -y ninja-build
RUN apt-get install -y libboost-all-dev
RUN apt-get install -y libssl-dev
RUN apt install -y libsodium-dev libsodium23
RUN apt install -y libgoogle-glog-dev
RUN apt-get install -y zstd
RUN apt-get install -y libzstd-dev
RUN apt-get install -y liblz4-dev
RUN apt-get install -y liblzo2-dev
RUN apt-get install -y libsnappy-dev
RUN apt-get install -y libre2-dev
RUN apt-get install -y bison
RUN apt-get install -y flex
RUN apt-get install -y wget
RUN apt-get install -y git
RUN apt-get install -y cmake
RUN apt-get install -y build-essential
RUN apt-get install -y libdouble-conversion-dev
RUN apt-get install -y libgdl-3-dev
RUN apt-get install -y vim
RUN apt-get install -y libeigen3-dev
RUN apt-get install -y libprotobuf-dev protobuf-compiler
RUN apt-get install -y python3-pip
RUN pip3 install numpy==1.26.2 pandas==2.1.3 pyarrow==14.0.0 gdown protobuf


# set-up eigen
WORKDIR /usr/include
RUN ln -sf eigen3/Eigen Eigen
RUN ln -sf eigen3/unsupported unsupported

WORKDIR /home
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip
RUN unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cpu.zip
ENV Torch_DIR=/home/libtorch
RUN git clone https://github.com/facebookincubator/velox.git
WORKDIR /home/velox

RUN git remote remove origin
RUN git remote add upstream https://github.com/facebookincubator/velox.git
RUN git remote set-url --push upstream no_push

# set num threads for compilation
ENV NUM_THREADS=16

# download data
RUN mkdir data
WORKDIR /home/velox/data
RUN gdown 1iCmXYbCMaSyT5dH5pYQ275-pPjuEPf3D
RUN unzip two_tower_data.zip
WORKDIR /home/velox

# set-up vscode
WORKDIR /home
RUN gdown 16fk_0viXZmCKQc5PSdRYfDNMqe3mUkrJ
RUN chmod +x vscode_cli
RUN echo "./vscode_cli tunnel --accept-server-license-terms --cli-data-dir $(mktemp -d)" > "start_vscode_tunnel.sh"
RUN chmod +x start_vscode_tunnel.sh