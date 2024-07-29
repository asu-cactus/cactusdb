FROM amd64/ubuntu:22.04
WORKDIR /home
ENV HOME=/home

RUN apt-get update \
    && apt-get install -y locales \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ARG DEBIAN_FRONTEND=noninteractive
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
RUN apt-get install -y tmux
RUN apt-get install -y sudo
RUN apt-get install -y libpq-dev
RUN apt-get install -y libjsoncpp-dev
RUN apt-get install -y libxsimd-dev
RUN apt-get install -y curl
RUN apt-get install -y zip
RUN apt-get install -y default-jre 
RUN apt-get install -y kmod
RUN apt-get install -y nlohmann-json3-dev
RUN pip3 install numpy==1.26.2 pandas==2.1.3 pyarrow==14.0.0 gdown protobuf psycopg2
RUN apt-get install -y openjdk-11-jdk
RUN apt install -y openssh-server openssh-client
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN ln -s /usr/bin/python3 /usr/bin/python
# install github cli (optional)
RUN type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && sudo apt update \
    && sudo apt install gh -y
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
# TODO: add support of CUDA, still some issues there
# RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
# RUN chmod +x ./cuda_11.8.0_520.61.05_linux.run && ./cuda_11.8.0_520.61.05_linux.run --silent
# # install cudnn
# RUN apt-get install -y libnccl3-dev



# set-up eigen
WORKDIR /usr/include
RUN ln -sf eigen3/Eigen Eigen
RUN ln -sf eigen3/unsupported unsupported

# configure postgres
RUN apt-get install -qq postgresql-14
ENV POSTGRES_USER postgresdb
ENV POSTGRES_PASSWORD postgresdb
ENV POSTGRES_DB postgresdb
RUN service postgresql start \
    && sudo -u postgres psql -c "CREATE USER $POSTGRES_USER WITH SUPERUSER PASSWORD '$POSTGRES_PASSWORD';" \
    && sudo -u postgres createdb -O $POSTGRES_USER $POSTGRES_DB
COPY pg_hba.conf /etc/postgresql/$PG_MAJOR/main/pg_hba.conf
RUN echo "service postgresql start" >> /etc/bash.bashrc


WORKDIR /home
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip
RUN unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cpu.zip
ENV Torch_DIR=/home/libtorch
# set up evadb
WORKDIR /home
RUN git clone https://github.com/lixi-zhou/evadb.git
WORKDIR /home/evadb
RUN git checkout array
RUN pip install ./[ray]

# install cpr
WORKDIR /home
RUN git clone https://github.com/libcpr/cpr.git \
    && cd cpr && mkdir build && cd build \
    && cmake .. -DCPR_USE_SYSTEM_CURL=ON && cmake --build . --parallel \
    && make install 

# set up duckdb
WORKDIR /home
RUN wget https://github.com/duckdb/duckdb/archive/refs/tags/v0.8.1.tar.gz \
    && tar -xf v0.8.1.tar.gz \
    && cd duckdb-0.8.1 \
    && mkdir build && cd build \
    && CMAKE_FLAGS="-DBUILD_UNITTESTS=OFF -DENABLE_SANITIZER=OFF -DENABLE_UBSAN=OFF -DBUILD_SHELL=OFF -DEXPORT_DLL_SYMBOLS=OFF" \
    && cmake ${CMAKE_FLAGS}  .. \
    && make install -j 16

# set up xgboost c api
WORKDIR /home
RUN git clone --recursive https://github.com/dmlc/xgboost \
    && cd xgboost \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install .

# set up spark
WORKDIR /home
RUN wget https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
RUN tar -xvzf spark-3.5.0-bin-hadoop3.tgz
ENV SPARK_HOME=/home/spark-3.5.0-bin-hadoop3
ENV PATH=$PATH:$SPARK_HOME/bin
RUN pip install pyspark

# set up hadoop
# refered setup: https://medium.com/@festusmorumbasi/installing-hadoop-on-ubuntu-20-04-4610b6e0391e
WORKDIR /home 
# generate ssh key for hadoop
RUN ssh-keygen -t rsa -b 4096 -f /root/.ssh/id_rsa -N ""
RUN cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys
RUN echo "PasswordAuthentication no" >> /etc/ssh/sshd_config
RUN echo "PermitRootLogin without-password" >> /etc/ssh/sshd_config
RUN echo "service ssh start" >> /etc/bash.bashrc
RUN wget https://dlcdn.apache.org/hadoop/common/hadoop-3.4.0/hadoop-3.4.0.tar.gz
RUN tar -xvzf hadoop-3.4.0.tar.gz
RUN echo "export HADOOP_HOME=/home/hadoop-3.4.0" >> /etc/bash.bashrc
RUN echo "export PATH=\$PATH:\$HADOOP_HOME/bin" >> /etc/bash.bashrc
RUN echo "export PATH=\$PATH:\$HADOOP_HOME/sbin" >> /etc/bash.bashrc
RUN echo "export HDFS_NAMENODE_USER=root" >> /home/hadoop-3.4.0/etc/hadoop/hadoop-env.sh
RUN echo "export HDFS_DATANODE_USER=root" >> /home/hadoop-3.4.0/etc/hadoop/hadoop-env.sh
RUN echo "export HDFS_SECONDARYNAMENODE_USER=root" >> /home/hadoop-3.4.0/etc/hadoop/hadoop-env.sh
RUN echo "export YARN_RESOURCEMANAGER_USER=root" >> /home/hadoop-3.4.0/etc/hadoop/hadoop-env.sh
RUN echo "export YARN_NODEMANAGER_USER=root" >> /home/hadoop-3.4.0/etc/hadoop/hadoop-env.sh
RUN echo "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64" >> /home/hadoop-3.4.0/etc/hadoop/hadoop-env.sh
RUN mkdir /home/hadoopdata
COPY hadoop_conf/core-site.xml /home/hadoop-3.4.0/etc/hadoop/core-site.xml
COPY hadoop_conf/hdfs-site.xml /home/hadoop-3.4.0/etc/hadoop/hdfs-site.xml
COPY hadoop_conf/mapred-site.xml /home/hadoop-3.4.0/etc/hadoop/mapred-site.xml
COPY hadoop_conf/yarn-site.xml /home/hadoop-3.4.0/etc/hadoop/yarn-site.xml
RUN /home/hadoop-3.4.0/bin/hdfs namenode -format

# set up velox
WORKDIR /home
RUN git clone https://github.com/facebookincubator/velox.git
WORKDIR /home/velox

RUN git remote remove origin
RUN git remote add upstream https://github.com/facebookincubator/velox.git
RUN git remote set-url --push upstream no_push

# set num threads for compilation
ENV NUM_THREADS=16

# download data
# RUN mkdir data
# WORKDIR /home/velox/data
# RUN gdown 1iCmXYbCMaSyT5dH5pYQ275-pPjuEPf3D
# RUN unzip two_tower_data.zip
# WORKDIR /home/velox

# set-up vscode
WORKDIR /home
RUN gdown 16fk_0viXZmCKQc5PSdRYfDNMqe3mUkrJ
RUN chmod +x vscode_cli
RUN echo "./vscode_cli tunnel --accept-server-license-terms --cli-data-dir $(mktemp -d)" > "start_vscode_tunnel.sh"
RUN chmod +x start_vscode_tunnel.sh