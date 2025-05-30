This file provides detailed instructions for installing the dependencies manually in your environment.

- [PostgreSQL](#postgresql)
- [XGBoost C APIs](#xgboost-c-apis)
- [LibTorch C APIs](#libtorch-c-apis)
  - [CPU-only Installation](#cpu-only-installation)
  - [CPU+GPU Installation (CUDA enabled)](#cpugpu-installation-cuda-enabled)
- [EvaDB](#evadb)
- [Spark](#spark)
- [Eigen](#eigen)
- [Catch2 \& H5CPP](#catch2--h5cpp)
- [CPR (C++ HTTP Requests Library)](#cpr-c-http-requests-library)
- [DuckDB](#duckdb)
- [Hadoop](#hadoop)
- [Madlib](#madlib)
- [PostgresML](#postgresml)
- [tokenizers-cpp](#tokenizers-cpp)
- [Faiss](#faiss)

## Summary

| Libraries      | Version                               |
|----------------|---------------------------------------|
| Catch2         | @74fcff6                              |
| H5Cpp          | @10a5719                              |
| Postgres       | 14                                    |
| EvaDB          | 0.3.9 with patch for array support    |
| CPR            | @e421287                              |
| DuckDB         | 0.8.1                                 |
| XGBoost        | @614cd54                               |
| Spark          | 3.5.0                                 |
| Hadoop         | 3.4.0                                 |
| Madlib         | 2.1.0 with patch for ML model support |
| Faiss          | @4c13a88                              |
| Tokenizers-cpp | @4fbe996                              |

## Libraries

### PostgreSQL

```bash
# Install PostgreSQL 14
apt-get install -qq postgresql-14

# Set environment variables
export POSTGRES_USER=postgresdb
export POSTGRES_PASSWORD=postgresdb
export POSTGRES_DB=postgresdb

# Start PostgreSQL service and create user and database
service postgresql start \
  && sudo -u postgres psql -c "CREATE USER $POSTGRES_USER WITH SUPERUSER PASSWORD '$POSTGRES_PASSWORD';" \
  && sudo -u postgres createdb -O $POSTGRES_USER $POSTGRES_DB

# Copy custom pg_hba.conf
cp ./docker-doc/pg_hba.conf /etc/postgresql/14/main/pg_hba.conf

# Ensure PostgreSQL service starts on bash login
echo "service postgresql start" >> /etc/bash.bashrc
```

### XGBoost C APIs

```bash
git clone --recursive https://github.com/dmlc/xgboost \
    && cd xgboost \
    && git checkout 614cd54 \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install .
```

### LibTorch C APIs

#### CPU-only Installation

Download the pre-built file from the Torch website and set the corresponding environment parameter.

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cpu.zip
export Torch_DIR=PATH_TO_LIBTORCH_ROOT
```

Note: If you are running an ARM-based OS, it is recommended to build the library from source as we have not tested the pre-built files for ARM-based instances. You can try the following command to build the project:

```bash
git clone https://github.com/pytorch/pytorch.git
cd pytorch
python3 setup.py install
# Modify the path correspondingly
export Caffe2_DIR=/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Caffe2
export Torch_DIR=/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch
```

#### CPU+GPU Installation (CUDA enabled)

Download the pre-built files for CUDA and set the environment variable:

```bash
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cu124.zip
export Torch_DIR=PATH_TO_LIBTORCH_ROOT
```

### EvaDB

EvaDB itself does not natively support the datatype `ARRAY` in PostgreSQL. We provide a patch to support that, please install it by following the instructions:

```bash
git clone https://github.com/lixi-zhou/evadb.git
cd /home/evadb
git checkout array
pip install ./[ray]
```

### Spark

```bash
wget https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
tar -xvzf spark-3.5.0-bin-hadoop3.tgz
export SPARK_HOME=PATH_TO_SPARK_ROOT_FOLDER
export PATH=$PATH:$SPARK_HOME/bin
pip install pyspark
```

### Eigen

Eigen can be installed directly through `apt-get update && apt-get install -y libeigen3-dev`. However, the Eigen library may need to be manually linked to the include folder through the following command:

```bash
cd /usr/include
ln -sf eigen3/Eigen Eigen
ln -sf eigen3/unsupported unsupported
```

### Catch2 & H5CPP

Catch2 is a dependency for H5CPP, a library to load H5 files into C++ programs.

```bash
# Install Catch2
git clone https://github.com/catchorg/Catch2.git \
  && cd Catch2 \
  && git checkout 74fcff6 \
  && cmake -Bbuild -H. -DBUILD_TESTING=OFF \
  && cmake --build build/ --target install
cd ~
git clone https://github.com/ess-dmsc/h5cpp.git \
  && cd h5cpp \
  && git checkout 10a5719 \
  && mkdir build \
  && cd build \
  && cmake .. -DH5CPP_CONAN=DISABLE \
  && make install
```

### CPR (C++ HTTP Requests Library)

CPR library is used for RESTful API requests.

```bash
git clone https://github.com/libcpr/cpr.git \
    && cd cpr && git checkout e421287 && mkdir build && cd build \
    && cmake .. -DCPR_USE_SYSTEM_CURL=ON && cmake --build . --parallel \
    && make install
```

### DuckDB

```bash
wget https://github.com/duckdb/duckdb/archive/refs/tags/v0.8.1.tar.gz \
    && tar -xf v0.8.1.tar.gz \
    && cd duckdb-0.8.1 \
    && mkdir build && cd build \
    && CMAKE_FLAGS="-DBUILD_UNITTESTS=OFF -DENABLE_SANITIZER=OFF -DENABLE_UBSAN=OFF -DBUILD_SHELL=OFF -DEXPORT_DLL_SYMBOLS=OFF" \
    && cmake ${CMAKE_FLAGS} .. \
    && make install -j 16
```

### Hadoop

```bash
# Generate SSH key for Hadoop
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
echo "PasswordAuthentication no" >> /etc/ssh/sshd_config
echo "PermitRootLogin without-password" >> /etc/ssh/sshd_config
echo "service ssh start" >> ~/.bashrc

# Download and extract Hadoop
wget https://dlcdn.apache.org/hadoop/common/hadoop-3.4.0/hadoop-3.4.0.tar.gz
tar -xvzf hadoop-3.4.0.tar.gz
export HADOOP_HOME=~/hadoop-3.4.0
echo "export HADOOP_HOME=$HADOOP_HOME" >> ~/.bashrc
echo "export PATH=\$PATH:\$HADOOP_HOME/bin" >> ~/.bashrc
echo "export PATH=\$PATH:\$HADOOP_HOME/sbin" >> ~/.bashrc

# Set Hadoop environment variables
echo "export HDFS_NAMENODE_USER=root" >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh
echo "export HDFS_DATANODE_USER=root" >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh
echo "export HDFS_SECONDARYNAMENODE_USER=root" >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh
echo "export YARN_RESOURCEMANAGER_USER=root" >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh
echo "export YARN_NODEMANAGER_USER=root" >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh
echo "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64" >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh

# Create Hadoop data directory
mkdir ~/hadoopdata

# Copy Hadoop configuration files
cp ./docker-doc/hadoop_conf/core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml
cp ./docker-doc/hadoop_conf/hdfs-site.xml $HADOOP_HOME/etc/hadoop/hdfs-site.xml
cp ./docker-doc/hadoop_conf/mapred-site.xml $HADOOP_HOME/etc/hadoop/mapred-site.xml
cp ./docker-doc/hadoop_conf/yarn-site.xml $HADOOP_HOME/etc/hadoop/yarn-site.xml

# Format HDFS namenode
$HADOOP_HOME/bin/hdfs namenode -format
```

### Madlib

The latest Madlib has issues with XGBoost model inference and loading TensorFlow models. We provide a patch for that and it can be installed through the following commands:

```bash
apt-get install postgresql-plpython3-14 -y
apt-get install postgresql-server-dev-14 -y
git clone -b madlib2-master --single-branch https://github.com/lixi-zhou/madlib.git
cd madlib \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make
export PATH=$PATH:/usr/lib/postgresql/14/bin
```

### PostgresML

PostgresML can be installed from source code or using `apt-get` in Ubuntu.

https://postgresml.org/docs/open-source/pgml/developers/self-hosting/building-from-source

```bash
cargo install cargo-pgrx --version 0.12.9
cargo pgrx install
```

Note: Add the following to `/etc/postgresql/14/main/postgresql.conf`

`shared_preload_libraries = 'pgml'`

If `pgml` is not found, set the absolute path to the `pgml.so`, for example:
`shared_preload_libraries = '/usr/lib/postgresql/14/lib/pgml.so'`.

https://postgresml.org/docs/open-source/pgml/developers/installation#dependencies

### tokenizers-cpp

Due to known issues with incorporating the linked libraries into our CMakeLists file, we are still investigating the problem. Currently, we are using the built shared objects for our program.

```bash
git clone --recursive https://github.com/mlc-ai/tokenizers-cpp.git \
  && cd tokenizers-cpp \
  && git checkout 4fbe996 \
  && cd example \
  && bash build_and_run.sh
```

### Faiss

```bash
git clone https://github.com/facebookresearch/faiss.git
cd faiss \
  && git checkout 4c13a88 \
  && mkdir build && cd build \
  && cmake -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_C_API=ON -DFAISS_ENABLE_GPU=OFF -DFAISS_OPT_LEVEL=generic -DCMAKE_INSTALL_PREFIX=$HOME/faiss-install .. \
  && make -j8 install
```
