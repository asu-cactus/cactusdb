# Installation
## xgboost C APIs
[Reference URL](https://github.com/asu-cactus/velox/edit/main/velox/ml_functions/README.md)

## libtorch
[Reference URL](https://github.com/asu-cactus/velox/edit/main/velox/ml_functions/README.md)

# The following installation may needed
## rapidjson
```shell
git clone https://github.com/Tencent/rapidjson.git
cd rapidjson
mkdir build
cd build
cmake ..
make
make install
```
## jsoncpp
```shell
git clone https://github.com/open-source-parsers/jsoncpp.git
cd jsoncpp
mkdir build
cd build
cmake ..
make
make install
```
## Eigen
```shell
sudo apt install libeigen3-dev
cd /usr/local/include
sudo ln -sf eigen3/Eigen Eigen
sudo ln -sf eigen3/unsupported unsupported
```

## openblas
```shell
sudo apt-get install libopenblas-dev
```
## gcc update(may needed)
```shell
# update gcc-9 to gcc-11

sudo apt install build-essential manpages-dev software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update && sudo apt install gcc-11 g++-11

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
                         --slave /usr/bin/gcov gcov /usr/bin/gcov-9 \
                         --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-9 \
                         --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-9 && \
sudo update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-9 90

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-11 \
                         --slave /usr/bin/gcov gcov /usr/bin/gcov-11 \
                         --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-11 \
                         --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-11 && \
sudo update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-11 110

# check gcc version
gcc --version;g++ --version;gcov --version;cpp --version;

# To reconfigure to any previous gcc version...
sudo update-alternatives --config gcc
```
# errors may occur
If occurs error like this:
```shell
/opt/miniconda-for-velox/include/fmt/core.h:1757:7: error: static assertion failed: Cannot format an argument. To make type T formattable provide a formatter<T> specialization: https://fmt.dev/latest/api.html#udt
 1757 |       formattable,
      |       ^~~~~~~~~~~
/opt/miniconda-for-velox/include/fmt/core.h:1757:7: note: ‘formattable’ evaluates to false
```
Check if there is another version of fmt exists or different verson of fmt was installed.
```shell
conda list
```
if fmt in the conda list, uninstall it.
```shell
conda uninstall fmt
```


