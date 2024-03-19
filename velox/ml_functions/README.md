# Installation

## xgboost C APIs

[Reference URL](https://xgboost.readthedocs.io/en/stable/tutorials/c_api_tutorial.html#sample-examples-along-with-code-snippet-to-use-c-api-functions)

### Step 1 Prerequisites

Install CMake - Follow the cmake installation documentation for instructions. 

Install Conda - Follow the [conda installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) documentation for instructions

I followed the fillowing steps to create an conda new environment (Don't use old environment that have many installations, which may cause conflicts):

conda create -n xgboost


### Step 2 Install XGBoost C API

```shell
# clone the XGBoost repository & its submodules
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
mkdir build
cd build
# Activate the Conda environment, into which we'll install XGBoost
conda activate xgboost 
# Build the compiled version of XGBoost inside the build folder
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
# install XGBoost in your conda environment (usually under [your home directory]/miniconda3)
make install
```

### Step 3 Build Velox

```shell
cd velox
sudo ./scripts/setup-ubuntu.sh
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
make release
```

## libtorch 

[Reference URL](https://pytorch.org/cppdocs/installing.html)

The libtorch version 2.0.1 with CUDA version 11.8 should be installed here.
```shell
cd ~
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip
unzip libtorch-shared-with-deps-latest.zip
export Torch_DIR=/home/ubuntu/libtorch/share/cmake/Torch
```

