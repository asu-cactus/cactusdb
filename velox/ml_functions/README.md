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

# Development guide of constructing the final results data

Approach 1

```c++
auto flatArray = BaseVector::create < FlatVector < float >> (
  REAL(), rows.end() * dims[1], context.pool());
float * outputValues = flatArray -> mutableRawValues < float > ();

// Allocate buffers for offsets and sizes
BufferPtr resultOffsets =
  AlignedBuffer::allocate < vector_size_t > (rows.end(), context.pool());
BufferPtr resultSizes =
  AlignedBuffer::allocate < vector_size_t > (rows.end(), context.pool());
auto rawOffsets = resultOffsets -> asMutable < vector_size_t > ();
auto rawSizes = resultSizes -> asMutable < vector_size_t > ();

vector_size_t outputOffset = 0;
rows.applyToSelected([ & ](vector_size_t row) {
  auto mappedIdx = rowMap[row];
  rawOffsets[row] = outputOffset;
  rawSizes[row] = dims[1];
  std::memcpy(
    outputValues + outputOffset,
    resultMatrix.row(mappedIdx).data(),
    dims[1] * sizeof(float));
  outputOffset += dims[1];
});

// Handle nulls (if any input is null, propagate)
BufferPtr newNulls = facebook::velox::functions::addNullsForUnselectedRows(flatArray, rows);

// Construct final ArrayVector and assign to result
auto localResult = std::make_shared < ArrayVector > (
  flatArray -> pool(),
  outputType,
  std::move(newNulls),
  rows.end(),
  std::move(resultOffsets),
  std::move(resultSizes),
  flatArray);

context.moveOrCopyResult(localResult, rows, output);
```

Appraoch 2

```c++
context.ensureWritable(rows, outputType, output);
output -> clearNulls(rows);
auto arrayOutput = output -> as < ArrayVector > ();
auto sizes = arrayOutput -> mutableSizes(rows.end());
auto rawSizes = sizes -> asMutable < int32_t > ();
auto offsets = arrayOutput -> mutableOffsets(rows.end());
auto rawOffsets = offsets -> asMutable < int32_t > ();

// Initialize sizes and offsets to zero.
std::fill(rawSizes, rawSizes + rows.countSelected(), 0);
std::fill(rawOffsets, rawOffsets + rows.countSelected(), 0);

auto elementsOutput = arrayOutput -> elements();
auto elementsPool = context.pool();
auto baseOffset = elementsOutput -> size();
elementsOutput -> resize(baseOffset + rows.countSelected() * dims[1]);

float * outputValues = elementsOutput -> values() -> asMutable < float > ();
vector_size_t outputOffset = baseOffset;
rows.applyToSelected([ & ](vector_size_t row) {
  if (rowMap.find(row) == rowMap.end()) {
    throw std::runtime_error(
      "Mapped index not found for the result matrix.");
  }
  auto mappedIndexInResultMatrix = rowMap[row];
  rawOffsets[row] = outputOffset;
  rawSizes[row] = dims[1];
  std::memcpy(
    outputValues + outputOffset,
    resultMatrix.row(mappedIndexInResultMatrix).data(),
    dims[1] * sizeof(float));

  outputOffset += dims[1];
});
arrayOutput -> setElements(elementsOutput);
```
