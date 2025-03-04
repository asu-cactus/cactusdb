# CactusDB Early Release for VLDB Submission

**Note:** This is an early release of the CactusDB codebase for VLDB reviewers. An official release will be available soon.

CactusDB is a UDF-centric database built on top of Meta's high-performance database engine, [Velox](https://github.com/facebookincubator/velox). It enables the co-optimization of SQL queries nested with model inference.

- [CactusDB Early Release for VLDB Submission](#cactusdb-early-release-for-vldb-submission)
  - [Getting Started](#getting-started)
    - [Dependencies](#dependencies)
    - [Set-Up Through Docker](#set-up-through-docker)
    - [Compile CactusDB](#compile-cactusdb)
    - [Data and Models](#data-and-models)
  - [Run Baselines](#run-baselines)
  - [Run Ours](#run-ours)
  - [Development Guide](#development-guide)
  - [FAQ](#faq)



## Getting Started

### Dependencies

The following dependencies are required to run CactusDB and other baselines:

- LibTorch (LibTorch_CUDA)
- PostgreSQL
- EvaDB
- tokenizers-cpp
- Spark
- Eigen
- Catch2
- h5cpp
- cpr
- xgboost
- hadoop
- Madlib
- PostgresML

To manually install the dependencies, please refer to [this file](/INSTALL_DEPENDENCIES.md) for more details.

### Set-Up Through Docker

We recommend using the provided Dockerfile to set up the environment. See the [Docker setup guide](/docker-doc/README.md) for more details. Alternatively, you can manually install the dependencies by following the instructions [here](/INSTALL_DEPENDENCIES.md). CactusDB has been tested and supports Linux (x86) and macOS (Apple Silicon). For Windows users, we recommend using Docker with Windows Subsystem for Linux (WSL).

### Compile CactusDB

After configuring all the dependencies, you can compile CactusDB by following these commands:

```bash
# Run Velox setup-ubuntu to install other dependencies
./scripts/setup-ubuntu.sh
# Compile Velox in release mode
make release
# Install Python libraries for baselines
pip install -r db-ml/baseline/requirements.txt
# Compile CactusDB at the root folder
make release
```

**Note:** If you are using an ARM chip, you need to set `CPU_TARGET="aarch64"` before running setup-ubuntu.sh.

### Data and Models

Run the following commands to download the datasets and models used in our paper. The resources will be extracted into the `resources` directory.

```bash
pip install gdown -U
gdown 1Fpb_jGpkxb7d5ZBC8Uqnq25uEgfOc7yV
unzip resources.zip -d resources
```

## Run Baselines

To run the baseline models, follow these steps:

1. Ensure all dependencies are installed as per the [installation guide](/INSTALL_DEPENDENCIES.md).
2. Download the datasets and models as described above.
3. Navigate to the `db-ml/baseline` directory.
4. Execute the following command to run the baselines:

```bash
python run_baselines.py
```

For more detailed instructions, refer to the [baseline README](/db-ml/baseline/README.md).

<!-- TODO: Add instructions to run baselines -->

## Run Ours

<!-- TODO: Add instructions to run your specific implementation -->

## Development Guide

Please check [this file](/DEVELOP_GUIDE.md) to see the supported ML kernels and how to implement the pipeline within CactusDB.

## FAQ

- **If Spark/Hadoop is not started, run the following commands:**
  ```bash
  service ssh start
  start-all.sh
  ```

- **The compilation is killed and used all the resources:**
  Please try to reduce the number of threads if the compilation takes all the memory and gets killed.
  ```bash
  export NUM_THREADS=4
  ```

<!-- TODO -->

<!-- 
## Examples

Examples of extensibility and integration with different component APIs [can be
found here](velox/examples)

## Documentation

Developer guides detailing many aspects of the library, in addition to the list
of available functions [can be found here.](https://facebookincubator.github.io/velox)

Blog posts are available [here](https://velox-lib.io/blog).

## Getting Started

We provide scripts to help developers setup and install Velox dependencies.

### Get the Velox Source
```
git clone --recursive https://github.com/facebookincubator/velox.git
cd velox
# if you are updating an existing checkout
git submodule sync --recursive
git submodule update --init --recursive
```

### Setting up on macOS

Once you have checked out Velox, on an Intel MacOS machine you can setup and then build like so:

```shell
$ ./scripts/setup-macos.sh 
$ make
```

On an M1 MacOS machine you can build like so:

```shell
$ CPU_TARGET="arm64" ./scripts/setup-macos.sh
$ CPU_TARGET="arm64" make
```

You can also produce intel binaries on an M1, use `CPU_TARGET="sse"` for the above.

### Setting up on aarch64 Linux (Ubuntu 20.04 or later)

On an aarch64 based machine, you can build like so:

```shell
$ CPU_TARGET="aarch64" ./scripts/setup-ubuntu.sh
$ CPU_TARGET="aarch64" make
```

### Setting up on x86_64 Linux (Ubuntu 20.04 or later)

Once you have checked out Velox, you can setup and build like so:

```shell
$ ./scripts/setup-ubuntu.sh 
$ make
```

### Building Velox

Run `make` in the root directory to compile the sources. For development, use
`make debug` to build a non-optimized debug version, or `make release` to build
an optimized version.  Use `make unittest` to build and run tests.

Note that,
* Velox requires a compiler at the minimum GCC 9.0 or Clang 14.0.
* Velox requires the CPU to support instruction sets:
  * bmi
  * bmi2
  * f16c
* Velox tries to use the following (or equivalent) instruction sets where available:
  * On Intel CPUs
    * avx  
    * avx2
    * sse
  * On ARM
    * Neon
    * Neon64

### Building Velox with docker-compose

If you don't want to install the system dependencies required to build Velox,
you can also build and run tests for Velox on a docker container
using [docker-compose](https://docs.docker.com/compose/).
Use the following commands:

```shell
$ docker-compose build ubuntu-cpp
$ docker-compose run --rm ubuntu-cpp
```
If you want to increase or decrease the number of threads used when building Velox
you can override the `NUM_THREADS` environment variable by doing:
```shell
$ docker-compose run -e NUM_THREADS=<NUM_THREADS_TO_USE> --rm ubuntu-cpp
```

## Contributing

Check our [contributing guide](CONTRIBUTING.md) to learn about how to
contribute to the project.

## Community

The main communication channel with the Velox OSS community is through the
[the Velox-OSS Slack workspace](http://velox-oss.slack.com). 
Please reach out to **velox@meta.com** to get access to Velox Slack Channel.


## License

Velox is licensed under the Apache 2.0 License. A copy of the license
[can be found here.](LICENSE) -->
