- [Use Docker to Build Your Development Environment](#use-docker-to-build-your-development-environment)
  - [Build Docker Image](#build-docker-image)
  - [Use Docker with GPU](#use-docker-with-gpu)
  - [Link to Our Private Velox Repository](#link-to-our-private-velox-repository)
  - [Set-Up Dependencies and Compile](#set-up-dependencies-and-compile)
  - [Develop with Visual Studio Code](#develop-with-visual-studio-code)
- [Pull our Pre-built Docker Development Images](#pull-our-pre-built-docker-development-images)
  - [Available Images](#available-images)
  - [Getting Started](#getting-started)


You can directly use our pre-built docker images by following the instructions [here](#pull-our-pre-built-docker-development-images).

## Use Docker to Build Your Development Environment

### Build Docker Image

*It is recommended to use Docker to create your development environment. This ensures that your dependencies are isolated and won't interfere with other projects or system configurations.*

Use the following commands to build your Docker image and start a container:

```bash
docker build --tag cactusdb-docker-amd64 .
docker run --name cactusdb-container -it cactusdb-docker-amd64
```

**Note:** If you are using an ARM-based chip, you need to use an ARM-compatible Docker image. Use the following commands:

```bash
# Build the Docker image for ARM architecture
docker build -t cactusdb-docker-arm -f Dockerfile_ARM .
# Run a container from the ARM-based image
docker run --name cactusdb-container-arm -it cactusdb-docker-arm
```

### Use Docker with GPU

Before creating the container from the image, install the NVIDIA container toolkit by following the commands [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
# to configure the docker in a rootless mode, please refer to the following link: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
service docker restart
```

If the NVIDIA container toolkit is successfully installed, you should see the NVIDIA-SMI output from the following command:
```bash
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
sudo docker run --rm --runtime=nvidia --gpus all -it ubuntu
sudo docker run --rm --gpus all -it nvidia/cuda:12.6.3-devel-ubuntu22.04 nvcc --version
```

```bash
# Build image with CUDA
docker build -t cactusdb-docker-cuda -f Dockerfile_CUDA .

# Use the following commands to grant GPU access
docker run --name cactusdb-container-cuda --runtime=nvidia --gpus all -it cactusdb-docker-cuda
# or
docker run --name cactusdb-container-cuda --gpus all -it velox-cuda
```

### Link to Our Private Velox Repository

Since the Docker image is not expected to contain any confidential credentials and our GitHub repository is private, you need to configure your Git settings using the following commands:

```bash
git config --global user.name "Your_Name"
git config --global user.email "Your_Email_Address"
eval "$(ssh-agent -s)"
ssh-add path_to_your_key
# Add our Velox repo as origin using one of the following commands
git remote add origin git@github.com:asu-cactus/cactusdb.git
# or
git remote add origin https://github.com/asu-cactus/cactusdb.git 
# Switch to our main branch
git switch origin/main
```

**Note:** For the VLDB early release, replace the above link with the following URL: `https://gitfront.io/r/lixizhou/MocouNoVr2h1/CactusDB-Early-Release-for-VLDB.git`

### Set-Up Dependencies and Compile

After successfully launching the container, you need to install Velox's dependencies and Python libraries.

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

Before running `make release` in the Velox folder, you are required to set the following two environment variables:

**Note:** If there is any missing library, please refer to [this file](../INSTALL_DEPENDENCIES.md) to install it manually within the Docker container.

### Develop with Visual Studio Code

It is recommended to code with Visual Studio Code. You can use the following script to start a VS Code Tunnel for remote development. If you want to keep the tunnel alive in the background, you can first launch a `tmux` session and then execute the following command:

```bash
bash ~/start_vscode_tunnel.sh
```

## Pull our Pre-built Docker Development Images

We provide pre-built Docker images for **x86_64**, **ARM64**, and **CUDA-enabled** architectures, all based on **Ubuntu 22.04**. These images are available on [Docker Hub](https://hub.docker.com/repository/docker/cactusdb/cactusdb/).

### Available Images

| Architecture | Tag              | Description                 |
| ------------ | ---------------- | --------------------------- |
| x86_64       | `cactusdb-amd64` | Standard image for x64 CPUs |
| ARM64        | `cactusdb-arm64` | For ARM-based platforms     |
| CUDA         | `cactusdb-cuda`  | With NVIDIA CUDA support    |

### Getting Started

Run a container using the appropriate image for your platform:

```bash
# Run the x64-based image
docker run --name cactusdb-container-amd64 -it cactusdb/cactusdb:cactusdb-amd64

# Run the ARM-based image
docker run --name cactusdb-container-arm64 -it cactusdb/cactusdb:cactusdb-arm64

# Run the x64-based image with CUDA enabled
docker run --name cactusdb-container-cuda -it cactusdb/cactusdb:cactusdb-cuda
```