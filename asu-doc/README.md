<!-- TOC -->

- [Use Docker to Build Your Development Environment](#use-docker-to-build-your-development-environment)
  - [Build Docker Image](#build-docker-image)
  - [Link to Our Private Velox Repository](#link-to-our-private-velox-repository)
  - [Set-Up Dependencies and Compile](#set-up-dependencies-and-compile)
  - [Develop with Visual Studio Code](#develop-with-visual-studio-code)
- [Run Two-Tower Model Pipeline](#run-two-tower-model-pipeline)

<!-- /TOC -->
## Use Docker to Build Your Development Environment

### Build Docker Image
*It is recommended to use docker to create your development environment, and your dependencies won't get messed up with other stuff.*

Using the following command to build your docker image and start a container
```bash
docker build --tag velox-asu .
docker run --name velox-container -it velox-asu
```

NOTE: if you are using arm chip, you need to use the following command:
```bash
docker build -t velox-arm -f Dockerfile_ARM .
docker run --name velox-container-arm -it velox-arm
```

### Use Docker with GPU

Before creating the container from the image, install the NVIDIA container toolkit by following the commands [Links](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit):
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# restart docker
service docker restart

# use the following commands to grant the GPU access
docker run --name velox-container -it --gpus all velox-asu
```




### Link to Our Private Velox Repository
Because the docker image is not expected to contain any confidential credentials and our GitHub repository is private now. You are required to configure your git configuration by using the following commands:

```bash
git config --global user.name "Your_Name"
git config --global user.email "Your_Email_Address"
eval "$(ssh-agent -s)"
ssh-add path_to_your_key
# add our velox repo as origin using the one of the following command
git remote add origin git@github.com:asu-cactus/velox.git
# or
git remote add origin https://github.com/asu-cactus/velox.git
# switch to our main branch
git switch origin/main
```

### Set-Up Dependencies and Compile

```bash
# run velox setup-ubuntu to install other dependencies
./scripts/setup-ubuntu.sh
# compile Velox in release mode
make release
```

**Note:** If you are using arm chip, you need to set `CPU_TARGET="aarch64"` before setup-ubuntu.sh

Since PyTorch does not provides pre-built files for aarch64, you are required to built locally by using the following command.

```
cd pytorch
python3 setup.py install
```

Before `make release` in Velox folder, you are required to set the following two environment variables:

The following paths can be directly used if you are using docker. Otherwise, you need to accordingly adjust it.
```
Caffe2_DIR=/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Caffe2
Torch_DIR=/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch
```


3rd-party dependecies 

EvaDB with modification to support `ARRAY` in Postgres.
```
git clone https://github.com/lixi-zhou/evadb.git
git checkout array
cd evadb
pip install ./[ray]
```

Madlib
```bash
# install madlib if not configured environment thourgh docker

apt-get install libpq-dev
apt-get install postgresql-plpython3-14
wget https://dist.apache.org/repos/dist/release/madlib/2.1.0/apache-madlib-2.1.0-src.tar.gz
tar -zxvf apache-madlib-2.1.0-src.tar.g
cd apache-madlib-2.1.0-src
mkdir build
cmake ..
# add postgresql 14 to path
export PATH=$PATH:/usr/lib/postgresql/14/bin

# install madlib to postgres
/usr/local/madlib/bin/madpack -s madlib -p postgres install -c postgresdb@localhost
```

### Develop with Visual Studio Code

It is recommended to code with Visual Studio Code. You can use the following script to start a VS Code Tunnel for remote development. If you want to keep the tunnel alive in the background, you can first launch a `tmux` session and then execute the following command.
```
bash ~/start_vscode_tunnel.sh
```

### Start Hadoop

```bash
start-dfs.sh
start-yarn.sh
```

## Run Two-Tower Model Pipeline

If you are going to run the two-tower model pipeline in the docker, there is no need to modify the path to the data parquet file. Otherwise, you are required to modify it correspondingly. After compilation, you will the executable file located at `_build/release/velox/ml_functions/two_tower_model_pipeline_test`. You can run the code with the following command.

` ./two_tower_model_pipeline_test --num_sample=50000 --num_split=10 --batch_size=5000  --num_driver=8`

| Argument   | Description          | Default Value |
|------------|----------------------|---------------|
| num_sample | Number of samples    |           500 |
| num_split  | Number of splits     |             1 |
| batch_size | Batch size           |           500 |
| num_repeat | Number of repeat run |             1 |
| num_driver | Number of drivers    |             1 |

