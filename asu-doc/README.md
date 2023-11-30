## Use Docker to Build Your Development Environment

### Build Docker Image
*It is recommended to use docker to create your development environment, and your dependencies won't get messed up with other stuffs.*

Using the following command to build your docker image and start a container
```bash
docker build --tag velox-asu .
docker run --name velox-container -it velox-asu
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
# run velox setup-ubuntu to install other dependencies
./scripts/setup-ubuntu.sh
```

### Develop with Visual Studio Code

It is recommended to code with Visual Studio Code. You can use the following script to start a VS Code Tunnel for remote development. If you want to keep the tunnel alive in the background, you can first launch a `tmux` session and then execute the following command.
```
bash ~/start_vscode_tunnel.sh
```

## Run Two-Tower Model Pipeline

<!-- TODO add details -->
If you are going to run the two-tower model pipeline in the docker, there is no need to modify the path to the data parquet file. Otherwise, you are required to modify it correspondingly. 