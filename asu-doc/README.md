## Use Docker to Build Your Development Environment

*It is recommended to use docker to create your development environment, and your dependencies won't get messed up with other stuffs.*

Using the following command to build your docker image
```bash
docker build --tag velox-asu .
docker run --name velox-container -it velox-asu
```

Because the docker image is not expected contain any confidentidal credential and our github repository is private now. You are required to configure your git configuration by using the following commands:

```bash
git config --global user.name "Your_Name"
git config --global user.email "Your_Email_Address"
eval "$(ssh-agent -s)"
ssh-add path_to_your_key
# add our velox repo as origin using the one of the following command
git remote add origin git@github.com:asu-cactus/velox.git
git remote add origin https://github.com/asu-cactus/velox.git
# please run the following setup script after you switch to the branch you want to work on
./scripts/setup-ubuntu.sh
```

## Run Two-Tower Model Pipeline

<!-- TODO add details -->
It is required to modify the path to the parquet file. 