import gdown
import os
import argparse
import utils


def download_dataset(dataset, data_dir):
    os.chdir(data_dir)
    if dataset == "movielens" or dataset == "all":
        if not os.path.exists("two_tower_data.zip"):
            gdown.download("https://drive.google.com/uc?id=1iCmXYbCMaSyT5dH5pYQ275-pPjuEPf3D")
        else:
            print("[INFO-Downloader] two_tower_data.zip downloaded")
        os.system("unzip -o two_tower_data.zip")


def main():
    parser = argparse.ArgumentParser(description="Argument parser")

    # Add arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="dataset name, available options: all, movielens",
        required=False
    )
    parser.add_argument(
        "--data_dir",   
        type=str,
        default="data",
        help="Dir to store the data",
        required=False
    )

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = args.data_dir

    utils.mkdir(data_dir)
    print(
        "[INFO-Downloader] Download dataset: {} to Path: {}".format(dataset, data_dir)
    )
    download_dataset(dataset, data_dir)


if __name__ == "__main__":
    main()
