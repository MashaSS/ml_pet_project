import argparse
import os

from model import RegressionResnetModel

data_dir = os.path.join(os.getcwd(), "data")
img_split_folder = os.path.join(os.getcwd(), "img_split")
caucasian_female_test = os.path.join(img_split_folder, "caucasian_female_test.csv")


def main():
    parser = argparse.ArgumentParser(description="Testing script for ResNet")
    parser.add_argument("-w", "--weights", type=str, default="weights.h5", help="Weights of model.")
    args = parser.parse_args()
    resnet_model = RegressionResnetModel()
    path = os.path.join(os.getcwd(), "weights")
    path = os.path.join(path, args.weights)
    resnet_model.load_weights(path)
    resnet_model.evaluate(data_dir, test=caucasian_female_test)


if __name__ == '__main__':
    main()