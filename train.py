from keras.optimizers import Adam, SGD, RMSprop

import os
import argparse
import logging
import datetime
import sys

from model import RegressionResnetModel

default_img_size = (224, 224)
log_folder = os.path.join(os.getcwd(), "logs")
img_split_folder = os.path.join(os.getcwd(), "img_split")
caucasian_female_train = os.path.join(img_split_folder, "caucasian_female_train.csv")
caucasian_female_val = os.path.join(img_split_folder, "caucasian_female_val.csv")
caucasian_female_test = os.path.join(img_split_folder, "caucasian_female_test.csv")


def parse_optimizer(optimizer, learning_rate):
    print(learning_rate)
    if optimizer == "SGD":
        return SGD(lr=learning_rate)
    elif optimizer == "RMSprop":
        return RMSprop(lr=learning_rate)
    elif optimizer == "Adam":
        return Adam(lr=learning_rate)
    else:
        raise AttributeError("Wrong optimizer value: {}".format(optimizer))


def parse_log_file(log_file_name, epoches, optimizer, learning_rate, activation, hidden_layers):
    if log_file_name is None:
        date = datetime.datetime.now()
        date = str(date).replace(" ", "_")
        date = str(date).replace(":", "-")
        date = str(date).replace(".", "_")
        log_file_name = "ep={}_opt={}_lr={}_act={}_hid_l={}_{}.txt".format(epoches, optimizer, learning_rate,
                                                                           activation, hidden_layers, date)
        return os.path.join(log_folder, log_file_name)
    else:
        log_file_name = os.path.join(log_folder, log_file_name)
        if os.path.isFile(log_file_name):
            raise FileExistsError("Error! File already exists.")
        else:
            return log_file_name


def parse_data_folder(folder):
    print(os.getcwd())
    fulldir = os.path.join(os.getcwd(), folder)
    print(fulldir)
    if not os.path.isdir(folder) and not os.path.isdir(fulldir):
        raise IsADirectoryError("Error! Directory does not exist.")
    if os.path.isdir(fulldir):
        folder = fulldir
    return folder


def main():
    parser = argparse.ArgumentParser(description="Training script for ResNet")
    parser.add_argument("-r", "--learning_rate", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("-e", "--epoches", type=int, default=15, help="Number of epoches.")
    parser.add_argument("-f", "--log_file", default=None, type=str, help="Learning rate.")
    parser.add_argument("-a", "--activation", default="softmax", choices=["softmax", "tangh", "relu"], type=str, help="Activation rule.")
    parser.add_argument("-o", "--optimizer", type=str, choices=["Adam", "SGD", "RMSprop"], default="Adam",
                        help="Model optimizer. Default: Adam")
    parser.add_argument("-l", "--hidden_layers", type=int, default=256, help="Size of hidden layer before last layer")
    parser.add_argument("-s", "--size", type=str, default="224:224", help="Size of images. Default: 224:224")
    parser.add_argument("-d", "--data_folder", type=str, default="data", help="Path to data folder.")
    args = parser.parse_args()

    data_folder = parse_data_folder(args.data_folder)
    learning_rate = args.learning_rate
    optimizer = parse_optimizer(args.optimizer, learning_rate)
    activation = args.activation
    epoches = args.epoches
    hidden_layers = args.hidden_layers
    size = [int(i) for i in args.size.split(":")]
    log_file_mame = parse_log_file(args.log_file, epoches, args.optimizer, learning_rate, activation, hidden_layers)

    logging.basicConfig(filename=log_file_mame)
    resnet_model = RegressionResnetModel()
    resnet_model.build((size[0], size[1], 3), optimizer, hidden_layers, activation)

    resnet_model.train(train=caucasian_female_train, val=caucasian_female_val, test=caucasian_female_test,
                      path=data_folder, epochs=epoches)


if __name__ == '__main__':
    main()
