from PIL import Image
import os
import glob
import argparse
import logging
import datetime

ERROR = 1
SUCCESS = 0

date = datetime.datetime.now()
date = str(date).replace(" ", "_")
date = str(date).replace(":", "-")
date = str(date).replace(".", "_")
log_file_name = "logs/data-resize-{}.log".format(date)
logging.basicConfig(level=logging.DEBUG, filename=log_file_name)


def img_processing(imgdir, outdir, new_size):
    img_name = os.path.basename(imgdir)
    exists = os.path.isfile(imgdir)
    if exists:
        logging.info("Processing image {} ...".format(imgdir))
        img = Image.open(imgdir)
        oldsize = img.size
        new_img = Image.new('RGB', (oldsize[0], oldsize[1]), (255, 255, 255))
        new_img.paste(img)
        new_img.thumbnail((new_size[0], new_size[1]), Image.ANTIALIAS)
        new_img.save(os.path.join(outdir, img_name))
        logging.info("Processing done.")
        return SUCCESS
    else:
        logging.error("The file {} doesn't exist.".format(imgdir))
        return ERROR


def data_resizing(inputdir, outdir, size):
    fulldir = os.path.join(os.getcwd(), inputdir)
    if os.path.isdir(inputdir) or os.path.isdir(fulldir):
        if os.path.isdir(fulldir):
            inputdir = fulldir
        imgs = glob.glob(os.path.join(inputdir, "*.jpg"))
        if outdir is None:
            outdir = os.path.join(os.getcwd(), "data")
        else:
            outdir = os.path.join(outdir, "data")
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        else:
            logging.error("Cannot create directory {} for resized images! Directory exists.".format(outdir))
            return ERROR
        for img_dir in imgs:
            ret = img_processing(img_dir, outdir, size)
            if ret != SUCCESS:
                break
        return ret
    else:
        logging.error("Error! '{}' or {} is not a directory.".format(inputdir, fulldir))
        print("Error! '{}' or {} is not a directory.".format(inputdir, fulldir))
        return ERROR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, help="Data directory.")
    parser.add_argument("-o", "--output_dir", type=str, help="Full path to new directory with resized images. Default:"
                                                             "current directory")
    parser.add_argument("-s", "--size", type=str, default="224:224", help="Size of images. Default: 224:224")
    args = parser.parse_args()

    logging.info(args.input_dir)
    size = [int(i) for i in args.size.split(":")]
    ret = data_resizing(args.input_dir, args.output_dir, size)
    if ret != SUCCESS:
        print("Error! Something went wrong! Please check log file {} for extra information.".format(log_file_name))
