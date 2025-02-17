import argparse
import os
from typing import Literal, Union
import train
import predict
from PIL import Image
from pathlib import Path
import pandas as pd
Image.MAX_IMAGE_PIXELS = None
# this directory
this_dir, this_filename = os.path.split(__file__)
rounding_scheme = Literal['default', 'nearest', 'stochastic']
SamplingScheme = Literal['rand', 'stratified', '1vN', 'MvN', 'all']



def main(args):

    if args.train:

        # if results directory does not exist, create it
        if not os.path.exists(args.results_dir):
            os.mkdir(args.results_dir)

        if args.remote_data:
            # check if the data exists
            if not os.path.exists(Path(this_dir) / 'data'):
                train.get_remote_data(args.remote_data)


        train.train_model(
            sampling_strategy=args.sampling_strategy,
            test_size=args.test_size,
            results_dir=args.results_dir
        )


        


    else:
        PATH = args.image_path
        img = Image.open(PATH)

        # strip the path from the image name
        img_name = PATH.split('/')[-1]
        # get the file extension
        img_ext = img_name.split('.')[-1]

        # check if output directory exists
        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)

        image, (tmin, tmax) =  predict.fastACE(img,
                        args.remote_model,
                        args.filter_sigma,
                        args.apply_filter
                        ,args.rounding_scheme)
        
        
        image.save(
            args.output_dir + f'/ace_{tmin}_{tmax}_{img_name}.{img_ext}' if args.output_dir else f'ace_{tmin}_{tmax}_{img_name}.{img_ext}'
        )

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script")
    parser.add_argument("--train", type=bool, default=False, help="Train the model")
    parser.add_argument('--remote_data', type=str, default='"https://drive.google.com/uc?export=download&id=1e_k00AwJmyrEfYVFYS2NODWxAdjEZJ1q"', required=False)
    parser.add_argument('--sampling_strategy', type=str, default="rand", required=False)
    parser.add_argument('--test_size', type=float, default=0.2, required=False)
    parser.add_argument('--results_dir', type=str, default='results', required=False)
    parser.add_argument('--n_jobs', type=int, default=-1, required=False)

    # inference only
    parser.add_argument("--remote_model", 
                        type=str, 
                        default="https://drive.google.com/uc?export=download&id=1r3YaEn930HRGg7ApgZ5xuAShgmc5zBNB", 
                        help="Remote model path")
    parser.add_argument('--rounding_scheme', 
                        type=str, 
                        default='default', help="Options; default (round down), nearest (round to nearest int), stochastic (randly up or down). Applies to the predicted thresholds and the rounding when the image is rescaled.", required=False)
    parser.add_argument("--image_path", type=str, default="image.tiff", help="Path to image")
    parser.add_argument('--output_dir', type=str, default='output', required=False)
    parser.add_argument('--use_filter', type=bool, default=False, required=False)
    parser.add_argument('--filter_sigma', type=float, default=1.0, required=False)
    args = parser.parse_args()
    main(args)
