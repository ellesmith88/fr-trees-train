import argparse
import math
import subprocess
import os
from glob import glob

from split_image import run_split
from run_template_matching import temp_match
from create_train_valid_test import split_data
from config import map_name, image_dir


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch', action='store_true', help='provide for batch processing for template matching/xml creating')
    parser.add_argument('-n', '--num', help='number of batch jobs to create')

    return parser.parse_args()


def run():

    args = arg_parse()

    # image to split provided in config
    print(f'Splitting image {map_name}')
    len_slices = run_split()

    # do template matching
    print(f'Doing template matching on image {map_name}')

    if args.batch:
        num = int(args.num)
        num_paths = math.ceil(len_slices/num)

        all_patch_paths = glob(f'{image_dir}/*.png')

        current_directory = os.getcwd()
        
        n = 0
        for i in range(num):

            paths = all_patch_paths[n:n+num_paths]
            paths = (',').join(paths)

            output_base = f"{current_directory}/outputs"

            # submit to lotus (jasmin)
            bsub_command = f"sbatch -p high-mem --mem=64000 -t 01:00:00 -o " \
                           f"{output_base}.out -e {output_base}.err {current_directory}" \
                           f"/run_template_matching.py -p {paths}"
            subprocess.call(bsub_command, shell=True)

            print(f"running {bsub_command}")

            n += num_paths

            print('batch processing template matching - split in to train test valid using create_train_valid_test.py after these jobs have finished')
    
    else:
        temp_match()
        print('template matching complete')

        # split into test/train/valid in training data folder
        print ('Splitting in to train, valid, test folders')
        split_data()

    print('Complete')

if __name__ == '__main__':
    run()