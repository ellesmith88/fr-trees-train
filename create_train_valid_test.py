import shutil
from glob import glob
from sklearn.model_selection import train_test_split
from config import TRAIN_TEST_VALID_DIR
import os

def split_data():
    all_files = glob(os.path.join(TRAIN_TEST_VALID_DIR,'*.png'))
    names = []
    for file in all_files:
        names.append(file.split('\\')[-1].split('.')[0])

    
    train_data, rest_of_data = train_test_split(names, train_size=0.8, shuffle=True)   
    validation_data, test_data = train_test_split(rest_of_data, test_size=0.5, shuffle=True)

    for path in [os.path.join(TRAIN_TEST_VALID_DIR,'train'), os.path.join(TRAIN_TEST_VALID_DIR,'test'), os.path.join(TRAIN_TEST_VALID_DIR,'valid')]:
        if not os.path.exists(path):
            # Create a new directory because it does not exist
            os.makedirs(path)

    for i in train_data:
        train_files = glob(os.path.join(TRAIN_TEST_VALID_DIR, f'{i}.*')) + glob(os.path.join(TRAIN_TEST_VALID_DIR, f'{i}_*'))
        for j in train_files:
            name = j.split('\\')[-1]
            shutil.copyfile(j, os.path.join(TRAIN_TEST_VALID_DIR, f'train/{name}'))

    for i in test_data:
        test_files = glob(os.path.join(TRAIN_TEST_VALID_DIR, f'{i}.*')) + glob(os.path.join(TRAIN_TEST_VALID_DIR, f'{i}_*'))
        for j in test_files:
            name = j.split('\\')[-1]
            shutil.copyfile(j, os.path.join(TRAIN_TEST_VALID_DIR, f'test/{name}'))

    for i in validation_data:
        valid_files = glob(os.path.join(TRAIN_TEST_VALID_DIR, f'{i}.*')) + glob(os.path.join(TRAIN_TEST_VALID_DIR, f'{i}_*'))
        for j in valid_files:
            name = j.split('\\')[-1]
            shutil.copyfile(j, os.path.join(TRAIN_TEST_VALID_DIR, f'valid/{name}'))

if __name__ == '__main__':
    split_data()