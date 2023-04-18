import os
import random
import shutil

os.chdir("..")
os.chdir('data')
data_path = os.getcwd()

test_path = os.path.join(data_path, "test")
train_path = os.path.join(data_path, "train")

def move_category_files(category):
    # We have 5 categories, we want to move half of the cars, minivans, trucks, suvs, and vans.
    source = os.path.join(test_path, category)
    dest = os.path.join(train_path, category)
    files = os.listdir(source)
    no_of_files = len(files) // 2
    for file_name in random.sample(files, no_of_files):
        if os.path.isfile(os.path.join(dest, file_name)):
            print('file already exists')
        shutil.move(os.path.join(source, file_name), dest)

    # print(len(os.listdir(source)))
    # print(len(os.listdir(dest)))


def move_files():
    # This will call the move_category_files for all folders in the test path
    categories = os.listdir(test_path)
    for category in categories:
        move_category_files(category)


move_files()