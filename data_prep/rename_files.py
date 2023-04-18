import os

os.chdir("..")
os.chdir('data')
data_path = os.getcwd()

test_path = os.path.join(data_path, "test")

def rename_files():
    categories = os.listdir(test_path)
    for category in categories:
        source = os.path.join(test_path, category)
        files = os.listdir(source)
        for file in files:
            fname = file.replace('0', '1', 1)
            os.rename(os.path.join(source, file), os.path.join(source, fname))


rename_files()