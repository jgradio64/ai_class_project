import os

def check_files():
    os.chdir("..")
    os.chdir('data')
    data_path = os.getcwd()

    test_path = os.path.join(data_path, "test")
    train_path = os.path.join(data_path, "train")

    categories = os.listdir(test_path)
    for category in categories:
        source = os.path.join(test_path, category)
        files = os.listdir(source)
        print("Num test files for " + category + ": " + str(len(files)))
    
    categories = os.listdir(train_path)
    for category in categories:
        source = os.path.join(train_path, category)
        files = os.listdir(source)
        print("Num test files for " + category + ": " + str(len(files)))


check_files()