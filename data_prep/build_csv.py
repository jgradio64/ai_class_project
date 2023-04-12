import os
import csv

# image folder is located up one directory
os.chdir("..")
# Get the base directory
root_dir = os.getcwd()
destination_dir = os.path.join(root_dir, "formatted_data")

field_names= ['Type', 'Image']
vehicles = []

def extract_images(subset_path, type):
    global vehicles
    os.chdir(subset_path)
    vehicle_images = os.listdir()
    for img in vehicle_images:
        vehicles.append({'Type': type, 'Image': img})


def get_images(folder):
    test_path = os.path.join(root_dir, 'data\\' + folder)
    os.chdir(test_path)
    # Get a list of the catagories
    categories = os.listdir()
    for cat in categories:
        path = os.path.join(test_path, cat)
        extract_images(path, cat)


def create_csv(type):
    path = os.path.join(destination_dir, type)
    os.chdir(path)
    check_csv_exists(path)
    with open('Classification.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = field_names)
        writer.writeheader()
        writer.writerows(vehicles)
    # Empty Vechile List
    vehicles.clear()


def check_csv_exists(path):
    my_file = os.path.join(path, 'Classification.csv')
    if os.path.exists(my_file):
        os.remove(my_file)

        # Print the statement once the file is deleted  
        print("The file: {} is deleted!".format(my_file))
    else:
        print("The file: {} does not exist!".format(my_file))


get_images('test')
create_csv('test')
get_images('train')
create_csv('train')