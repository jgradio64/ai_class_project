from PIL import Image
from statistics import median
import matplotlib.pyplot as plt
import os, shutil

widths = []
heights = []
max_height = None
min_height = None
median_height = None
max_width = None
min_width = None
median_width = None

# image folder is located up one directory
os.chdir("..")
# Get the base directory
root_dir = os.getcwd()
# change to data folder
os.chdir('data')
# Sets the path to the dataset folder
path = os.getcwd()


def get_image_info(path):
	global big_img_count, little_img_count
	print("getting image info");
	#Iterate through all images
	for root, dirs, files in os.walk(path):
		for file in files:
			with Image.open(os.path.join(root, file)) as img:
				#Record width and height
				width, height = img.size
				if(width > 2500):
					continue
				elif(width < 200):
					continue
				else:
					widths.append(width)
					heights.append(height)
	

def calculate_statistics():
	# #Max, min, and mean of the widths of the images
	global max_width, min_width, median_width, max_height, min_height, median_height
	max_width = max(widths)
	min_width = min(widths)
	median_width = median(widths)

	# #Max, min, and mean of the heights of the images
	max_height = max(heights)
	min_height = min(heights)
	median_height = median(heights)


def show_statistics():
	print("minimum width | height \n" + str(min_width) + "\t" + str(min_height))
	print("Maximum width | height \n" + str(max_width) + "\t" + str(max_height))
	print("Median width | height \n" + str(median_width) + "\t" + str(median_height))


def show_histogram():
	#Create histogram
	plt.hist(widths, bins=50, alpha=0.5, label='Width')
	plt.hist(heights, bins=50, alpha=0.5, label='Height')
	plt.legend(loc='upper right')
	plt.title('Distribution of Image Dimensions')
	plt.xlabel('Number of Pixels')
	plt.ylabel('Frequency')
	plt.show()


def resize_images(wd, ht, folder_name):
	print("initialize folder for formatted data")
	path = os.path.join(root_dir, folder_name)
	for root, dirs, files in os.walk(path):
		for file in files:
			delete_img = False
			with Image.open(os.path.join(root, file)) as img:
				#Record width and height
				width, height = img.size
				if(width > 2500):
					delete_img = True
				elif(width < 200):
					delete_img = True
				else:
					#Resize the image
					resized_img = img.resize((wd, ht))
					resized_img.save(os.path.join(root, file))
			if(delete_img):
				os.remove(os.path.join(root, file))


def copy_images(folder_name):
	source_dir = os.path.join(root_dir, "data")
	destination_dir = os.path.join(root_dir, folder_name)
	# Check if the directory already exists. Cannot copy to a location that already exists
	if(os.path.isdir(destination_dir)):
		# Delete the folder
		shutil.rmtree(destination_dir) 
	
	shutil.copytree(source_dir, destination_dir)


get_image_info(path)
calculate_statistics()
show_statistics()
show_histogram()
copy_images("median_data")
resize_images(int(median_width), int(median_height), "median_data")