import pandas as pd
import numpy as np
import os

def breed_dictionary(filename):
	breed_dict = {}
	classes = open(filename).read().split("\n")
	classes = classes[:-1]
	for i, breed in enumerate(classes):
		breed_dict[breed] = i
	return breed_dict

def load_data(image_dir, labels_file, classes_file):
	labels = np.array(pd.read_csv("labels.csv"))
	breed_dict = breed_dictionary(classes_file)
	labels_dict = {}
	for label in labels:
		labels_dict[label[0]] = breed_dict[label[1]]
	images = os.listdir(image_dir)
	labels = []
	error = []
	for image in images:
		try:
			labels.append(labels_dict[image.split(".")[0]])
		except:
			error.append(image)
	return images, labels, error

