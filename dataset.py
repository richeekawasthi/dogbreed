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
		labels.append(labels_dict[image.split(".")[0]])
	images = [os.path.join(image_dir,image) for image in images]
	return images, labels

'''class Batcher(object):

	def __init__(self, image_dir, labels_file="", classes_file, mode="TRAIN"):
		self.mode = mode
		if(self.mode=="TRAIN"):
			self.images, self.labels = load_data(image_dir,labels_file,classes_file)
		else:
			self.images = [os.path.join(image_dir,image) for image in os.listdir(image_dir)]
'''