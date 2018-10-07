import pandas as pd
import numpy as np
import os
import shutil

def breed_dictionary(filename):
	breed_dict = {}
	classes = open(filename).read().split("\n")
	classes = classes[:-1]
	for i, breed in enumerate(classes):
		breed_dict[breed] = i
	return breed_dict

def load_data(image_dir, labels_file, classes_file):
	labels = np.array(pd.read_csv("labels.csv"))
	#breed_dict = breed_dictionary(classes_file)
	labels_dict = {}
	for label in labels:
		labels_dict[label[0]] = label[1]
	images = os.listdir(image_dir)
	labels = []
	error = []
	for image in images:
		labels.append(labels_dict[image.split(".")[0]])
	images = [os.path.join(image_dir,image) for image in images]
	if not os.path.exists("training_data/"):
		os.mkdir("training_data/")
		os.mkdir("validation_data/")
		for breed in list(breed_dict.keys()):
			os.mkdir("training_data/"+breed)
			os.mkdir("validation_data/"+breed)
	shift_dict = {}
	for breed in list(set(labels)):
		shift_dict[breed] = []
	for i in range(len(labels)):
		shift_dict[labels[i]].append(images[i])
	for breed in list(set(labels)):
		#print(shift_dict[breed])
		num_val = int(0.2*len(shift_dict[breed]))
		for i in range(num_val):
			shutil.copy(shift_dict[breed][i],os.path.join("validation_data",breed))
		for i,img in enumerate(shift_dict[breed]):
			if(i<num_val):
				continue
			else:
				shutil.copy(shift_dict[breed][i],os.path.join("training_data",breed))
		
'''class Batcher(object):

	def __init__(self, image_dir, labels_file="", classes_file, mode="TRAIN"):
		self.mode = mode
		if(self.mode=="TRAIN"):
			self.images, self.labels = load_data(image_dir,labels_file,classes_file)
		else:
			self.images = [os.path.join(image_dir,image) for image in os.listdir(image_dir)]
'''