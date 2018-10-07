import keras
import numpy as np

def create_graph(num_classes):
	base = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
	penultimate = keras.layers.GlobalAveragePooling2D()(base.output)
	logits = keras.layers.Dense(num_classes, activation='softmax')(penultimate)
	model = keras.models.Model(input=base.input, output=logits)
	print(base.input)
	print(logits)

###################HYPERPARAMETERS#########################

num_classes = 120
learning_rate = 1e-3
decay = 1e-6
momentum = 0.9

###########################################################


graph = create_graph(num_classes)