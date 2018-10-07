import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2

def preprocessing_fn(img):
	img = img/255.0
	img = img - 0.5
	img = img*2
	return img

###################HYPERPARAMETERS#########################

num_classes = 120
batch_size = 128
num_epochs = 50
learning_rate = 1e-3
decay = 1e-6
momentum = 0.9
test_dir = "test/"
saved_model = "result/"
if not os.path.exists(saved_model):
    os.makedirs(saved_model)

###########################################################

graph = keras.model.load_mode("model.h5")
optimizer = keras.optimizers.SGD(lr=learning_rate,decay=decay,momentum=momentum)
graph.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
test_images = os.listdir(test_dir)
results = {}
for image in test_images:
	img = cv2.imread(os.path.join(test_dir,image))
	img = cv2.resize(img, (299, 299),0,0, cv2.INTER_LINEAR)
	img = img.astype(np.float32)
	img = preprocessing_fn(img)
	img = np.array([img])
	classes = model.predict(img, batch_size=1)
	result[image] = np.argmax(classes[0])

'''test_generator = ImageDataGenerator(preprocessing_function=preprocessing_fn)
test_generator = test_generator.flow_from_directory(test_dir,target_size=(299,299),batch_size=batch_size,class_mode='binary')
result = model.predict_generator(test_generator)'''