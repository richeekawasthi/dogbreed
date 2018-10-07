import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

def preprocessing_fn(img):
	img = img/255.0
	img = img - 0.5
	img = img*2
	return img

def create_graph(num_classes):
	base = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
	penultimate = keras.layers.GlobalAveragePooling2D()(base.output)
	logits = keras.layers.Dense(num_classes, activation='softmax')(penultimate)
	model = keras.models.Model(input=base.input, output=logits)
	print("###################################################")
	print(base.input)
	print("###################################################")
	print("###################################################")
	print(logits)
	print("###################################################")
	return model

###################HYPERPARAMETERS#########################

num_classes = 120
batch_size = 128
num_epochs = 50
learning_rate = 1e-3
decay = 1e-6
momentum = 0.9
train_steps_per_epoch = 8221 // batch_size
valid_steps_per_epoch = 2001 // batch_size
training_dir = "training_data/"
validation_dir = "validation_data/"
saved_model = "result/weights.h5"
if not os.path.exists(saved_model):
    os.makedirs(saved_model)

###########################################################


graph = create_graph(num_classes)
tr_generator = ImageDataGenerator(preprocessing_function=preprocessing_fn,
																width_shift_range=0.2,
																height_shift_range=0.2,
																horizontal_flip=True)
train_generator = tr_generator.flow_from_directory(training_dir,target_size=(299,299),batch_size=batch_size,class_mode='binary')
val_generator = ImageDataGenerator(preprocessing_function=preprocessing_fn)
valid_generator = val_generator.flow_from_directory(validation_dir,target_size=(299,299),batch_size=batch_size,class_mode='binary')

optimizer = keras.optimizers.SGD(lr=learning_rate,decay=decay,momentum=momentum)
graph.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#checkpoint = keras.callbacks.ModelCheckpoint(saved_model,monitor='loss',verbose=1,save_best_only=True,mode='max')
#callbacks_list = [checkpoint]

graph.fit_generator(train_generator,epochs=num_epochs,steps_per_epoch=train_steps_per_epoch,validation_data=valid_generator,
			validation_steps=valid_steps_per_epoch)
graph.save_weights(saved_model)