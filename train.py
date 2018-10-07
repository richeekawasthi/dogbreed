import keras
import numpy as np

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
	print("#######\nInput Tensor : "+base.input+"\n#######")
	print("#######\nOutput Tensor : "+logits+"\n#######")

###################HYPERPARAMETERS#########################

num_classes = 120
batch_size = 5
num_epochs = 50
learning_rate = 1e-3
decay = 1e-6
momentum = 0.9
pretrained_weights = ""
training_dir = "training_data/"
validation_dir = "validation_data/"
saved_model = "result/"
if not os.path.exists(saved_model):
    os.makedirs(saved_model)

###########################################################


graph = create_graph(num_classes)
train_generator = keras.preprocessing.Image.ImageDataGenerator(preprocessing_function=preprocessing_fn,
																width_shift_range=0.2,
																height_shift_range=0.2,
																horizontal_flip=True)
train_generator = train_generator.flow_from_directory(training_dir,target_size=(299,299),batch_size=batch_size,class_mode='categorical')
valid_generator = keras.preprocessing.Image.ImageDataGenerator(preprocessing_function=preprocessing_fn)
train_generator = train_generator.flow_from_directory(validation_dir,target_size=(299,299),batch_size=batch_size,class_mode='categorical')

optimizer = keras.optimizers.SGD(lr=learning_rate,decay=decay,momentum=momentum)
model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = keras.callbacks.ModelCheckpoint(saved_model,monitor='loss',verbose=1,save_best_only=True,mode='max')
callbacks_list = [checkpoint]

model.fit_generator(train_generator,epochs=num_epochs,steps_per_epochs=train_steps_per_epoch,validation_data=valid_generator,
			validation_steps=valid_steps_per_epoch,callbacks=callbacks_list)
model.save_weights(saved_model)