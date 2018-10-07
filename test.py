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
	results[image] = classes[0]
f = open("submission.csv","w")
out_str = "id,affenpinscher,afghan_hound,african_hunting_dog,airedale,american_staffordshire_terrier,appenzeller,australian_terrier,basenji,basset,beagle,bedlington_terrier,bernese_mountain_dog,black-and-tan_coonhound,blenheim_spaniel,bloodhound,bluetick,border_collie,border_terrier,borzoi,boston_bull,bouvier_des_flandres,boxer,brabancon_griffon,briard,brittany_spaniel,bull_mastiff,cairn,cardigan,chesapeake_bay_retriever,chihuahua,chow,clumber,cocker_spaniel,collie,curly-coated_retriever,dandie_dinmont,dhole,dingo,doberman,english_foxhound,english_setter,english_springer,entlebucher,eskimo_dog,flat-coated_retriever,french_bulldog,german_shepherd,german_short-haired_pointer,giant_schnauzer,golden_retriever,gordon_setter,great_dane,great_pyrenees,greater_swiss_mountain_dog,groenendael,ibizan_hound,irish_setter,irish_terrier,irish_water_spaniel,irish_wolfhound,italian_greyhound,japanese_spaniel,keeshond,kelpie,kerry_blue_terrier,komondor,kuvasz,labrador_retriever,lakeland_terrier,leonberg,lhasa,malamute,malinois,maltese_dog,mexican_hairless,miniature_pinscher,miniature_poodle,miniature_schnauzer,newfoundland,norfolk_terrier,norwegian_elkhound,norwich_terrier,old_english_sheepdog,otterhound,papillon,pekinese,pembroke,pomeranian,pug,redbone,rhodesian_ridgeback,rottweiler,saint_bernard,saluki,samoyed,schipperke,scotch_terrier,scottish_deerhound,sealyham_terrier,shetland_sheepdog,shih-tzu,siberian_husky,silky_terrier,soft-coated_wheaten_terrier,staffordshire_bullterrier,standard_poodle,standard_schnauzer,sussex_spaniel,tibetan_mastiff,tibetan_terrier,toy_poodle,toy_terrier,vizsla,walker_hound,weimaraner,welsh_springer_spaniel,west_highland_white_terrier,whippet,wire-haired_fox_terrier,yorkshire_terrier\n"
for image in test_images:
	out_str = out_str + image.split(".jpg")[0] + ","
	output = results[image]
	for i in range(120):
		out_str = out_str + str(output[i])
		if(i!=119):
			out_str+=','
		else:
			out_str+='\n'
f.write(out_str)
f.close()

'''test_generator = ImageDataGenerator(preprocessing_function=preprocessing_fn)
test_generator = test_generator.flow_from_directory(test_dir,target_size=(299,299),batch_size=batch_size,class_mode='binary')
result = model.predict_generator(test_generator)'''