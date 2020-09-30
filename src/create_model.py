""" Cats/Dogs Classification Model
author: Jason Brownlee
code from: https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
"""
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from os import makedirs
from os import listdir
from shutil import copyfile


dataset_home = 'finalize_dogs_vs_cats/'


def populate_training_data():
	# create directories
	# create label subdirectories
	labeldirs = ['dogs/', 'cats/']
	for labldir in labeldirs:
		newdir = dataset_home + labldir
		makedirs(newdir, exist_ok=True)
	# copy training dataset images into subdirectories
	src_directory = 'dogs-vs-cats/train/'
	for file in listdir(src_directory):
		src = src_directory + '/' + file
		if file.startswith('cat'):
			dst = dataset_home + 'cats/'  + file
			copyfile(src, dst)
		elif file.startswith('dog'):
			dst = dataset_home + 'dogs/'  + file
			copyfile(src, dst)


# define cnn model
def define_model():
	# load model
	model = VGG16(include_top=False, input_shape=(224, 224, 3))
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(1, activation='sigmoid')(class1)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model


# run the test harness for evaluating a model
def run_test_harness():
# define model
	model = define_model()
	# create data generator
	datagen = ImageDataGenerator(featurewise_center=True)
	# specify imagenet mean values for centering
	datagen.mean = [123.68, 116.779, 103.939]
	# prepare iterator
	train_it = datagen.flow_from_directory(dataset_home,
		class_mode='binary', batch_size=64, target_size=(224, 224))
	# fit model
	model.fit_generator(train_it, steps_per_epoch=len(train_it), epochs=10, verbose=0)
	# save model
	model.save('final_model.h5')


if __name__ == '__main__':
	populate_training_data()
	# entry point, run the test harness
	run_test_harness()