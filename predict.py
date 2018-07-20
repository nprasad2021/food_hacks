from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf


color = {'black': 0, 'bright-green': 1, 'green': 2, 'green-purple': 3, 'light-green': 4, 
		'purple-black': 5, 'yellow': 6, 'yellow-brown': 7, 'yellow-green': 8, 'yellow-red': 9}
fruit = {'Avocado': 0, 'Banana': 1, 'Papaya': 2}
quality = {'not_ripe': 0, 'ripe': 1, 'spoilt': 2}
datasets = {'color':color, 'fruit':fruit, 'quality':quality}
models = {'quality':'models/Description/inceptionv3_64_224.hdf5', 'color':'models/Color/inceptionv3_64_224.hdf5', 'fruit':'models/Color/resnet_64_224.hdf5'}

def model_predict(img_path, model_path, mapping):
	model = load_model(model_path)
	model._make_predict_function()
	print('Model loaded. Start serving...')

	img = image.load_img(img_path, target_size=(224, 224))

	# Preprocessing the image
	x = image.img_to_array(img)
	x = ((x/255.) - 0.5) * 2.

	x = np.expand_dims(x, axis=0)
	preds = model.predict(x)
	return mapping[preds.argmax(axis=-1)[0]]

def create_datasets(dataset_to_inv):
	new_dict = {}
	for key in dataset_to_inv:
		new_dict[key] = invert_dict(dataset_to_inv[key])
	return new_dict

def invert_dict(sample_dict):
	inv_map = {v: k for k, v in sample_dict.items()}
	return inv_map

def main_model_predict(img_path):
	dataset_to_index = create_datasets(datasets)
	predictions = {}
	for key in dataset_to_index:
		tf.reset_default_graph()
		print(key)
		tmp = model_predict(img_path, models[key], dataset_to_index[key])
		predictions[key] = tmp
		print(tmp)
	return predictions


print(main_model_predict('fruit_images/black_Avocado_spoilt_0.jpg'))
