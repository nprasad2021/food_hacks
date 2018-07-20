import data_processing, nets
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, Callback
from keras.layers import Average, Dense, GlobalAveragePooling2D
from keras import optimizers
import os
import os.path
import sys
import experiments
from keras import backend as K
import numpy as np
import pandas as pd

DF_OUT = 'accuracy.csv'
ID = int(sys.argv[1])
print(ID)

opt = experiments.create_experiments(ID)

nnet = opt['network']
dataset = opt['dataset']
input_shape = opt['input_shape']	
batch_size=opt['batch_size']
epochs=opt['num_epochs']

archs = nets.all_nets()

def run():

	df_length, num_classes = data_processing.read_and_create('./fruit_image_index.csv', dataset)

	base_model = archs[nnet](input_shape)
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu')(x)
	predictions = Dense(num_classes, activation='softmax')(x)

	model = Model(inputs=base_model.input, outputs=predictions)

	for layer in base_model.layers:
		layer.trainable = False

	lr = data_processing.CustomLRScheduler(data_processing.lr_sched, verbose = 1)
	model.compile(optimizer=optimizers.SGD(), 
	                  loss='categorical_crossentropy', 
	                  metrics=['accuracy'])
	
	training_generator, validation_generator = data_processing.get_gen(dataset)

	filepath = 'models/' + dataset + '/'
	accessory = nnet + '_' + str(batch_size) + '_' + str(input_shape[0])
	best_model_checkpoint = ModelCheckpoint(filepath + accessory + ".hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	
	if not os.path.exists(filepath):
		os.makedirs(filepath)

	tensorboard = TensorBoard(log_dir="logs/" + nnet + '/')
	es = EarlyStopping(min_delta=0.1, patience = 15)

	callbacks_list = [best_model_checkpoint, lr, es]

	nb_training_samples = df_length*.9
	nb_validation_samples = df_length*.1

	print(dataset)
	
	model.fit_generator(
		training_generator,
		steps_per_epoch=int(nb_training_samples/batch_size),
		epochs=epochs,
		validation_data=validation_generator,
		validation_steps=int(nb_validation_samples/batch_size),
		callbacks = callbacks_list,
		verbose=2)
	
	acc = model.evaluate_generator(
		validation_generator,
		steps=nb_validation_samples/batch_size)

	column_names = ['accuracy', 'network', 'dataset', 'batch_size', 'epochs', 'image_size']
	if os.path.exists(DF_OUT):
		df_out = pd.read_csv(DF_OUT)
	else:
		df_out = pd.DataFrame(columns=column_names)

	tmp = [acc, nnet, dataset, batch_size, epochs, input_shape[0]]
	df_out.loc[df_out.shape[0]] = tmp
	df_out.to_csv(DF_OUT,index=False)

	print(acc)

if __name__ == "__main__":
	run()

