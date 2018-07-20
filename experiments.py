
def create_experiments(ID):

	opt_tmp = [{'dataset':'Color',
			'batch_size':64,
			'num_epochs':40,
			'network':'vggnet',
			'input_shape':(224,224,3),
			'df_input':'./fruit_image_index.csv'}]
	opt = []

	for network in ['vggnet', 'resnet', 'inceptionv3', 'inception_res']:
		for dataset in ['Color', 'Fruit', 'Description']:
			for batch_size in [32, 64, 128]:
				for input_shape in [(224,224,3), (256, 256, 3)]:
					tmp = dict(opt_tmp[0])

					tmp['dataset'] = dataset
					tmp['network'] = network
					tmp['batch_size'] = batch_size
					tmp['input_shape'] = input_shape

					opt.append(tmp)

	return opt[ID]


