
def create_experiments(ID):

	opt_tmp = [{'dataset':'Color',
			'batch_size':64,
			'num_epochs':20,
			'network':'vggnet',
			'input_shape':(224,224,3),
			'df_input':'./fruit_image_index.csv'}]
	opt = []

	for network in ['vggnet', 'resnet', 'inceptionv3', 'inception_res']:
		for dataset in ['Color', 'Fruit', 'Description']:
			tmp = dict(opt_tmp[0])

			tmp['dataset'] = dataset
			tmp['network'] = network

			opt.append(tmp)

	return opt[ID]


