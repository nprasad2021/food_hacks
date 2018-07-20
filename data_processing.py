from keras.preprocessing import image
from keras.utils import multi_gpu_model
from keras.layers import Input
from glob import glob
import os
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, Callback
from keras.models import Model
import numpy as np
import shutil
import random
import pandas as pd
from scipy.misc import imread


# training generator configuration
def create_dataset(df, class_name, recreate=True):
    if recreate:
        if os.path.exists('./data/' + class_name + '/'):
            shutil.rmtree('./data/' + class_name + '/')

    category_tmp = df[class_name].tolist()
    category = [x.replace(' ', '_') for x in category_tmp]
    image_path = df['image_path'].tolist()

    c = list(zip(category, image_path))
    random.shuffle(c)
    category_shuffle, image_path_shuffle = zip(*c)
    num_classes = len(set(category_shuffle))

    num_images = len(category_shuffle)
    i = 0

    for cat, path in zip(category_shuffle, image_path_shuffle):
        i += 1
        try:
            a = imread(path).shape
        except:
            print('ERROR:', path, 'NOT READ')
            continue

        if i >= int(num_images*.9):
            subdir = 'validation'
        else:
            subdir = 'train'
        

        class_path = './data/' + class_name + '/' + subdir + '/' + cat + '/' 

        if not os.path.exists(class_path):
            os.makedirs(class_path)

        class_image = class_path +  path.split('/')[-1]
        
        if os.path.exists(class_image):
            continue

        shutil.copy(path, class_image)
        

    return len(df), num_classes

def read_and_create(df_input_path, class_name):
    print('workding directory', os.getcwd())
    df = pd.read_csv(df_input_path)
    df_length, num_classes = create_dataset(df, class_name)
    return df_length, num_classes


def get_gen(dataset, batch_size=64, epochs=100, img_dim = (224,224), input_shape=(224,224,3)):

    print(os.getcwd())

    # dimensions of our images.
    img_width, img_height = img_dim

    input_tensor = Input(shape=input_shape)
    
    training_data_dir = './data/' + dataset + '/train/'
    training_datagen = FixedImageDataGenerator(
        rescale=1./255,
        featurewise_center=True,
        featurewise_std_normalization=True)


    training_generator = training_datagen.flow_from_directory(
        training_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    # validation generator configuration
    validation_data_dir = './data/' + dataset + '/validation/'
    validation_datagen = FixedImageDataGenerator(
        rescale=1./255,
        featurewise_center=True,
        featurewise_std_normalization=True)

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    return training_generator, validation_generator

class CustomLRScheduler(Callback):

    def __init__(self, schedule, verbose = True):
        super(CustomLRScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        last_lr = K.get_value(self.model.optimizer.lr)
        lr = self.schedule(last_lr)

        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)

def lr_sched(last_lr):
    return 0.99*last_lr

class FixedImageDataGenerator(image.ImageDataGenerator):
    def standardize(self, x):
        if self.featurewise_center:
            x = ((x/255.) - 0.5) * 2.
        return x



    