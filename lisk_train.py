import os, sys
sys.path.insert(0, os.path.abspath('../code_venv1/'))
print(sys.path)
from extract_patches_l import *
from keras.utils.np_utils import to_categorical
from training_datagen import *
from train_retina import *
from matplotlib import pyplot as plt
import configparser as cfp
import models_setup as M
import numpy as np
import gc

class Lisk:
    def __init__(self):
        self.train_original = None
        self.train_masks = None
        self.name_experiment = None
        self.N_epochs = None
        self.batch_size = None
        self.model = None

    def load_dataL(self):

        print("Load Liskowski Data\n")
        config = cfp.RawConfigParser()
        config.read('../code_venv1/configuration_l.txt')
        # path to the datasets
        data_path = config.get('data paths', 'path_local')
        # Experiment name
        self.name_experiment = config.get('experiment name', 'name')
        # training settings
        self.N_epochs = int(config.get('training settings', 'N_epochs'))
        N_subimgs = int(config.get('training settings', 'N_subimgs'))
        inside_FOV = config.get('training settings', 'inside_FOV')
        SP = config.get('training settings', 'SP')
        self.batch_size = int(config.get('training settings', 'batch_size'))
        patch_height = int(config.get('data attributes', 'patch_height'))
        patch_width = int(config.get('data attributes', 'patch_width'))

        # create a folder for the results
        result_dir = self.name_experiment
        print("\n1. Create directory for the results (if not already existing)")
        if os.path.exists(result_dir):
            print("Dir already existing")
        elif sys.platform == 'win32':
            os.system('mkdir ' + result_dir)
        else:
            os.system('mkdir -p ' + result_dir)

        DRIVE_train_imgs_original = data_path + 'init_img_train.hdf5'
        DRIVE_train_groundTruth = data_path + 'init_gtruth_train.hdf5'
        patches_imgs_train, patches_gt_train, windows = get_data_training1(
            DRIVE_train_imgs_original,
            DRIVE_train_groundTruth,
            patch_height,
            patch_width,
            N_subimgs,
            inside_FOV,
            SP)  # SP is True or False for structured prediction process

        print('Patches extracted')
        print('\nOf shape: ', patches_imgs_train.shape)

        # =========== Construct and save the Model Architecture =======
        self.train_original = patches_imgs_train.reshape((-1, 27, 27, 3))
        patches_gt_train = patches_gt_train.reshape((-1, 27, 27, 1))

        self.model = M.PLAIN_net(input_size=(27, 27, 3))
        print(self.model.summary())

        print("Check: final output of the network:")
        print(self.model.output_shape)
        # plot(model, to_file='./' + name_experiment + '/' + name_experiment + '_model.png')  # check how the model
        # looks like
        json_string = self.model.to_json()
        open('./' + self.name_experiment + '/' + self.name_experiment + '_architecture.json', 'w').write(json_string)
        print('Load Liskowski\n')

        patches_gt_train = to_categorical(patches_gt_train,
                                          num_classes=2)  # alternative for the masks_net function call
        patches_gt_train = patches_gt_train.reshape((-1, 27, 27, 2))
        print(patches_gt_train.shape)
        self.train_masks = patches_gt_train[:, 27 - 13 - 1, 27 - 13 - 1, :]

        del patches_gt_train

        # # Preprocess in batches
        # print("normalization\n")
        # from img_datagen import NormImageDataset
        # from augmentation_l import dataset_whitening
        # for step in range(0, patches_imgs_train.shape[0], self.batch_size):
        #     arr = np.asarray(NormImageDataset(patches_imgs_train[step:step + self.batch_size,:,:,:], dataset_normalized))
        #     print(arr.shape, arr[0])
        #     patches_imgs_train[step:step + self.batch_size,:,:,:] = arr

        # print("zca\n")
        # train_original, zca_matrix = np.asarray(dataset_whitening(patches_imgs_train))
        # import pickle as p
        # with open("zca.pickle", "wb") as f:
        #     p.dump(zca_matrix, f)

        del patches_imgs_train



    def Liskowski_train(self, x_train, y_train, x_val, y_val):
        train_data = Custom_Generator(x_train, y_train, self.batch_size, augmentation=True, shuffle=True)
        val_data = Custom_Generator(x_val, y_val, self.batch_size, augmentation=False, shuffle=False)

        filepath = "weights_best.hdf5"
        # ModelCheckpoint only saves the weight only if loss decreases
        checkpoint = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', verbose=1, mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, epsilon=1e-7,
                                           mode='min')

        history = self.model.fit(
            train_data,
            epochs=self.N_epochs,
            # class_weight = weights,
            validation_data=val_data,
            callbacks=[reduce_lr_loss, checkpoint]
        )
        pred_patches = self.model.evaluate(val_data, verbose=1)
        del train_data, val_data, x_train, y_train, x_val, y_val

        # Only saves last epoch's weights where the performance (aka loss) could not be optimal
        self.model.save_weights('./' + self.name_experiment + '/' + self.name_experiment + '_last_weights.h5',
                                overwrite=True)
        import json
        import pandas as pd

        with open('history.json', 'w') as f:
            json.dump(str(history.history), f)

        history_df = pd.DataFrame(history.history)
        print(history_df)
        plt.plot(history_df['loss'])
        plt.plot(history_df['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        plt.savefig('./outputs/loss_100k20e64b.png')

        plt.figure()
        plt.plot(history_df['accuracy'])
        plt.plot(history_df['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        plt.savefig('./outputs/accuracy_100k20e64b.png')
        return pred_patches
