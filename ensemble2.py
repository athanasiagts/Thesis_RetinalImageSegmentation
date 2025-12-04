## Ensemble Stacking Learning for Heterogeneous Deep Learning Models
## Use a Meta-Learner Model that takes as inputs the outputs of the 2 "weak-learners" and
## will learn and return the final predictions.

# 1. Split train data into 2 folds (k-fold cross-training)
# 2. Choose the "weak-learners" and fit them to 1st train fold
# 3. For each of the "weak-learners" make predictions on the 2nd validation fold
# 4. Fit the Meta-Model on the 2nd fold using  the previous weak preidictions as inputs

# compare standalone models for binary classification
import numpy as np
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot
import os, sys
sys.path.insert(0, os.path.abspath('../code_venv1/'))
print(sys.path)

from sinenet_torch_BN import *
import models_setup as M
from extract_patches import *
from extract_patches_l import *
from training_datagen import *
from prediction_sn import *
from train_retina import *
from test_retina_keras import *

# # Each model will be evaluated using repeated k-fold cross-validation.
# # evaluate a given model using cross-validation -> 3 repeats of stratified 5-fold cross-validation
# def evaluate_model(model, X, y):
# 	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42, shuffle=True)
# 	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# 	return scores
#
# # get the models to evaluate
# models = get_models()
# # evaluate the models and store results
# results, names = list(), list()
# for name, model in models.items():
# 	scores = evaluate_model(model, X, y)
# 	results.append(scores)
# 	names.append(name)
# 	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# # plot model performance for comparison
# pyplot.boxplot(results, labels=names, showmeans=True)
# pyplot.show()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

####### ----------------- 2nd way of stacking --------------------------
class Ensemble:
    def __init__(self):
        self.patches_imgs_train = None
        self.patches_gt_train = None
        self.test_original = None
        self.test_gt = None
        self.test_FOVs = None
        self.patch_width = None
        self.patch_height = None
        self.batch_size = None
        self.df_all = None
        self.model = None
        self.optimizer = None
        self.val_loader = None
        self.train_loader = None
        self.criterion = None
        self.train_original = None
        self.x_test = None
        self.train_masks = None
        self.y_test = None
        self.train_data = None
        self.val_data = None
        self.check_path_cont = None
        self.N_epochs = None
        self.N_subimgs = None
        self.inside_FOV = None
        self.SP = None
        self.name_experiment = None
        self.patches_imgs_test = None
        self.images2test = None
        self.check_path_best = 'SineNet_layer_%d_filter_%d_best_test_Loss.pt7'% (15,3)
        self.check_path_cont = 'SineNet_layer_%d_filter_%d_continue_test_Loss.pt7'% (15,3)

        self.k = 2

    def loadall(self):
        config = cfp.RawConfigParser()
        config.read('configuration.txt')
        self.data_path = config.get('data paths', 'path_local')
        path_drive = '../../../../DRIVE/DRIVE'  
        self.name_experiment = config.get('experiment name', 'name')
        # training settings
        N_epochs = int(config.get('training settings', 'N_epochs'))

        # LOAD DATA
        print('extracting patches')
        DRIVE_train_imgs_original = self.data_path + 'init_img_train.hdf5'
        DRIVE_train_groundTruth = self.data_path + 'init_gtruth_train.hdf5'
        self.patch_height = int(config.get('data attributes', 'patch_height'))
        self.patch_width = int(config.get('data attributes', 'patch_width'))
        inside_FOV = True

        image_folder = path_drive + '/training/images/'
        images = [img for img in os.listdir(image_folder) if img.endswith(".tif")]
        images = sorted(images, key=natural_sort_key)  # sorted

        gt_folder = path_drive + '/training/1st_manual/'
        gts = [img for img in os.listdir(gt_folder) if img.endswith(".gif")]
        gts = sorted(gts, key=natural_sort_key)  # sorted

        mask_folder = path_drive + '/training/mask/'
        masks = [img for img in os.listdir(mask_folder) if img.endswith(".gif")]
        masks = sorted(masks, key=natural_sort_key)  # sorted

        train_original, train_masks, train_FOVs = np.zeros(shape=(len(images), 584, 565), dtype=np.uint8), np.zeros(
            shape=(len(gts), 584, 565), dtype=np.uint8), np.zeros(shape=(len(images), 584, 565), dtype=np.uint8)
        for i in range(len(images)):
            train_original[i] = cv2.imread(os.path.join(image_folder, images[i]), 0)
            gif_reader = imageio.get_reader(os.path.join(gt_folder, gts[i]))
            gif_length = gif_reader.get_length()
            frame_index = 0
            if frame_index < gif_length:
                train_masks[i] = gif_reader.get_data(frame_index)
            gif_reader = imageio.get_reader(os.path.join(mask_folder, masks[i]))
            gif_length = gif_reader.get_length()
            frame_index = 0
            if frame_index < gif_length:
                train_FOVs[i] = gif_reader.get_data(frame_index)

        ###### ----------------- TEST ----------------
        image_folder = path_drive + '/test/images/'
        images = [img for img in os.listdir(image_folder) if img.endswith(".tif")]
        images = sorted(images, key=natural_sort_key)  # sorted

        gt_folder = path_drive + '/test/1st_manual/'
        gts = [img for img in os.listdir(gt_folder) if img.endswith(".gif")]
        gts = sorted(gts, key=natural_sort_key)  # sorted

        mask_folder = path_drive + '/test/mask/'
        masks = [img for img in os.listdir(mask_folder) if img.endswith(".gif")]
        masks = sorted(masks, key=natural_sort_key)  # sorted

        test_original, test_masks, test_FOVs = np.zeros(shape=(len(images), 584, 565), dtype=np.uint8), np.zeros(
            shape=(len(gts), 584, 565), dtype=np.uint8), np.zeros(shape=(len(masks), 584, 565), dtype=np.uint8)
        for i in range(len(images)):
            test_original[i] = cv2.imread(os.path.join(image_folder, images[i]), 0)
            gif_reader = imageio.get_reader(os.path.join(gt_folder, gts[i]))
            gif_length = gif_reader.get_length()
            frame_index = 0
            if frame_index < gif_length:
                test_masks[i] = gif_reader.get_data(frame_index)
            gif_reader = imageio.get_reader(os.path.join(mask_folder, masks[i]))
            gif_length = gif_reader.get_length()
            frame_index = 0
            if frame_index < gif_length:
                test_FOVs[i] = gif_reader.get_data(frame_index)

        self.name_experiment = config.get('experiment name', 'name')
        path_experiment = './' + self.name_experiment + '/'
        # N full images to be predicted
        Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))

        test_original = test_original[:, 10:575, :]
        test_masks = test_masks[:, 10:575, :]
        test_FOVs = test_FOVs[:, 10:575, :]

        print(train_original.dtype, train_masks.dtype, train_original.max(), train_masks.max())

        del gts, images, gif_reader
        gc.collect()
        return train_original, train_masks, train_FOVs, test_original, test_masks, test_FOVs, inside_FOV


    def load_dataL(self):
        ## os.system('python train_retina.py')
        print("Load Liskowski Data\n")

        config = cfp.RawConfigParser()
        config.read('../code_venv1/configuration_l.txt')
        # path to the datasets
        self.data_path = config.get('data paths', 'path_local')
        # Experiment name
        self.name_experiment = config.get('experiment name', 'name')
        # training settings
        self.N_epochs = int(config.get('training settings', 'N_epochs'))
        self.N_subimgs = int(config.get('training settings', 'N_subimgs'))
        self.inside_FOV = config.get('training settings', 'inside_FOV')
        self.SP = config.get('training settings', 'SP')
        self.batch_size = int(config.get('training settings', 'batch_size'))
        self.patch_height = int(config.get('data attributes', 'patch_height'))
        self.patch_width = int(config.get('data attributes', 'patch_width'))

        # create a folder for the results
        result_dir = self.name_experiment
        print("\n1. Create directory for the results (if not already existing)")
        if os.path.exists(result_dir):
            print("Dir already existing")
        elif sys.platform == 'win32':
            os.system('mkdir ' + result_dir)
        else:
            os.system('mkdir -p ' + result_dir)
        # ---------------------------------------  Load Data : Input Data should be numpy ------------------------------------
        DRIVE_train_imgs_original = self.data_path + 'init_img_train.hdf5'
        DRIVE_train_groundTruth = self.data_path + 'init_gtruth_train.hdf5'
        patches_imgs_train, patches_gt_train, windows = get_data_training1(
            DRIVE_train_imgs_original,
            DRIVE_train_groundTruth,
            self.patch_height,
            self.patch_width,
            self.N_subimgs,
            self.inside_FOV,
            self.SP)  # SP is True or False for structured prediction process

        print('Patches extracted')
        print('\nOf shape: ', patches_imgs_train.shape)

        # =========== Construct and save the Model Architecture =======
        self.train_original = patches_imgs_train.reshape((-1, 27, 27, 3))
        patches_gt_train = patches_gt_train.reshape((-1, 27, 27, 1))

        self.model = M.PLAIN_net(input_size=(27, 27, 3))
        print(self.model.summary())

        print("Check: final output of the network:")
        print(self.model.output_shape)
        # plot(model, to_file='./' + name_experiment + '/' + name_experiment + '_model.png')  # check how the model looks like
        json_string = self.model.to_json()
        open('./' + self.name_experiment + '/' + self.name_experiment + '_architecture.json', 'w').write(json_string)
        print('Load Liskowski\n')

        patches_gt_train = to_categorical(patches_gt_train, num_classes=2)  # alternative for the masks_net function call
        patches_gt_train = patches_gt_train.reshape((-1, 27, 27, 2))
        print(patches_gt_train.shape)
        self.train_masks = patches_gt_train[:, 27 - 13 - 1, 27 - 13 - 1, :]
        
        del patches_imgs_train, patches_gt_train

        # Preprocess in batches
        print("normalization\n")
        from img_datagen import NormImageDataset
        from augmentation_l import dataset_whitening
        # for step in range(0, patches_imgs_train.shape[0], self.batch_size):
        #     patches_imgs_train[step:step + self.batch_size] = np.asarray(
        #         NormImageDataset(patches_imgs_train[step:step + self.batch_size], dataset_normalized))

        print("zca\n")
        # self.train_original, zca_matrix = np.asarray(dataset_whitening(patches_imgs_train))
        # import pickle as p
        # with open("zca.pickle", "wb") as f:
        #     p.dump(zca_matrix, f)

    ######--------------------- LOAD TEST DATA -------------------
    def Liskowski_predict(self):
        
        config = cfp.RawConfigParser()
        config.read('../code_venv1/configuration_l.txt')
        # path to the datasets
        self.data_path = config.get('data paths', 'path_local')
        DRIVE_test_imgs_original = self.data_path + config.get('data paths', 'test_imgs_original')
        self.test_original = load_hdf5(DRIVE_test_imgs_original)

        # the ground truth masks
        DRIVE_test_groudTruth = self.data_path + config.get('data paths', 'test_groundTruth')
        self.test_gt = load_hdf5(DRIVE_test_groudTruth)          # self.test_gt for all images masks

        # the border masks provided by the DRIVE
        DRIVE_test_border_masks = self.data_path + config.get('data paths', 'test_border_masks')
        self.test_FOVs = load_hdf5(DRIVE_test_border_masks)

        path_experiment = '../' + self.name_experiment + '/'
        # Grouping of the predicted images
        average_mode = False
        self.batch_size = int(config.get('training settings', 'batch_size'))
        self.patch_height = int(config.get('data attributes', 'patch_height'))
        self.patch_width = int(config.get('data attributes', 'patch_width'))

        self.df_all = pd.DataFrame(columns=['Test Image', 'Pixels inside FOV', 'Threshold', 'ConfusionMat', 'IOU', 'Accuracy',
                                        'Specificity', 'Sensitivity', 'Precision', 'Weighted Average'])
        self.images2test = int(config.get('testing settings', 'images2test'))

        layers = 9
        filters = 3
        check_path = 'PlainNet_layer_%d_filter_%d.pt7' % (layers, filters)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)

        if self.SP == True:
            model_name = "NOPOOL_SP_5"
            self.model = M.NOPOOL_SP_5(input_size=(27, 27, 3))
        else:
            model_name = "PLAIN_net"
            self.model = M.PLAIN_net(input_size=(27, 27, 3))

        # Loads the weights
        self.model.load_weights('../code_venv1/weights_best_bn.hdf5')
        #model.load_weights('./' + name_experiment + '/' + name_experiment + '_last_weights.h5')

        for im in range(self.images2test):

            test_imgs = self.test_original[im:im+1, 10:575, :, :]
            test_masks = self.test_gt[im:im+1, 10:575, :, :]
            test_border_masks = self.test_FOVs[im:im+1, 10:575, :, :]

            print("\ntest images/masks shape:")
            print(test_imgs.shape, test_masks.shape)
            print("test images range (min-max): " + str(np.min(test_imgs)) + ' - ' + str(np.max(test_imgs)))

            # Keep pixels only inside borders of FOV => extract patches
            test_imgs, test_masks = pred_only_FOV_l(test_imgs, test_masks, test_border_masks)     # test_border_masks

            # test_gt = []
            # for i in range(test_masks.shape[0]):  # loop over the full images
            #     for h in range(test_masks.shape[1]):
            #         for w in range(test_masks.shape[2]):
            #             if not inside_FOV_DRIVE(i,h,w,test_border_masks):  # is pixel inside FOV?
            #                 continue
            #             validpixel = test_masks[i, h, w]
            #             test_gt.append(validpixel)
            # test_gt = np.array(test_gt)
            # print("Number of pixels in ground truth images inside FOV: ", test_gt.shape, test_masks.shape)

            if average_mode == True:
                # For another implementation of U-NET testing averaging the decision of 5 pixels instead of 1 in TEST patches
                self.patches_imgs_test, new_height, new_width, patches_gt_test = get_data_testing_overlap(
                    test_imgs,  # original
                    test_masks,  # masks
                    patch_height=patch_height,
                    patch_width=patch_width,
                    stride_height=stride_height,
                    stride_width=stride_width
                )
            elif self.SP == True:
                self.patches_imgs_test, patches_gt_test, windows = get_data_testing1(
                    test_imgs,  # original
                    test_masks, # truth
                    test_border_masks,  # border masks
                    self.patch_height,
                    self.patch_width,
                    self.SP
                )
            else:
                self.patches_imgs_test, test_masks, test_flat = get_data_testing1(   # return flat masks as test_gt
                    test_imgs,  # original
                    test_masks,  # truth
                    test_border_masks,  # border masks
                    self.patch_height,
                    self.patch_width,
                    self.SP
                )
            del test_border_masks, test_imgs, test_masks

            # ================ Run the prediction of the patches =============
            test_flat = to_categorical(test_flat, num_classes=2)
            print(test_flat.shape)

            print("GCN normalization:\n")
            # for step in range(0, patches_imgs_test.shape[0], self.batch_size):
            #     patches_imgs_test[step:step + self.batch_size] = np.asarray(
            #         NormImageDataset(patches_imgs_test[step:step + self.batch_size], dataset_normalized))

            # Apply the ZCA transformation already trained in the train set
            # with open('zca.pickle', 'rb') as f:
            #     zca_matrix = p.load(f)        
            # del f 
            # print("ZCA whitening:\n")
            # first_dim = patches_imgs_test.shape[0]
            # shape = (first_dim, patches_imgs_test.shape[1] * patches_imgs_test.shape[2] * patches_imgs_test.shape[3])
            # patches_imgs_test = patches_imgs_test.reshape(shape)
            # X_norm = patches_imgs_test - np.mean(patches_imgs_test, 0)  # zero-center the data to 1st dimension [400, 2187]
            # del patches_imgs_test
            # X_ZCA = zca_matrix.dot(X_norm.T).T
        
            # del X_norm, zca_matrix
            # print("XZCA min-max\n")
            # xzca_min = np.min(X_ZCA)
            # xzca_max = np.max(X_ZCA)
            # denom = xzca_max - xzca_min
            # del xzca_max
            # denom = max(denom, 0.00001)
            # X_ZCA = (X_ZCA - xzca_min) / denom
            # del xzca_min, denom
            # self.patches_imgs_test = X_ZCA.reshape(first_dim, 27, 27, 3)
            # del X_ZCA

            print('==> Liskowski Test from checkpoint..')
            test_generator = Test_Generator(self.patches_imgs_test, test_flat, 1)

            print(self.patches_imgs_test.shape)
            del self.patches_imgs_test
            gc.collect()
            self.pred_patches = self.model.predict(test_generator)
            print(type(self.pred_patches), "ok predicted\n")
            self.pred_patches = np.array(self.pred_patches)  #patches predictions for all pixels in test images
            print(self.pred_patches.shape)    # (patches, 2)
            print("Prediction finished\n")

            # threshold = 0.5
            # preds_over, df_all = evaluation_metrics(predictions, test_gt, threshold, im, df_all)

            threshold = optimal_thres_tuning(test_flat, self.pred_patches)
            preds_over, self.df_all = evaluation_metrics(self.pred_patches, test_flat, threshold, im, self.df_all)

            # Convert predictions to image and plot:
            # pred2image(preds_over, self.test_gt, im)
            del preds_over, test_flat, test_generator

        del self.df_all, self.test_original, self.test_FOVs, self.test_gt
        return


    def load_dataS(self):
        train_original, train_masks, train_FOVs, test_original, test_masks, test_FOVs, inside_FOV = self.loadall()
        train_original, train_masks = get_data_training(
            train_original,
            train_masks,
            self.patch_height,
            self.patch_width,
            inside_FOV)  # select the patches only inside the FOV  (default == True)

        # Preprocess in batches
        # MTHT ENHANCEMENT + CLAHE
        self.train_original = (train_original / 255.).astype(np.float32)
        print("SINE train original shape: ", self.train_original.shape)
        self.train_masks = (train_masks / 255.).astype(np.float32)
        self.train_FOVs = (train_FOVs / 255.).astype(np.float32)
        self.test_original = (test_original / 255.).astype(np.float32)
        self.test_masks = (test_masks / 255.).astype(np.float32)
        self.test_FOVs = (test_FOVs / 255.).astype(np.float32)
        print("End of extraction and preprocessing\n")
        print(self.train_original.mean(), self.train_masks.mean())
        print("Train test split for SINE: \n")
        del train_original, train_masks, train_FOVs, test_original, test_masks, test_FOVs

        ## number of patches equal the number of images
        # indices = range(20)
        ## self.x_train, self.x_test, self.y_train, self.y_test
        # self.xtrain, self.xval, self.ytrain, self.yval, indices_train, indices_test = train_test_split(train_original, train_masks,
        #                                                                            indices, test_size=0.1,
        #                                                                            random_state=42)
        # print(indices_train, indices_test)
        # indices_list = [x + 1 for x in indices_train]


    def Liskowski_train(self, x_train, y_train, x_val, y_val):
        self.train_data = Custom_Generator(x_train, y_train, self.batch_size, augmentation=True, shuffle=True)
        self.val_data = Custom_Generator(x_val, y_val, self.batch_size, augmentation=False, shuffle=False)
        # X, y = train_dataset[0] # evaluate custom generator
        # print("Dataset: ", X, y)

        filepath = "weights_best.hdf5"
        # ModelCheckpoint only saves the weight only if loss decreases
        checkpoint = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', verbose=1, mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, epsilon=1e-7,
                                           mode='min')

        history = self.model.fit(
            self.train_data,
            epochs=self.N_epochs,
            # class_weight = weights,
            validation_data=self.val_data,
            callbacks=[reduce_lr_loss, checkpoint]
        )
        self.pred_patches = self.model.evaluate(self.val_data, verbose=1)
        del self.train_data, self.val_data, x_train, y_train, x_val, y_val

        # Only saves last epoch's weights where the performance (aka loss) could not be optimal
        self.model.save_weights('./' + self.name_experiment + '/' + self.name_experiment + '_last_weights.h5', overwrite=True)
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
        return


    def StackingClassifier(self):

        # Define weak learners
        # weak_learners = [M.PLAIN_net(input_size=(27, 27, 3)), Sine_Net(1,1)]
        weak_learners = ["PLAIN_net", "Sine_Net"]

        # Final learner or meta-model
        final_learner = LogisticRegression()

        train_meta_model = None
        test_meta_model = None

        # Start stacking
        for clf_id, clf in enumerate(weak_learners):
            # Predictions for each classifier based on k-fold
            print("{}-Classifier".format(clf_id))

            if clf_id == 0:     #Liskowski model
                ensemble.load_dataL()
            else:               # SineNet model
                ensemble.load_dataS()

            # Predictions for each classifier based on k-fold
            predictions_clf = self.k_fold_cross_validation(clf_id)
            print("Predictions of each Weak Learner Classifier\n")

            # Predictions for test set for each classifier based on train of level 0
            self.train_level_0(clf_id)

            # Stack predictions which will form
            # the input data for the data model
            if isinstance(train_meta_model, np.ndarray):
                train_meta_model = np.vstack((train_meta_model, predictions_clf))
            else:
                train_meta_model = predictions_clf

            # Stack predictions from test set
            # which will form test data for meta model
            if isinstance(test_meta_model, np.ndarray):
                test_meta_model = np.vstack((test_meta_model, self.pred_patches))
            else:
                test_meta_model = self.pred_patches
        
        del self.pred_patches
        # Transpose train_meta_model
        train_meta_model = train_meta_model.T

        # Transpose test_meta_model
        test_meta_model = test_meta_model.T

        # Training level 1
        self.train_level_1(final_learner, train_meta_model, test_meta_model)

    def train(self, epoch, params, train_losses, train_corrects_history):
        metric_monitor = MetricMonitor()  # initialize (reset) metrics to zero
        self.model.train()
        self.train_loader = tqdm(self.train_loader)
        total_train = 0
        correct_train = 0
        for batch_idx, (images, target) in enumerate(train_loader, start=1):
            images = images.to(params["device"], dtype=torch.half,
                               non_blocking=True)  # for GPU usage: move the device we are training on
            target = target.to(params["device"], dtype=torch.half, non_blocking=True)
            self.optimizer.zero_grad()  # clear the last error gradient

            # plt.figure()
            # plt.hist(images.cpu().numpy().ravel(), bins=50, density=True)
            # plt.xlabel("pixel values")
            # plt.ylabel("relative frequency")
            # plt.title("distribution of pixels")
            # plt.savefig("image_distrib.png")

            output_s = self.model(images)
            ##output_s = torch.sigmoid(output_s)
            loss = self.criterion(output_s, target)

            metric_monitor.update("Train_loss", loss.item(), images.size(0))
            accuracy = calculate_acc(output_s, target, total_train, correct_train)
            metric_monitor.update("Train_acc", accuracy, images.size(0))
            loss.backward()  # backccopagate the error through the model
            self.optimizer.step()  # update model weights
            self.train_loader.set_description(
                "Epoch: {epoch}. Train. {metric_monitor}.".format(epoch=epoch, metric_monitor=metric_monitor)
            )

        epoch_loss = metric_monitor.metrics['Train_loss']['avg']
        epoch_acc = metric_monitor.metrics['Train_acc']['avg']
        train_losses.append(epoch_loss)
        train_corrects_history.append(epoch_acc)
        print("Epoch training loss/accuracy: ", epoch_loss, " / ", epoch_acc)
        print("Epoch: {}. Learning rate: {}".format(epoch, self.optimizer.param_groups[0]['lr']))

        return train_losses, train_corrects_history

    def validate(self, epoch, params, val_losses, val_corrects_history):
        global best_loss
        metric_monitor = MetricMonitor()  # initialize (reset) metrics to zero
        self.model.eval()  # affects certain model layers to be used during model inference
        self.val_loader = tqdm(self.val_loader)
        total_val = 0
        correct_val = 0
        with torch.no_grad():  # no_grad: turn off gradients computation/usage during eval time -> speed up / reduce memory
            for batch_idx, (images, target) in enumerate(val_loader, start=1):
                images = images.to(params["device"], dtype=torch.half,
                                   non_blocking=True)  # for GPU usage: move to the device we are training on
                target = target.to(params["device"], dtype=torch.half, non_blocking=True)
                # print(target.shape)

                output_s = self.model(images)
                ##output_s = torch.sigmoid(output_s)
                loss = self.criterion(output_s, target)

                metric_monitor.update("Val_loss", loss.item(), images.size(0))
                accuracy = calculate_acc(output_s, target, total_val, correct_val)
                metric_monitor.update("Val_acc", accuracy, images.size(0))
                self.val_loader.set_description(
                    "Epoch: {epoch}. Validation. {metric_monitor}.".format(epoch=epoch, metric_monitor=metric_monitor)
                )

        # epoch loss = running loss accumulation / data length
        epoch_loss = metric_monitor.metrics['Val_loss']['avg']
        epoch_acc = metric_monitor.metrics["Val_acc"]['avg']
        val_losses.append(epoch_loss)
        val_corrects_history.append(epoch_acc)
        print("Epoch validation loss/accuracy: ", epoch_loss, " / ", epoch_acc)

        return val_losses, val_corrects_history, metric_monitor.metrics['Val_loss']['sum']

    def train_and_validate(self, params, train_losses, val_losses,
                           train_corrects_history, val_corrects_history):
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
            pin_memory=True,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_data,
            batch_size=params["batch_size"],
            shuffle=False,
            num_workers=params["num_workers"],
            pin_memory=True,
        )
        # criterion = nn.BCEWithLogitsLoss().to(params["device"])
        self.criterion = WBCE_DiceLoss().to(params["device"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=.1, verbose=True)
        print(f'Starting epoch: {params["start_epoch"]}, - Amount of epochs to train: {params["epochs"]}')
        ## Epochs Iteration
        best_loss = float("inf")
        for epoch in range(params["start_epoch"], params["start_epoch"] + params["epochs"]):
            # enumerate mini-batches
            train_losses, train_corrects_history = train(epoch, params, train_losses, train_corrects_history)
            val_losses, val_corrects_history, val_loss_ep = validate(epoch, params, val_losses, val_corrects_history)
            scheduler.step(val_losses[epoch - 1])
            # Save checkpoint for best validation loss model.
            if val_loss_ep < best_loss:  # the accumulated loss
                print('Saving..')
                best_loss = val_loss_ep
                state = {
                    'model_state': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'epoch': epoch
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/' + check_path_best)

        print('Saving last model after number of epochs:', epoch)
        state = {
            'model_state': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': val_losses[-1],
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        print("checkpoint saved\n")
        torch.save(state, './checkpoint/' + check_path_cont)
        return self.model, train_losses, val_losses, train_corrects_history, val_corrects_history


    def Sinenet_train(self, x_all, y_all, fov_val, indices_train, indices_test):
        # 2 augmentation steps
        augmented_path = self.data_path + 'Augmented_Images/'
        new_file_path = augmented_path + 'augmented_img_train'
        new_file_path_mask = augmented_path + 'augmented_gt_train'
        orig_path = augmented_path + 'img_train'
        orig_path_mask = augmented_path + 'gt_train'

        xtrain_aug, ytrain_aug = [], []
        from augmentation import aug_apply
        #aug_apply(x_all, y_all, indices_train, self.patch_height, self.data_path, 1)   # step 1 of augmentations

        # load saved images after step 1:
        for i in indices_train:
            xtrain_aug.append(torch.load(new_file_path + str(i) + '_0.pt'))
            ytrain_aug.append(torch.load(new_file_path_mask + str(i) + '_0.pt'))

        #aug_apply(x_all, y_all, indices_train, self.patch_height, self.data_path, 2)
        # load saved images after step 2:
        for i in indices_train:
            for j in range(1, 9):
                xtrain_aug.append(torch.load(new_file_path + str(i) + '_' + str(j) + '.pt'))
                ytrain_aug.append(torch.load(new_file_path_mask + str(i) + '_' + str(j) + '.pt'))

        # load saved original cropped images:
        for i in indices_train:
            xtrain_aug.append(torch.load(orig_path + str(i) + '.pt'))
            ytrain_aug.append(torch.load(orig_path_mask + str(i) + '.pt'))

        xtrain_aug = np.stack(xtrain_aug)
        ytrain_aug = np.stack(ytrain_aug)

        # ----------------- Normalizing -----------------
        # Data Augmentation transforms
        train_transform = Compose([
            Rotate(limit=45, p=0.5, interpolation=cv2.INTER_LINEAR),
            Affine(translate_px=(0, 117), p=0.5, interpolation=cv2.INTER_LINEAR),
            Rotate(limit=90, p=0.5, interpolation=cv2.INTER_LINEAR),
            OneOf([VerticalFlip(), HorizontalFlip()], p=0.5),
            Resize(self.patch_height, self.patch_width, p=1),
            ToTensorV2(p=1)
        ])
        val_transform = Compose([
            Resize(self.patch_height, self.patch_width, p=1),
            ToTensorV2(p=1)
        ])

        x_val, y_val = x_all[indices_test], y_all[indices_test]

        scaler = StandardScaler()
        arr_norm = scaler.fit_transform(xtrain_aug.reshape(-1, 565 * 565))
        arr_normv = scaler.transform(x_val.reshape(-1, 565 * 565))

        # save scaler for test data
        import pickle as p
        with open("scaler.pickle", "wb") as f:
            p.dump(scaler, f)

        self.train_data = AugImageDataset(arr_norm.reshape(-1, 565, 565, 1), ytrain_aug, train_transform)
        self.val_data = AugImageDataset(arr_normv.reshape(-1, 565, 565, 1), y_val, val_transform)
        del arr_norm, arr_normv, x_val, y_val, x_all, y_all, xtrain_aug, ytrain_aug

        #### ------------ Sine Net training Process -----------
        sampler = "no_sampler"

        print('Training')
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            print('CUDA is not available. Training on CPU...')
            device = torch.device('cpu')
        else:
            print('CUDA is available. Training on GPU...')
            device = torch.device('cuda')

        params = {
            "model_name": "Sine_Net",
            "lr": 0.0001,
            "device": device,
            "batch_size": self.batch_size,
            "num_workers": 0,
            "epochs": 70,
            "start_epoch": 1
        }
        ## model initialization
        torch.set_default_dtype(torch.float16)
        gc.collect()
        torch.cuda.empty_cache()
        checkpoint = torch.load('./checkpoint/' + check_path_best, map_location='cpu')
        batch_size, C, H, W = int(params["batch_size"]), 1, 448, 448
        self.model = Sine_Net(C, 1)  # initialize with #channels & #classes
        #self.model.load_state_dict(checkpoint['model_state'])
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        print(f)
        self.model = self.model.half().to(params["device"])  # model.half()

        # Weights Initialization
        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.model.apply(weights_init)

        for layer in self.model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

        resume = False
        if resume == True:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/' + check_path_best)
            self.model.load_state_dict(checkpoint['model_state'])
            params["start_epoch"] = checkpoint['epoch'] + 1
        else:
            params["start_epoch"] = 1

        print(f"Training on device {device}.")
        optimizer1 = torch.optim.SGD(self.model.parameters(), lr=params["lr"], momentum=0.9,
                                     nesterov=True)  ##weight_decay=0.1
        self.optimizer = optimizer1
        params["epochs"] = 5
        # Initialization of total epochs metrics
        train_losses, val_losses = [], []  # Training Loss & Validation Loss history
        train_corrects_history, val_corrects_history = [], []  # Accuracy history
        self.model, train_losses, val_losses, train_corrects_history, val_corrects_history = train_and_validate(params,
                                                                                                           train_losses,
                                                                                                           val_losses,
                                                                                                           train_corrects_history,
                                                                                                           val_corrects_history)
        # save best trained model in validate function

        resume = True
        if resume == True:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            # load the model checkpoint
            checkpoint = torch.load('./checkpoint/' + check_path_cont)
            # load model weights state_dict
            self.model.load_state_dict(checkpoint['model_state'])
            # # load trained optimizer state_dict
            optimizer = checkpoint['optimizer']
            params["start_epoch"] = checkpoint['epoch'] + 1
            # print(f"Previously trained for {epochs} number of epochs...")
            print('Loading epoch {} successful! '.format(params["start_epoch"]))
            # If there is a saved model, load the model and continue training based on it
        else:
            params["start_epoch"] = 1
            print('No saving model, training from scratch! ')

        print(f"Training on device {device}.")
        # params["lr"] = optimizer['param_groups'][0]['lr']
        params["lr"] = 1e-06
        print("Continue from learning rate: ", params["lr"])
        optimizer2 = torch.optim.Adam(self.model.parameters(), lr=params["lr"], betas=(0.9, 0.999),
                                      eps=1e-04)  ##weight_decay=0.1
        params["epochs"] = 5
        # Initialization of total epochs metrics
        self.model, train_losses, val_losses, train_corrects_history, val_corrects_history = train_and_validate(model,
                                                                                                           self.train_data,
                                                                                                           self.val_data,
                                                                                                           sampler,
                                                                                                           params,
                                                                                                           optimizer2,
                                                                                                           train_losses,
                                                                                                           val_losses,
                                                                                                           train_corrects_history,
                                                                                                           val_corrects_history)

        ## Evaluate for k-fold for sef.val_data predictions (not binary images)
        ## Separate tuple val_data to X and y
        for i in range(len(self.val_data)):
            sample = self.val_data[i]
            val_img[i] = sample['image']
            val_mask[i] = sample['mask']

            eval = Test(params, 448, 448, self.val_img[i], self.val_mask[i], self.fov_val[i], path_experiment, i)
            eval.inference(self.model)      ## computes self.pred_patches


        del val_img, val_mask, self.train_data, self.val_data
        # loss plots
        plt.figure(figsize=(10, 7))
        plt.plot(train_losses, color='green', label='train loss')
        plt.plot(val_losses, color='blue', label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if not os.path.isdir('outputs'):
            os.mkdir('outputs')
        plt.savefig('./outputs/loss_batch_onlineBCELoss.png')
        plt.show()

        # accuracy plots
        plt.figure(figsize=(10, 7))
        plt.plot(train_corrects_history, color='green', label='train accuracy')
        plt.plot(val_corrects_history, color='blue', label='validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        # create a folder for the results
        result_dir = 'outputs'
        print("\n1. Create directory for the results (if not already existing)")
        if os.path.exists(result_dir):
            print("Dir already existing")
        else:
            os.system('mkdir ' + result_dir)
        plt.savefig('./outputs/accuracy_batch_onlineBCELoss.png')
        plt.show()
        return

    def k_fold_cross_validation(self, clf_id):
        
        i = 0
        datatest_x, datatest_y, predictions_clf = [], [], []
        kfold = KFold(n_splits=self.k, shuffle=True, random_state=42)
        for train_ix, test_ix in kfold.split(self.train_original):
            # get data
            # train_X, test_X = self.train_original[train_ix], self.train_original[test_ix]
            # train_y, test_y = self.train_masks[train_ix], self.train_masks[test_ix]

            i += 1
            print("Training begins...\n", str(i))
            # fit and make predictions with clf
            if clf_id == 0:     ## Liskowski process
                self.Liskowski_train(self.train_original[train_ix], self.train_masks[train_ix], self.train_original[test_ix], self.train_masks[test_ix]) 
            elif clf_id == 1:     ## SineNet process
                # FOV mask only for validation set -> SineNet
                print(train_ix)
                print(self.test_FOVs.shape)
                print("start sine net\n")
                self.Sinenet_train(self.train_original, self.train_masks, self.test_FOVs[test_ix], train_ix, test_ix)

            predictions_clf.append(self.pred_patches)
            print(len(predictions_clf))
            print("train done\n")
        
        del self.train_original, self.train_masks, self.test_FOVs, self.model
        return predictions_clf


    def Sinenet_predict(self):
        print('==> Sine-Net Test from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/' + self.check_path_cont, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state'])

        params = {
            "model_name": "Sine_Net",
            "device": 'cuda',       ### device
            "batch_size": 4,
            "num_workers": 2,
            "model": model
        }
        
        self.model.to(params["device"])
        ## Create test image's patches per image
        for i, test_ori in enumerate(test_original.shape[0]):
            eval = Test(params, 448, 448, self.test_original[i], self.test_masks[i], self.test_FOVs[i], path_experiment)
            eval.inference(model)
            print(eval.evaluate())
            eval.save_segmentation_result()


    def train_level_0(self, clf_id):
        # Train in full real training set
        test_X, test_y = None, None
        train_ix = range(20)
        if clf_id==0:
            self.Liskowski_predict()
        elif clf_id==1:
            # Sinenet_train(test_X, test_y, train_ix)     ## train with no evaluation
            # Get predictions from full real test set
            self.Sinenet_predict()           
        return 

    def train_level_1(self, final_learner, train_meta_model, test_meta_model):
        # Train is carried out with final learner or meta model
        final_learner.fit(train_meta_model, self.train_masks)
        # Getting train and test accuracies from meta_model
        print(f"Train accuracy: {final_learner.score(train_meta_model, self.train_masks)}")
        print(f"Test accuracy: {final_learner.score(test_meta_model, self.test_masks)}")

def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('lr', LogisticRegression()))
	level0.append(('knn', KNeighborsClassifier()))
	level0.append(('cart', DecisionTreeClassifier()))
	level0.append(('svm', SVC()))
	level0.append(('bayes', GaussianNB()))
	# define meta learner model
	level1 = LogisticRegression()
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
	return model


# get a list of models to evaluate
def get_models():
    models = dict()
    models['lisk'] = M.PLAIN_net(input_size=(27, 27, 3))
    models['sine'] = Sine_Net(1,1)
    models['stacking'] = get_stacking()
    return models

if __name__ == "__main__":
    ensemble = Ensemble()
    ensemble.StackingClassifier()
