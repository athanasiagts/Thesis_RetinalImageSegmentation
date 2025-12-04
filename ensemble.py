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

from sinenet_torch_model import *
import models_setup as M
from extract_patches import *
from extract_patches_l import *
from training_datagen import *
from prediction_sn import *
from train_retina import *
from test_retina_keras import *

####### ----------------- 2nd way of stacking --------------------------
class Ensemble:
    def __init__(self):

        self.test_original = None
        self.test_gt = None
        self.test_FOVs = None
        self.patch_width = None
        self.patch_height = None
        self.batch_size = None
        self.df_all = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.x_test = None
        self.y_test = None
        self.check_path_cont = None
        self.N_epochs = None
        self.N_subimgs = None
        self.inside_FOV = None
        self.SP = None
        self.name_experiment = None
        self.patches_imgs_test = None
        self.images2test = None
        self.predictions_clf = []

        self.k = 2

    def loadall(self):
        config = cfp.RawConfigParser()
        config.read('configuration.txt')
        self.data_path = config.get('data paths', 'path_local')
        path_drive = '../../data/DRIVE'  
        self.name_experiment = config.get('experiment name', 'name')
        # training settings
        N_epochs = int(config.get('training settings', 'N_epochs'))

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

        del gts, images, masks, gif_reader
        gc.collect()
        return test_original, test_masks, test_FOVs, inside_FOV


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

        del self.pred_patches, self.df_all, self.test_original, self.test_FOVs, self.test_gt
        return


    def Liskowski_train(self):
        
        config = cfp.RawConfigParser()
        config.read('../code_venv1/configuration_l.txt')
        self.name_experiment = config.get('experiment name', 'name')
        filepath = '../code_venv1/' + self.name_experiment + '/' + self.name_experiment + '_last_weights.h5'
        check_path = '../code_venv1/checkpoint/' + 'PLAIN_net_layer_%d_filter_%d.pt7'% (9,3)
        # load json and create model
        # Create the same model from json and load weights
        json_file_loaded = open('./' +  self.name_experiment + '/' +  self.name_experiment + '_architecture.json', 'r')
        loaded_model_json = json_file_loaded.read()

        # load model using the saved json file
        loaded_model = tf.keras.models.model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(filepath)
        print("Loaded model from disk\n")


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

            # Predictions for each classifier based on k-fold
            self.k_fold_cross_validation(clf_id)                        # changes self.predictions_clf
            print("Predictions of each Weak Learner Classifier\n")

            # Predictions for test set for each classifier based on train of level 0
            self.train_level_0(clf_id)

            # Stack predictions which will form
            # the input data for the data model
            if isinstance(train_meta_model, np.ndarray):
                train_meta_model = np.vstack((train_meta_model, self.predictions_clf))
            else:
                train_meta_model = self.predictions_clf

            # Stack predictions from test set
            # which will form test data for meta model
            if isinstance(test_meta_model, np.ndarray):
                test_meta_model = np.vstack((test_meta_model, self.pred_patches))
            else:
                test_meta_model = self.pred_patches

        # Transpose train_meta_model
        train_meta_model = train_meta_model.T

        # Transpose test_meta_model
        test_meta_model = test_meta_model.T

        # Training level 1
        self.train_level_1(final_learner, train_meta_model, test_meta_model)   


    def Sinenet_train(self):

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

        batch_size, C, H, W = int(params["batch_size"]), 1, 448, 448
        self.model = Sine_Net(C, 1)  # initialize with #channels & #classes
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


    def k_fold_cross_validation(self, clf_id):
        
        # kfold = KFold(n_splits=self.k, shuffle=True, random_state=42)
        # for train_ix, test_ix in kfold.split(self.train_original):

        print("Training begins...\n")
        # fit and make predictions with clf
        if clf_id == 0:
            # Load trained model on k-fold
            self.Liskowski_train()
        elif clf_id == 1:
            # Load trained model on k-fold
            self.Sinenet_train()

        # # Save (append) 0 level predictions to test data for 2nd phase training+prediction
        # self.predictions_clf.append(self.pred_patches)
        # print(len(self.predictions_clf))
        print("training done\n")


    def Sinenet_predict(self):

        test_original, test_masks, test_FOVs, inside_FOV = self.loadall()
        self.test_original = (test_original / 255.).astype(np.float32)
        self.test_masks = (test_masks / 255.).astype(np.float32)
        self.test_FOVs = (test_FOVs / 255.).astype(np.float32)
        print("End of extraction and preprocessing\n")
        del test_original, test_masks, test_FOVs

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
            # Get predictions from full real test set
            self.Liskowski_predict()
        elif clf_id==1:
            # Get predictions from full real test set
            self.Sinenet_predict()        

        # Save (append) 0 level predictions to test data for 2nd phase training+prediction
        self.predictions_clf.append(self.pred_patches)
        print(len(self.predictions_clf))   
        return 

    def train_level_1(self, final_learner, train_meta_model, test_meta_model):
        # Train is carried out with final learner or meta model
        final_learner.fit(train_meta_model, self.train_masks)
        # Getting train and test accuracies from meta_model
        print(f"Train accuracy: {final_learner.score(train_meta_model, self.train_masks)}")
        print(f"Test accuracy: {final_learner.score(test_meta_model, self.test_masks)}")

    # def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 
    #                    model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
    #                    scoring_test=r2_score, do_probabilities = False):
    #     gs = GridSearchCV(
    #         estimator=model,
    #         param_grid=param_grid, 
    #         cv=cv, 
    #         n_jobs=-1, 
    #         scoring=scoring_fit,
    #         verbose=2
    #     )
    #     fitted_model = gs.fit(X_train_data, y_train_data)
    #     best_model = fitted_model.best_estimator_
        
    #     if do_probabilities:
    #         pred = fitted_model.predict_proba(X_test_data)
    #     else:
    #         pred = fitted_model.predict(X_test_data)
        
    #     score = scoring_test(y_test_data, pred)
        
    #     return [best_model, pred, score]



if __name__ == "__main__":
    ensemble = Ensemble()
    ensemble.StackingClassifier()

    # from sklearn.ensemble import RandomForestRegressor
    # from lightgbm import LGBMRegressor
    # from xgboost import XGBRegressor

    # # Defining our estimator, the algorithm to optimize
    # models_to_train = [XGBRegressor(), LGBMRegressor(), RandomForestRegressor()]

    # # Defining the hyperparameters to optimize
    # grid_parameters = [
    #     { # XGBoost
    #         'n_estimators': [400, 700, 1000],
    #         'colsample_bytree': [0.7, 0.8],
    #         'max_depth': [15,20,25],
    #         'reg_alpha': [1.1, 1.2, 1.3],
    #         'reg_lambda': [1.1, 1.2, 1.3],
    #         'subsample': [0.7, 0.8, 0.9]
    #     },
    #     { # LightGBM
    #         'n_estimators': [400, 700, 1000],
    #         'learning_rate': [0.12],
    #         'colsample_bytree': [0.7, 0.8],
    #         'max_depth': [4],
    #         'num_leaves': [10, 20],
    #         'reg_alpha': [1.1, 1.2],
    #         'reg_lambda': [1.1, 1.2],
    #         'min_split_gain': [0.3, 0.4],
    #         'subsample': [0.8, 0.9],
    #         'subsample_freq': [10, 20]
    #     }, 
    #     { # Random Forest
    #         'max_depth':[3, 5, 10, 13], 
    #         'n_estimators':[100, 200, 400, 600, 900],
    #         'max_features':[2, 4, 6, 8, 10]
    #     }
    # ]

    # models_preds_scores = []

    # for i, model in enumerate(models_to_train):
    #     params = grid_parameters[i]
        
    #     result = algorithm_pipeline(X_train, X_test, y_train, y_test, 
    #                                 model, params, cv=5)
    #     models_preds_scores.append(result)
