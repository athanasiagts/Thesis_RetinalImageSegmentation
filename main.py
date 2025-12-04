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
from lisk_train import *
from sine_train import *
from lisk_predict import *
from sine_predict import *
import gc

sys.path.insert(0, os.path.abspath('../code_venv1/'))
print(sys.path)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


class Ensemble:
    def __init__(self):
        self.patches_imgs_train = None
        self.patches_gt_train = None
        self.test_original = None
        self.test_gt = None
        self.patch_width = None
        self.patch_height = None
        self.batch_size = None
        self.df_all = None
        self.model = None
        self.optimizer = None
        self.val_loader = None
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
        self.check_path_best = 'SineNet_layer_%d_filter_%d_best_test_Loss.pt7' % (15, 3)
        self.check_path_cont = 'SineNet_layer_%d_filter_%d_continue_test_Loss.pt7' % (15, 3)

        self.k = 2

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
                data = Lisk()
                data.load_dataL()
            else:               # SineNet model
                data = Sine()
                data.load_dataS()

            # Predictions for each classifier based on k-fold
            predictions_clf = self.k_fold_cross_validation(clf_id, data)
            print("Predictions of each Weak Learner Classifier\n")
            del data

            # Predictions for test set for each classifier based on train of level 0
            pred_patches = self.train_level_0(clf_id)

            # Stack predictions which will form
            # the input data for the data model
            if isinstance(train_meta_model, np.ndarray):
                train_meta_model = np.vstack((train_meta_model, predictions_clf))
            else:
                train_meta_model = predictions_clf

            # Stack predictions from test set
            # which will form test data for meta model
            if isinstance(test_meta_model, np.ndarray):
                test_meta_model = np.vstack((test_meta_model, pred_patches))
            else:
                test_meta_model = pred_patches

        del pred_patches, data_t
        # Transpose train_meta_model
        train_meta_model = train_meta_model.T

        # Transpose test_meta_model
        test_meta_model = test_meta_model.T

        # Training level 1
        self.train_level_1(final_learner, train_meta_model, test_meta_model)

    def k_fold_cross_validation(self, clf_id, data):

        i = 0
        datatest_x, datatest_y, predictions_clf = [], [], []
        kfold = KFold(n_splits=self.k, shuffle=True, random_state=42)
        for train_ix, test_ix in kfold.split(data.train_original):

            i += 1
            print("Training begins...\n", str(i))
            # fit and make predictions with clf
            if clf_id == 0:  # Liskowski process
                pred_patches = data.Liskowski_train(data.train_original[train_ix], data.train_masks[train_ix],
                                               data.train_original[test_ix], data.train_masks[test_ix])
            elif clf_id == 1:  # SineNet process
                # FOV mask only for validation set -> SineNet
                print("start sine net\n")
                pred_patches = data.Sinenet_train(data.train_original, data.train_masks, train_ix, test_ix)

            predictions_clf.append(pred_patches)
            print(len(predictions_clf))
            print("train done\n")

        self.model
        return predictions_clf

    def train_level_0(self, clf_id):
        # Train in full real training set
        test_X, test_y = None, None
        train_ix = range(20)
        if clf_id == 0:
            data_t = Lisk_t()
            pred_patches = data_t.Liskowski_predict()
        elif clf_id == 1:
            # Sinenet_train(test_X, test_y, train_ix)     ## train with no evaluation
            # Get predictions from full real test set
            data_t = Sine_t()
            pred_patches = data_t.Sinenet_predict()
        return pred_patches

    def train_level_1(self, final_learner, train_meta_model, test_meta_model):
        # Train is carried out with final learner or meta model
        final_learner.fit(train_meta_model, data.train_masks)
        # Getting train and test accuracies from meta_model
        print(f"Train accuracy: {final_learner.score(train_meta_model, self.train_masks)}")
        print(f"Test accuracy: {final_learner.score(test_meta_model, self.test_masks)}")


if __name__ == "__main__":
    ensemble = Ensemble()
    ensemble.StackingClassifier()
