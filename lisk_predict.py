from extract_patches_l import *
from prediction_sn import *
from test_retina_keras import *
import configparser as cfp
import gc

class Lisk_t:
    def __init__(self):
        self.test_original = None
        self.test_FOVs = None
        self.test_gt = None
        self.name_experiment = None
        self.N_epochs = None
        self.batch_size = None
        self.model = None
        self.patch_height = None
        self.patch_width = None
        self.df_all = None
        self.images2test = None
        self.SP = None
        self.pred_patches = None
        self.patches_imgs_test = None


    def Liskowski_predict(self):
        config = cfp.RawConfigParser()
        config.read('../code_venv1/configuration_l.txt')
        # path to the datasets
        self.data_path = config.get('data paths', 'path_local')
        DRIVE_test_imgs_original = self.data_path + config.get('data paths', 'test_imgs_original')
        self.test_original = load_hdf5(DRIVE_test_imgs_original)

        # the ground truth masks
        DRIVE_test_groudTruth = self.data_path + config.get('data paths', 'test_groundTruth')
        self.test_gt = load_hdf5(DRIVE_test_groudTruth)  # self.test_gt for all images masks

        # the border masks provided by the DRIVE
        DRIVE_test_border_masks = self.data_path + config.get('data paths', 'test_border_masks')
        self.test_FOVs = load_hdf5(DRIVE_test_border_masks)

        self.name_experiment = config.get('experiment name', 'name')
        path_experiment = '../' + self.name_experiment + '/'
        # Grouping of the predicted images
        average_mode = False
        self.batch_size = int(config.get('training settings', 'batch_size'))
        self.patch_height = int(config.get('data attributes', 'patch_height'))
        self.patch_width = int(config.get('data attributes', 'patch_width'))

        self.df_all = pd.DataFrame(
            columns=['Test Image', 'Pixels inside FOV', 'Threshold', 'ConfusionMat', 'IOU', 'Accuracy',
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
        # model.load_weights('./' + name_experiment + '/' + name_experiment + '_last_weights.h5')

        for im in range(self.images2test):

            test_imgs = self.test_original[im:im + 1, 10:575, :, :]
            test_masks = self.test_gt[im:im + 1, 10:575, :, :]
            test_border_masks = self.test_FOVs[im:im + 1, 10:575, :, :]

            print("\ntest images/masks shape:")
            print(test_imgs.shape, test_masks.shape)
            print("test images range (min-max): " + str(np.min(test_imgs)) + ' - ' + str(np.max(test_imgs)))

            # Keep pixels only inside borders of FOV => extract patches
            test_imgs, test_masks = pred_only_FOV_l(test_imgs, test_masks, test_border_masks)  # test_border_masks

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
                    test_masks,  # truth
                    test_border_masks,  # border masks
                    self.patch_height,
                    self.patch_width,
                    self.SP
                )
            else:
                self.patches_imgs_test, test_flat = get_data_testing1(  # return flat masks as test_gt
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
            self.pred_patches = np.array(self.pred_patches)  # patches predictions for all pixels in test images
            print(self.pred_patches.shape)  # (patches, 2)
            print("Prediction finished\n")

            # threshold = 0.5
            # preds_over, df_all = evaluation_metrics(predictions, test_gt, threshold, im, df_all)

            threshold = optimal_thres_tuning(test_flat, self.pred_patches)
            preds_over, _ = evaluation_metrics(self.pred_patches, test_flat, threshold, im, self.df_all)

            # Convert predictions to image and plot:
            # pred2image(preds_over, self.test_gt, im)
            del test_flat, test_generator, self.pred_patches

        del self.test_original, self.test_FOVs, self.test_gt
        return preds_over
