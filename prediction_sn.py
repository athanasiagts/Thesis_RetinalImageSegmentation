# Predict labels for images and visualize those predictions
############################################################
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#############################################################
import numpy as np
import pandas as pd
from tqdm import tqdm
import configparser
from matplotlib import pyplot as plt
import torch, os, sys
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader,Dataset
from tensorflow.keras.utils import plot_model, to_categorical, Sequence
#scikit learn
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score # AUC is the percentage of the ROC plot that is underneath the curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import f1_score
import sys
sys.path.insert(0, '../')
# extract_patches.py
from extract_patches import *
import torch, torchvision
import torch.nn as nn
import random, cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms, Compose, functional
import skimage
from collections import OrderedDict
import sys
# np.set_printoptions(threshold=sys.maxsize)
from albumentations import OneOf, Rotate, Flip, HorizontalFlip, VerticalFlip, Compose, CenterCrop, Affine, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re, imageio
from metrics import *


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [
        int(text)
        if text.isdigit() else text.lower()
        for text in _nsre.split(s)]


def visualization(pred_imgs, orig_imgs, gtruth_masks):
    pred_imgs = pred_imgs.reshape(1, 565, 565, 1)
    print("Orig imgs shape: " +str(orig_imgs.shape))
    print("pred imgs shape: " +str(pred_imgs.shape))
    print("Gtruth imgs shape: " +str(gtruth_masks.shape))
    visualize(orig_imgs, path_experiment+"all_originals")
    visualize(pred_imgs, path_experiment+"all_predictions")
    visualize(gtruth_masks, path_experiment+"all_groundTruths")


def recompone_overlap(preds, img_h, img_w):  # stride_h = 136 if height==584 (strides for 565 = 117)
    # preds = preds.numpy()
    assert (len(preds.shape) == 4)
    assert (preds.shape[1] == 1 or preds.shape[1] == 3)  # check the channel is 1 or 3
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    stride_h = img_h - patch_h
    stride_w = img_w - patch_w
    N_patches_h = (img_h - patch_h) // stride_h + 1
    N_patches_w = (img_w - patch_w) // stride_w + 1
    N_patches_img = N_patches_h * N_patches_w
    print("N_patches_h: " + str(N_patches_h))
    print("N_patches_w: " + str(N_patches_w))
    print("N_patches_img: " + str(N_patches_img))
    assert (preds.shape[0] % N_patches_img == 0)
    N_full_imgs = preds.shape[0] // N_patches_img  # batch_size // 4
    print("According to the dimension inserted, there are " + str(N_full_imgs) + " full images (of " + str(
        img_h) + "x" + str(img_w) + " each)")
    full_prob = np.zeros(
        (N_full_imgs, preds.shape[1], img_h, img_w))  # itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))

    k = 0  # iterator over all the patches
    for i in range(N_full_imgs):
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                print(h * stride_h, (h * stride_h) + patch_h)
                full_prob[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w] += preds[k]
                full_sum[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w] += 1
                k += 1
    assert (k == preds.shape[0])
    assert (np.min(full_sum) >= 1.0)  # at least one
    final_avg = full_prob / full_sum
    print(final_avg.shape, np.max(final_avg))
    # assert (np.max(final_avg) <= 1.0)  # max value for a pixel is 1.0
    # assert (np.min(final_avg) >= 0.0)  # min value for a pixel is 0.0

    # plt.figure(figsize=(30, 30))
    # plt.imshow(final_avg[0, :, :, :].reshape(img_h, img_w), cmap="gray")
    # plt.show()
    # plt.close()
    return final_avg


def predict(model, params, test_dataset, test_masks, batch_size):
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    total = 0
    correct = 0
    model.eval()
    predictions = []
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(100, 100))
    # fig.suptitle('predicted_mask//original_mask')
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader),start=1):
            images, target = batch
            images = images.to(params["device"], dtype=torch.float, non_blocking=True)
            target = target.to(params["device"], dtype=torch.float, non_blocking=True)
            plt.figure()
            plt.hist(images.cpu().numpy().ravel(), bins=50, density=True)
            plt.xlabel("pixel values")
            plt.ylabel("relative frequency")
            plt.title("distribution of pixels")
            plt.show()

            output = model(images)
            batch_probs = torch.tanh(output)
            batch_probs = torch.sigmoid(batch_probs)
            ### sigmoid
            print(output.mean(), batch_probs.mean())
            print("Model Output shape: ", batch_probs.shape)
            predictions = batch_probs.cpu()
            print("Predictions Batch shape: ", predictions.shape)
            print("Target Batch shape: ", target.shape)
            mask = test_masks[batch_idx-1].reshape(1,565,565,1)
            print("Test Masks shape: ", mask.shape)
            
            ## Merge 4 overlapping patches predictions into 1 predictions image
            #avg_preds = average_predictions(predictions)
            avg_preds = recompone_overlap(predictions, 565, 565)
            avg_preds = np.moveaxis(avg_preds, 1, -1)
            print(avg_preds.shape)
            ## convert prediction probabilities to binary image
            #p = predictions[0].numpy().reshape(448,448)
            #t = target[0].numpy().reshape(448,448)
            threshold = optimal_thres_tuning(mask, avg_preds)
            preds_over, y_test = evaluation_metrics(avg_preds, mask, threshold)
            # Convert predictions to image and plot:
            pred2image(preds_over, y_test, batch_idx)

    return preds_over, y_test


class TestDataset(Dataset):
    def __init__(self, img, gt, transform):  # tfms used to label train or val dataset
        self.X = img
        self.y = gt
        self.transform = transform
        self.data_len = len(self.X)

    def __len__(self):
        return self.data_len

    def __getitem__(self, i):
        image = np.asarray(self.X[i]).astype(np.float32)
        mask = np.asarray(self.y[i]).astype(np.float32)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
        
        return image, torch.moveaxis(mask, -1, 0)

# Round off
def dict_round(dic,num):
    for key,value in dic.items():
        dic[key] = round(value,num)
    return dic


class Test():
    def __init__(self, params, patch_height, patch_width, test_original, test_masks, test_FOVs, path_experiment, i):
        self.batch_size = params["batch_size"]
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.model = params["model"]
        self.device = params["device"]
        self.i = i
        # save path
        self.path_experiment = path_experiment
                        
        self.patches_imgs_test, self.patches_gt_test, self.test_original, self.test_masks, self.test_FOVs = get_data_testing(
            test_original,
            test_masks,
            test_FOVs,
            patch_height,
            patch_width
        )
        # self.test_original = np.expand_dims(test_original, axis=1)
        self.test_original = np.moveaxis(test_original, -1, 1)
        self.test_masks = np.moveaxis(self.test_masks, -1, 1)
        self.test_FOVs = np.moveaxis(self.test_FOVs, -1, 1)
        self.img_height = self.test_masks.shape[2]
        self.img_width = self.test_masks.shape[3]
        test_transform = Compose([
            Resize(patch_height, patch_width, p=1),
            ToTensorV2(p=1)
        ])
        test_set = TestDataset(self.patches_imgs_test, self.patches_gt_test, transform=test_transform)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=params["num_workers"])

    # Inference prediction process
    def inference(self, net):
        net.eval()
        preds = []
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                inputs, target = batch
                inputs = inputs.to(self.device, dtype=torch.half, non_blocking=True)
                outputs = self.model(inputs)
                outputs = torch.sigmoid(outputs)
                outputs = outputs.data.cpu().numpy()
                preds.append(outputs)
        self.pred_patches = np.concatenate(preds, axis=0)
        print(self.pred_patches.shape)
        return self.pred_patches
        
    # Evaluate ate and visualize the predicted images
    def evaluate(self):
        self.pred_imgs = recompone_overlap(self.pred_patches, 565, 565)
        ## restore to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]

        #predictions only inside the FOV
        y_scores, y_true = pred_only_FOV(self.pred_imgs, self.test_masks, self.test_FOVs)

        eval = Evaluate(save_path=self.path_experiment, i=self.i)
        eval.add_batch(y_true, y_scores)
        log = eval.save_all_result(i=self.i, plot_curve=True,save_name="performance"+str(self.i)+".txt")     #self.i -> performance_metrics
        # save labels and probs for plot ROC and PR curve when k-fold Cross-validation
        np.save('{}/{}result.npy'.format(self.path_experiment, str(self.i)), np.asarray([y_true, y_scores]))
        return dict_round(log, 6)

    # save segmentation imgs
    def save_segmentation_result(self):

        kill_border(self.pred_imgs, self.test_FOVs) # only for visualization
        self.save_img_path = join(self.path_experiment,'result_img')
        if not os.path.exists(join(self.save_img_path)):
            os.makedirs(self.save_img_path)
        # self.test_imgs = my_PreProc(self.test_imgs) # Uncomment to save the pre processed image
        # for i in range(self.test_original.shape[0]):
        total_img = concat_result(self.test_original,self.pred_imgs,self.test_masks)
        save_img(total_img,join(self.save_img_path, "Result_image"+str(self.i)+".png"))

    # Val on the test set at each epoch
    def val(self):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, 565, 565)
        ## recover to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]

        #predictions only inside the FOV
        y_scores, y_true = pred_only_FOV(self.pred_imgs, self.test_masks, self.test_FOVs)
        eval = Evaluate(save_path=self.path_experiment, i=self.i)
        eval.add_batch(y_true, y_scores)
        confusion,accuracy,specificity,sensitivity,precision = eval.confusion_matrix()
        log = OrderedDict([('val_auc_roc', eval.auc_roc()),
                           ('val_f1', eval.f1_score()),
                           ('val_acc', accuracy),
                           ('SE', sensitivity),
                           ('SP', specificity)])
        return dict_round(log, 6)


if __name__ == '__main__':

    # ========= CONFIG FILE TO READ FROM =======
    config = configparser.RawConfigParser()
    config.read('./configuration.txt')
    # ===========================================
    # run the training on invariant or local
    path_data = config.get('data paths', 'path_local')
    path_drive = '../../data/DRIVE/'
    name_experiment = config.get('experiment name', 'name')
    # training settings
    N_epochs = int(config.get('training settings', 'N_epochs'))
    batch_size = int(config.get('training settings', 'batch_size'))
    patch_height = int(config.get('data attributes', 'patch_height'))
    patch_width = int(config.get('data attributes', 'patch_width'))
    inside_FOV = True

    image_folder = path_drive + '/test/images/'
    images = [img for img in os.listdir(image_folder) if img.endswith(".tif")]
    images = sorted(images, key=natural_sort_key)   #sorted

    gt_folder = path_drive + '/test/1st_manual/'
    gts = [img for img in os.listdir(gt_folder) if img.endswith(".gif")]
    gts = sorted(gts, key=natural_sort_key) #sorted

    mask_folder = path_drive + '/test/mask/'
    masks = [img for img in os.listdir(mask_folder) if img.endswith(".gif")]
    masks = sorted(masks, key=natural_sort_key) #sorted

    print(len(masks), len(gts), len(images))
    test_original, test_masks, test_FOVs = np.zeros(shape=(len(images), 584, 565), dtype=np.uint8), np.zeros(shape=(len(gts), 584, 565), dtype=np.uint8), np.zeros(shape=(len(masks), 584, 565), dtype=np.uint8)
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

    # dimension of the patches
    patch_height = int(config.get('data attributes', 'patch_height'))
    patch_width = int(config.get('data attributes', 'patch_width'))
    # # the stride in case output with average
    # stride_height = int(config.get('testing settings', 'stride_height'))
    # stride_width = int(config.get('testing settings', 'stride_width'))
    # assert (stride_height < patch_height and stride_width < patch_width)

    name_experiment = config.get('experiment name', 'name')
    path_experiment = './' + name_experiment + '/'
    # N full images to be predicted
    Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
    # Grouping of the predicted images
    N_visual = int(config.get('testing settings', 'N_group_visual'))

    # Load the images and divide them in patches
    patches_imgs_test = None
    new_height = None
    new_width = None
    masks_test = None
    patches_masks_test = None
    test_original = test_original[:, 10:575, :]
    test_masks = test_masks[:, 10:575, :]
    test_FOVs = test_FOVs[:, 10:575, :]

    # # up_points = (565, 565)
    # # scaled_up = np.array([cv2.resize(x.reshape(448,448), up_points) for x in patches_imgs_test])
    # # load scaler
    # import pickle as p
    # with open("scaler.pickle", "rb") as f:
    #     scaler = p.load(f, encoding='bytes')
    # arr_norm = scaler.transform(test_original.reshape(-1, 565*565))
    # test_original = arr_norm.reshape(-1, 565, 565)
    # # down_points = (448, 448)
    # # arr_norm = np.array([cv2.resize(x.reshape(565,565), down_points) for x in arr_norm])
    # # test_original = arr_norm.reshape(-1, 448, 448, 1)

    # ================ Run the prediction of the patches ==================================
    best_last = config.get('testing settings', 'best_last')

    layers = 15
    filters = 3
    
    check_path_best = 'SineNet_layer_%d_filter_%d_best_test_DiceBCEsigmoid.pt7' % (layers, filters)   # 1st optimizer
    check_path_cont = 'SineNet_layer_%d_filter_%d_continue_test_DiceBCEsigmoid.pt7' % (layers, filters)   # +2nd optimizer

    from sinenet_torch_BN import *

    model = Sine_Net(n_channels=1, n_classes=1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/' + check_path_cont, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state'])
    start_epoch = checkpoint['epoch'] + 1 

    # test_transform = Compose([
    #     Resize(patch_height, patch_width, p=1),
    #     ToTensorV2(p=1)
    # ])
    # test_set = TransfTestDataset(arr_norm, patches_gt_test, transform=test_transform)

    params = {
        "model_name": "Sine_Net",
        "device": device,
        "batch_size": 4,
        "num_workers": 2,
        "model": model
    }
    model.to(params["device"])
    for i in range(test_original.shape[0]):
        eval = Test(params, patch_height, patch_width, test_original[i], test_masks[i], test_FOVs[i], path_experiment, i)
        pred_patches = eval.inference(model)
        print(eval.evaluate())
        eval.save_segmentation_result()

    # predim01, ytest = predict(model, params, test_set, test_masks, batch_size=params["batch_size"])
    # visualization(predim01, test_imgs, test_masks)
