import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, h5py, gc
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# % matplotlib inline
from augmentation_l import adjust_gamma, scaling, random_rotation, horizontal_vertical_flip
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras import callbacks, utils
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model, to_categorical, Sequence
import configparser as cfp
from help_functions import *
from torch_models_setup import *
from extract_patches import get_data_training
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.nn import BCELoss, BCEWithLogitsLoss
from collections import defaultdict
from augmentation_l import dataset_normalized, dataset_whitening, dataset_standardized, dataset_whitening2


def step_decay(epoch):
    # the initial learning rate: lrate = 0.001
    if epoch % 6 == 0:
        return lrate * 0.1
    else:
        return lrate


def lr_scheduler(epoch, lr, step_decay=0.1):
    return float(lr * step_decay) if epoch % 6 == 0 else lr


def data_augmentation(image):
    image = scaling(image)
    image = random_rotation(image)
    image = horizontal_vertical_flip(image)
    image = adjust_gamma(image)
    return image

# =============================================
# Train Data Generator -> Yields a specific amount of train images with an uniform classes distribution.
# ...performing real-time data augmentation
# ImageDataGenerator for X images, will produce batch_size=256 images in each iteration = steps_per_epoch = samples/batch_size
# 30,000 iterations /1578 iterations per epoch = 19 epochs
# 10,000 iterations /1563 iterations per epoch = 6 epochs
# iteration := steps_per_epoch = total number of samples / batch_size = 400,000 / 256 = 1563

# If NOT mentioned "steps_per_epoch" in the training generator,
# the ImageDataGenerator generates different random augmented images for every batch in each epoch.
# Eg. if you have 500 images and batches=50, in every epoch, ImageDataGenerator generates 10 different augmented image series.
# resize_rescale = tf.keras.Sequential(
#     [layers.experimental.preprocessing.Resizing(54, 54),
#      layers.experimental.preprocessing.Rescaling(1./255)])


AUTOTUNE = tf.data.experimental.AUTOTUNE

# # ======================== METHOD 2: ===========================
# # Provide the same seed and keyword arguments to the fit and flow methods
# # Fit is required only if featurewise_center, featurewise_std_normalization or zca_whitening == True
# seed = 1
# # images & masks already preprocessed input
# image_datagen = ImageDataGenerator(images, batchSize=256, aug = augmentator(), binarize = True, classesNum=2)
# mask_datagen = ImageDataGenerator(masks, batchSize=256, aug = augmentator(), binarize = True, classesNum=2)
# image_datagen.fit(images, augment=True, seed=seed) # apply augmentation on data
# mask_datagen.fit(masks, augment=False, seed=seed)
#
# # flow_from_direcotry: Takes the path to the directory & generates batches of augmented data
# # image_datagen & mask_datagen = train & val data respectively (after augmentation)
#
# # Combine generators into one which yields image and masks
# train_generator = zip(image_generator, mask_generator)
# model.fit(image_generator, epochs=19, validation_data=mask_generator, shuffle=True, callbacks=callbacks_list)
#
# # flow_from_directory(directory, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, subset=None, interpolation='nearest')


# ========================== METHOD 3 ============================
import torch, torchvision
import torch.nn as nn
import random, cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import transforms, Compose, functional
import models_setup as M
import skimage

# class RandomGammaCorrection(object):
#     def __init__(self, gamma):
#         self.gamma = gamma
#
#     def __call__(self, image):
#         random_gamma = random.uniform(0.25, 4)
#         self.gamma = random_gamma
#         print(self.gamma)
#         return transforms.functional.adjust_gamma(image, self.gamma, gain=1)


# Compose an augmentation pipeline
def aug(p=1):
    return A.Compose([A.RandomScale(scale_limit=(0.7, 1.2), interpolation=cv2.INTER_NEAREST, p=1),
                      A.Rotate(limit=90, interpolation=cv2.INTER_NEAREST, p=1),
                      A.HorizontalFlip(),
                      A.VerticalFlip(),
                      A.RandomGamma(gamma_limit=(0.25, 4), p=1),
                      ToTensorV2()])


# image dataset module for GCN/ZCA Preprocessing
class NormImageDataset(Dataset):
    def __init__(self, imgs, transf=None):
        self.X = imgs
        self.transform = transf

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        image = self.X[i]
        if self.transform == dataset_normalized:
            transformed = self.transform(image)
            print(transformed.shape)
            return transformed


class ZCAImageDataset(Dataset):
    def __init__(self, imgs, transf=None):
        self.X = imgs
        self.transform = transf

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        images_list = self.X
        if self.transform == dataset_whitening:
            transformed = self.transform(images_list)
            return transformed


# image dataset module for Augmentation pipeline
class AlbumentationImageDataset(Dataset):  # or Sequence
    def __init__(self, img, gt, tfms):   # tfms used to label train or val dataset
        self.X = img
        self.y = gt
        self.tfms = tfms
        # apply 2 sets of image augmentation, one for validation and the other one for training.
        if self.tfms == 0:  # if validating
            self.transform = ToTensorV2() # A.Resize(27, 27, p=1)
        else:  # if training
            self.transform = A.Compose([A.RandomScale(scale_limit=(0.7, 1.2), interpolation=cv2.INTER_NEAREST, p=1),
                        A.Rotate(limit=90, interpolation=cv2.INTER_NEAREST, p=1),
                        A.OneOf([A.HorizontalFlip(), A.VerticalFlip()]),
                        A.RandomGamma(gamma_limit=(25, 400), p=1),
                        A.Resize(27, 27, p=1),
                        ToTensorV2()])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        image = self.X[i]
        mask = self.y[i]  # get the label of the image
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = np.transpose(transformed["mask"], (2, 0, 1))
        return image, mask


# =============== MODEL CONFIGURATION ==============
layers = 9
filters = 3
check_path_best = 'PLAIN_net_layer_%d_filter_%d.pt7'% (layers,filters)
check_path_cont = 'PLAIN_net_layer_%d_filter_%d.pt7'% (layers,filters)
best_loss = np.Inf

# Track metrics such as accuracy or loss during training and validation.
class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]  # create new metric "metric_name"

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


# def calculate_acc(output, target):
    # output = output >= 0.5
    # target = target == 1.0
    # return torch.true_divide((target == output).sum(dim=0), output.size(0))


def multi_acc(y_pred, y_test):
    _, y_pred = torch.max(y_pred, dim = 1, keepdim=True)
    _, y_test = torch.max(y_test.data, 1, keepdim=True)
    correct_pred = (y_pred == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    
    return acc


# Probably check different thresholds for evaluating the probabilities results
# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('float')


def calculate_acc(output, target):

    from sklearn import metrics
    # target = target.cpu().numpy()
    # predicted = output.data.cpu().numpy()
    # # evaluate each threshold
    # pos_probs = predicted[:, 1]   # keep probabilities for the positive outcome only
    # thresholds = np.arange(0, 1, 0.01)
    # scores = [metrics.accuracy_score(target[:, 1], to_labels(pos_probs, t)) for t in thresholds]
    # # get best threshold
    # ix = np.argmax(scores)
    # print('Threshold=%.3f, Acc-Score=%.5f' % (thresholds[ix], scores[ix]))
    
    _, predicted = torch.max(output.data, 1, keepdim=True)
    _, target = torch.max(target, 1, keepdim=True)
    target = target.cpu()
    predicted = predicted.cpu()
    accuracy2 = metrics.accuracy_score(target, predicted)
    pos_probs = output[:, 1].cpu().detach().numpy()
    # roc_auc = metrics.roc_auc_score(target, pos_probs)
    thresholds = np.arange(0, 1, 0.01)
    scores = [metrics.accuracy_score(target, to_labels(pos_probs, t)) for t in thresholds]
    # get best threshold
    ix = np.argmax(scores)
    print('Threshold=%.3f, Acc-Score=%.5f' % (thresholds[ix], scores[ix]))
    # print("\nBatch roc auc: ", roc_auc)

    return accuracy2


def create_model(params):
    import models_setup as M

    batch_size, C, H, W = int(params["batch_size"]), 3, 27, 27

    pytorch_model = PLAIN_net(C, 2)  # initialize with #channels & #classes
    if params["model_name"] == 'NOPOOL_net':
        pytorch_model = NOPOOL_net(C, 2)
    elif params["model_name"] == 'NOPOOL_SP_5':
        pytorch_model = NOPOOL_SP_5(C, 2)
    
    # # Also useful: will only print those layers with params
    # state_dict = pytorch_model.state_dict()
    # torch.save(state_dict, './f{rbvs}_plainnet_best_model.pth')
    # # print(util.state_dict_layer_names(state_dict))

    model = pytorch_model.to(params["device"])
   
    return model


def train(train_loader, model, criterion, optimizer, epoch, params, train_losses, train_corrects_history):
    metric_monitor = MetricMonitor()  # initialize (reset) metrics to zero
    model.train()
    train_epoch_acc = 0
    stream = tqdm(train_loader)
    for batch_idx, (images, target) in enumerate(stream, start=1):
        images = images.to(params["device"], dtype=torch.float, non_blocking=True)  # for GPU usage: These 2 lines move imgs and labels to the device we are training on
        target = target.to(params["device"], dtype=torch.float, non_blocking=True)
        target = target[:,:,-1,-1]

        output = model(images)
        output_s = torch.sigmoid(output)
        # print(output_s)
        loss = criterion(output_s, target)
        metric_monitor.update("Train_loss", loss.item())
        train_epoch_acc += calculate_acc(output_s, target)
        # train_epoch_acc += multi_acc(output_s, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train. {metric_monitor}. Accuracy. {accuracy}".format(epoch=epoch, metric_monitor=metric_monitor, accuracy=train_epoch_acc)
        )
    epoch_loss = metric_monitor.metrics['Train_loss']['val'] / len(train_loader)
    epoch_acc = train_epoch_acc / len(train_loader)  # avg_accuracy
    train_losses.append(epoch_loss)
    train_corrects_history.append(epoch_acc)
    print("Epoch training loss | accuracy: ", epoch_loss, " | ", epoch_acc)
    print("Epoch: {}. Learning rate: {}".format(epoch, optimizer.param_groups[0]['lr']))
    
    gc.collect()
    torch.cuda.empty_cache()

    return train_losses, train_corrects_history


def validate(val_loader, model, criterion, epoch, params, val_losses, val_corrects_history):
    global best_loss
    metric_monitor = MetricMonitor() # initialize (reset) metrics to zero
    train_epoch_acc = 0
    model.eval()  # affects certain model layers to be used during model inference
    stream = tqdm(val_loader)
    with torch.no_grad():  # no_grad: turn off gradients computation/usage during eval time -> speed up / reduce memory
        for batch_idx, (images, target) in enumerate(stream, start=1):
            images = images.to(params["device"], dtype=torch.float, non_blocking=True)  # for GPU usage: These 2 lines move imgs and labels to the device we are training on
            target = target.to(params["device"], dtype=torch.float, non_blocking=True)
            target = target[:, :, -1, -1]
            # print(target.shape)

            output = model(images)
            output_s = torch.sigmoid(output)
            # print(output_s)
            loss = criterion(output_s, target)
            metric_monitor.update("Val_loss", loss.item())
            train_epoch_acc += calculate_acc(output_s, target)
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}. Accuracy. {accuracy}".format(epoch=epoch, metric_monitor=metric_monitor, accuracy=train_epoch_acc)

            )

    # epoch loss = running loss accumulation / data length
    epoch_loss = metric_monitor.metrics['Val_loss']['val'] / len(val_loader)
    epoch_acc = train_epoch_acc / len(val_loader)  # avg_accuracy (OR accuracy)
    val_losses.append(epoch_loss)
    val_corrects_history.append(epoch_acc)
    # Save checkpoint for best validation loss model.
    print("Epoch validation loss/accuracy: ", epoch_loss, " / ", epoch_acc)
    if metric_monitor.metrics['Val_loss']['val'] < best_loss: # the accumulated loss
        print('Saving..')
        state = {
            'model_state': model.state_dict(),
            'best_loss': best_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + check_path_best)
        best_loss = metric_monitor.metrics['Val_loss']['val']

    gc.collect()
    torch.cuda.empty_cache()

    return val_losses, val_corrects_history


def train_and_validate(model, train_dataset, val_dataset, params, train_losses, val_losses, train_corrects_history, val_corrects_history):
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    criterion = nn.BCELoss().to(params["device"]) # BCEWithLogitsLoss or BCELoss
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9, nesterov=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1, last_epoch=-1, verbose=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=.1, verbose=True)
    for epoch in range(1, params["epochs"] + 1):
        train_losses, train_corrects_history = train(train_loader, model, criterion, optimizer, epoch, params, train_losses, train_corrects_history)
        val_losses, val_corrects_history = validate(val_loader, model, criterion, epoch, params, val_losses, val_corrects_history)
        scheduler.step(val_losses[epoch-1])
    print('Saving after number of epochs:', epoch)
    state = {
        'model_state': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': val_losses,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + check_path_cont)

    return model, train_losses, val_losses, train_corrects_history, val_corrects_history



if __name__ == '__main__':

    # ========= Load settings from Config file
    path_data = '../../data/DRIVE_training_testing/Augmented_Images/'  # path to augmented data
    # Experiment name
    name_experiment = 'test_all_plain'
    config = cfp.RawConfigParser()
    config.read('configuration.txt')
    path_data_local = config.get('data paths', 'path_local')
    name_experiment = config.get('experiment name', 'name')
    # training settings
    N_epochs = int(config.get('training settings', 'N_epochs'))
    SP = config.get('training settings', 'SP')
    N_subimgs = int(config.get('training settings', 'N_subimgs'))
    batch_size = int(config.get('training settings', 'batch_size'))

    # Callbacks list:
    checkpointer = ModelCheckpoint(filepath='./' + name_experiment + '/' + name_experiment + '_best_weights.h5',
                                   verbose=1,
                                   monitor='val_loss', mode='auto', save_best_only=True)
    lrate_drop = LearningRateScheduler(lr_scheduler, verbose=1)

    callbacks_list = [checkpointer, lrate_drop]

    # LOAD DATA
    print('extracting patches')
    data_path = '../../data/DRIVE_training_testing/'
    DRIVE_train_imgs_original = data_path + 'init_img_train.hdf5'
    DRIVE_train_groundTruth = data_path + 'init_gtruth_train.hdf5'
    patch_height = 27
    patch_width = 27
    N_subimgs = N_subimgs  # 20 * 20,000 random patches per training image
    inside_FOV = True

    patches_imgs_train, patches_gt_train, windows = get_data_training(
        DRIVE_train_imgs_original,
        DRIVE_train_groundTruth,
        patch_height,
        patch_width,
        N_subimgs,
        inside_FOV,  # select the patches only inside the FOV  (default == True)
        SP) # SP is True or False for structured prediction process

    # patches_gt_train = utils.to_categorical(patches_gt_train, num_classes=None)  # alternative for the masks_net function call
    from help_functions import masks_net
    if SP==True:
        patches_gt_train = masks_net(windows)
        patches_gt_train = patches_gt_train.reshape((N_subimgs, 5, 5, 2))
    else:
        patches_gt_train = masks_net(patches_gt_train)
        patches_gt_train = patches_gt_train.reshape((N_subimgs, 27, 27, 2))
    # patches_gt_train = (patches_gt_train == 1).astype('float32') # convert all the -1 labels to zeros
    # patches_gt_train = patches_gt_train.reshape(-1, 2) # Reshape the labels to have a shape of (n_samples, 1)
    print(patches_gt_train.shape)

    # Preprocess in batches
    print("normalization\n")
    for step in range(0, patches_imgs_train.shape[0], batch_size):
        patches_imgs_train[step:step+batch_size] = np.asarray(NormImageDataset(patches_imgs_train[step:step+batch_size], dataset_normalized))
    
    print("zca\n")
    patches_imgs_train, zca_matrix = np.asarray(dataset_whitening(patches_imgs_train))
    import pickle as p
    with open("zca.pickle", "wb") as f:
        p.dump(zca_matrix, f)

    print(patches_imgs_train.shape)

    print("Train test split: \n")
    (xtrain, xval, ytrain, yval) = train_test_split(patches_imgs_train, patches_gt_train, test_size=0.1, random_state=42)
    # scaler = MinMaxScaler()
    # xtrain = scaler.fit_transform(xtrain)
    # xval = scaler.transform(xval)
    # # xtest = scaler.transform(xtest)

    train_data = AlbumentationImageDataset(xtrain, ytrain, 1) # train_data  tfms=1
    val_data = AlbumentationImageDataset(xval, yval, 0) # val_data  tfms=0

    print('Training')
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available. Training on CPU...')
        device = torch.device('cpu')
    else:
        print('CUDA is available. Training on GPU...')
        device = torch.device('cuda:0')

    if SP==True: 
        model_name = "NOPOOL_SP_5"
    else: 
        model_name = "PLAIN_net"
    params = {
        "model_name": model_name,
        "device": device,
        "batch_size": batch_size,
        "num_workers": 4,
        "epochs": N_epochs
    }
    model = create_model(params)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0.0, 0.001)
            torch.nn.init.zeros_(m.bias)

    model.apply(weights_init)

    resume = False
    if device == 'cuda':
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/' + check_path_cont)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']

    print(f"Training on device {device}.")
    # Initialization of total epochs metrics
    train_losses, val_losses = [], []   # Training Loss & Validation Loss history
    train_corrects_history, val_corrects_history = [], []   # Accuracy history
    model, train_losses, val_losses, train_corrects_history, val_corrects_history = train_and_validate(model, train_data, val_data, params, train_losses,
                                                                                                       val_losses, train_corrects_history, val_corrects_history)

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, color='green', label='train loss')
    plt.plot(val_losses, color='blue', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # create a folder for the results
    result_dir = 'outputs'
    print("\n1. Create directory for the results (if not already existing)")
    if os.path.exists(result_dir):
        print("Dir already existing")
    else:
        os.system('mkdir ' + result_dir)
    plt.savefig('./outputs/loss.png')
    plt.show()

    # Acc plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_corrects_history, color='green', label='train acc')
    plt.plot(val_corrects_history, color='blue', label='validation acc')
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
    plt.savefig('./outputs/accuracy.png')
    plt.show()