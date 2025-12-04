from sinenet_torch_BN import *
from extract_patches import *
from train_retina import *
from albumentations import OneOf, Rotate, Flip, HorizontalFlip, VerticalFlip, Compose, CenterCrop, Affine, Normalize, Resize
from albumentations.pytorch import ToTensorV2
import os, h5py, gc, cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import configparser as cfp
from sinenet_torch_BN import *
from sklearn.model_selection import train_test_split
from torch.nn import BCELoss, BCEWithLogitsLoss
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from extract_patches import *
from torch.utils.data.sampler import Sampler
from torch import Tensor
import psutil
from losses import *
from albumentations import OneOf, Rotate, Flip, HorizontalFlip, VerticalFlip, Compose, CenterCrop, Affine, Normalize, Resize
from albumentations.pytorch import ToTensorV2
import re, imageio
from sklearn.preprocessing import StandardScaler, MinMaxScaler


layers = 15
filters = 3
check_path_best = 'SineNet_layer_%d_filter_%d_best_test_TverskyLoss.pt7'% (layers,filters)
check_path_cont = 'SineNet_layer_%d_filter_%d_continue_test_TverskyLoss.pt7'% (layers,filters)
best_loss = np.Inf
import sys, random
np.set_printoptions(threshold=sys.maxsize)


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [
        int(text)
        if text.isdigit() else text.lower()
        for text in _nsre.split(s)]


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
    
    def _get_label(self, dataset, idx, labels = None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max*len(self.keys)


def calculate_acc(output, target, total, correct):
    
    predicted = (output >= 0.5)#.float()
    total += target.nelement()
    correct += predicted.eq(target.data).sum().item()
    accuracy = (correct / total)
    return accuracy

# Track metrics such as accuracy or loss during training and validation.
class MetricMonitor:
    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "sum": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, n):
        metric = self.metrics[metric_name]  # create new metric "metric_name"
        metric["val"] = val
        metric["sum"] += val * n
        metric["count"] += n
        metric["avg"] = metric["sum"] / metric["count"]


    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


class AugImageDataset(Dataset):
    def __init__(self, img, gt, transform=None):  # tfms used to label train or val dataset
        self.X = img
        self.y = gt
        # After 1 random image is generated, we augment it to 8 with rotations (-90°, 90°) and flipping operations
        self.transform = transform
        self.data_len = len(self.X)

    def __len__(self):
        return self.data_len

    def __getitem__(self, i):
        # # Tensor of shape C x H x W or numpy ndarray of shape H x W x C
        image = self.X[i].astype(np.float32)
        mask = self.y[i].astype(np.float32)

        if self.transform is not None:
            
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
            return (torch.from_numpy(np.moveaxis(image, -1, 0)), torch.from_numpy(np.moveaxis(mask, -1, 0)))

        else:
            image, mask = torch.from_numpy(image), torch.from_numpy(mask)
            return (torch.moveaxis(image, -1, 0).to(torch.float32), torch.moveaxis(mask, -1, 0).to(torch.float32))

class Sine:

    def __init__(self):
        self.optimizer = None
        self.model = None
        self.train_data = None
        self.val_data = None
        self.criterion = None
        self.val_loader = None
        self.train_original = None
        self.train_masks = None
        self.train_FOVs = None
        self.name_experiment = None
        self.path_experiment = None
        self.patch_height = None
        self.patch_width = None
        self.batch_size = None
        self.N_epochs = None
        self.train_loader = None
        self.data_path = None
        self.pred_patches = []

    def loadall(self):
        config = cfp.RawConfigParser()
        config.read('configuration.txt')
        self.data_path = config.get('data paths', 'path_local')
        path_drive = '../../data/DRIVE/'
        self.name_experiment = config.get('experiment name', 'name')
        # training settings
        self.N_epochs = int(config.get('training settings', 'N_epochs'))
        self.batch_size = int(config.get('training settings', 'batch_size'))

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

        print(train_original.dtype, train_masks.dtype, train_original.max(), train_masks.max())

        del gts, images, gif_reader
        gc.collect()
        return train_original, train_masks, train_FOVs, inside_FOV

    def load_dataS(self):
        train_original, train_masks, train_FOVs, inside_FOV = self.loadall()
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
        print("End of extraction and preprocessing\n")
        print(self.train_original.mean(), self.train_masks.mean())
        print("Train test split for SINE: \n")
        del train_original, train_masks, train_FOVs


    def train(self, epoch, params, train_losses, train_corrects_history):
        metric_monitor = MetricMonitor()  # initialize (reset) metrics to zero
        self.model.train()
        self.train_loader = tqdm(self.train_loader)
        total_train = 0
        correct_train = 0
        for batch_idx, (images, target) in enumerate(self.train_loader, start=1):
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

        metric_monitor = MetricMonitor()  # initialize (reset) metrics to zero
        self.model.eval()  # affects certain model layers to be used during model inference
        self.val_loader = tqdm(self.val_loader)
        total_val = 0
        correct_val = 0
        with torch.no_grad():  # no_grad: turn off gradients computation/usage during eval time -> speed up / reduce memory
            for batch_idx, (images, target) in enumerate(self.val_loader, start=1):
                images = images.to(params["device"], dtype=torch.half,
                                   non_blocking=True)  # for GPU usage: move to the device we are training on
                target = target.to(params["device"], dtype=torch.half, non_blocking=True)
                # print(target.shape)

                output_s = self.model(images)
                self.pred_patches.append(output_s.data.cpu().numpy())
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


    def train_and_validate(self, params, train_losses, val_losses, train_corrects_history, val_corrects_history):
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
            pin_memory=False,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_data,
            batch_size=params["batch_size"],
            shuffle=False,
            num_workers=params["num_workers"],
            pin_memory=False,
        )
        # criterion = nn.BCEWithLogitsLoss().to(params["device"])
        self.criterion = WBCE_DiceLoss().to(params["device"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=20, factor=.1, verbose=True)
        print(f'Starting epoch: {params["start_epoch"]}, - Amount of epochs to train: {params["epochs"]}')
        ## Epochs Iteration
        best_loss = float("inf")
        for epoch in range(params["start_epoch"], params["start_epoch"] + params["epochs"]):
            # enumerate mini-batches
            train_losses, train_corrects_history = self.train(epoch, params, train_losses, train_corrects_history)
            val_losses, val_corrects_history, val_loss_ep = self.validate(epoch, params, val_losses, val_corrects_history)
            scheduler.step(val_losses[epoch - 1])
            # Save checkpoint for best validation loss model.
            if val_loss_ep < best_loss:  # the accumulated loss
                print('Saving..')
                best_loss = val_loss_ep
                state = {
                    'model_state': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
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
        return train_losses, val_losses, train_corrects_history, val_corrects_history


    def Sinenet_train(self, x_all, y_all, indices_train, indices_test):

        # 2 augmentation steps
        augmented_path = self.data_path + 'Augmented_Images/'
        new_file_path = augmented_path + 'augmented_img_train'
        new_file_path_mask = augmented_path + 'augmented_gt_train'
        orig_path = augmented_path + 'img_train'
        orig_path_mask = augmented_path + 'gt_train'

        xtrain_aug, ytrain_aug = [], []
        from augmentation import aug_apply
        aug_apply(x_all, y_all, indices_train, self.patch_height, self.data_path, 1)   # step 1 of augmentations

        # load saved images after step 1:
        for i in indices_train:
            xtrain_aug.append(torch.load(new_file_path + str(i) + '_0.pt'))
            ytrain_aug.append(torch.load(new_file_path_mask + str(i) + '_0.pt'))

        aug_apply(x_all, y_all, indices_train, self.patch_height, self.data_path, 2)
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
            Resize(self.patch_height, self.patch_width, p=1)
        ])
        val_transform = Compose([
            Resize(self.patch_height, self.patch_width, p=1)
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
        # self.model.load_state_dict(checkpoint['model_state'])
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
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
                layer.float().half()

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
        train_losses, val_losses, train_corrects_history, val_corrects_history = self.train_and_validate(params, train_losses, val_losses,
        train_corrects_history, val_corrects_history)
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
        self.optimizer = optimizer2
        # Initialization of total epochs metrics
        train_losses, val_losses, train_corrects_history, val_corrects_history = self.train_and_validate(params, train_losses, val_losses, 
        train_corrects_history, val_corrects_history)

        # # Evaluate for k-fold for sef.val_data predictions (not binary images)
        # # Separate tuple val_data to X and y
        # for i in range(len(self.val_data)):
        #     sample = self.val_data[i]
        #     val_img[i] = sample['image']
        #     val_mask[i] = sample['mask']

        #     eval = Test(params, 448, 448, val_img[i], val_mask[i], fov_val[i], path_experiment, i)
        #     eval.inference(self.model)  ## computes self.pred_patches

        del self.train_data, self.val_data
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

        return self.pred_patches