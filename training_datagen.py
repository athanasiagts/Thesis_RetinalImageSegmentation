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
from torch.utils.data.sampler import WeightedRandomSampler

layers = 15
filters = 3
check_path_best = 'SineNet_layer_%d_filter_%d_best_test_TverskyLoss.pt7'% (layers,filters)
check_path_cont = 'SineNet_layer_%d_filter_%d_continue_test_TverskyLoss.pt7'% (layers,filters)
best_loss = np.Inf
import sys, random
np.set_printoptions(threshold=sys.maxsize)

def add_sample_weights(train_data, batch_size):
    
    labels = []
    for i in range(0, 180):
        labels.append(train_data[i][1].flatten())
    labels = np.array(torch.stack(labels))
    labels = labels.reshape(labels.shape[0]*labels.shape[1])

    pos = sum(labels)    # sum of 1s in labels
    total = len(labels.flatten())
    neg = total - pos
    w0 = (1.0 / neg) * (total / 2.0)
    w1 = (1.0 / pos) * (total / 2.0)
    class_weights = [w0, w1]

    # class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    # weight = 1. / class_sample_count
    # sample_weights = np.array([weight[int(t)] for t in labels])

    # Create an image of `sample_weights` by using the label at each pixel as an index into the `class weights` .
    sample_weights = [0] * total
    sample_weights = np.array([w1*x if x==1.0 else w0 for x in labels])

    print(sample_weights.shape)
    return torch.from_numpy(sample_weights)


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
    correct += predicted.eq(target.data).sum()  #.item()
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


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

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
            return (image.to(torch.float32), torch.moveaxis(mask, -1, 0).to(torch.float32))

        else:
            image, mask = torch.from_numpy(image), torch.from_numpy(mask)
            return (torch.moveaxis(image, -1, 0).to(torch.float32), torch.moveaxis(mask, -1, 0).to(torch.float32))


def train(train_loader, model, criterion, optimizer, epoch, params, train_losses, train_corrects_history):
    metric_monitor = MetricMonitor()  # initialize (reset) metrics to zero
    model.train()
    train_loader = tqdm(train_loader)
    total_train = 0
    correct_train = 0
    for batch_idx, (images, target) in enumerate(train_loader, start=1):
        images = images.to(params["device"], dtype=torch.half, non_blocking=True)  # for GPU usage: move the device we are training on
        target = target.to(params["device"], dtype=torch.half, non_blocking=True)
        optimizer.zero_grad()   # clear the last error gradient

        # plt.figure()
        # plt.hist(images.cpu().numpy().ravel(), bins=50, density=True)
        # plt.xlabel("pixel values")
        # plt.ylabel("relative frequency")
        # plt.title("distribution of pixels")
        # plt.savefig("image_distrib.png")

        output_s = model(images)
        ##output_s = torch.sigmoid(output_s)
        loss = criterion(output_s, target)
        
        metric_monitor.update("Train_loss", loss.item(), images.size(0))
        accuracy = calculate_acc(output_s, target, total_train, correct_train)
        metric_monitor.update("Train_acc", accuracy, images.size(0))
        loss.backward()     # backccopagate the error through the model
        optimizer.step()    # update model weights
        train_loader.set_description(
            "Epoch: {epoch}. Train. {metric_monitor}.".format(epoch=epoch, metric_monitor=metric_monitor)
        )
    
    epoch_loss = metric_monitor.metrics['Train_loss']['avg']
    epoch_acc = metric_monitor.metrics['Train_acc']['avg']
    train_losses.append(epoch_loss)
    train_corrects_history.append(epoch_acc)
    print("Epoch training loss/accuracy: ", epoch_loss, " / ", epoch_acc)
    print("Epoch: {}. Learning rate: {}".format(epoch, optimizer.param_groups[0]['lr']))
   
    return train_losses, train_corrects_history


def validate(val_loader, model, criterion, epoch, params, val_losses, val_corrects_history):
    global best_loss
    metric_monitor = MetricMonitor() # initialize (reset) metrics to zero
    model.eval()  # affects certain model layers to be used during model inference
    val_loader = tqdm(val_loader)
    total_val = 0
    correct_val = 0
    accsum = 0
    with torch.no_grad():  # no_grad: turn off gradients computation/usage during eval time -> speed up / reduce memory
        for batch_idx, (images, target) in enumerate(val_loader, start=1):
            images = images.to(params["device"], dtype=torch.half, non_blocking=True)  # for GPU usage: move to the device we are training on
            target = target.to(params["device"], dtype=torch.half, non_blocking=True)
            # print(target.shape)

            output_s = model(images)
            ##output_s = torch.sigmoid(output_s)
            loss = criterion(output_s, target)

            metric_monitor.update("Val_loss", loss.item(), images.size(0))
            accuracy = calculate_acc(output_s, target, total_val, correct_val)
            metric_monitor.update("Val_acc", accuracy, images.size(0))
            val_loader.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}.".format(epoch=epoch, metric_monitor=metric_monitor)
            )

    # epoch loss = running loss accumulation / data length
    epoch_loss = metric_monitor.metrics['Val_loss']['avg']
    epoch_acc =  metric_monitor.metrics["Val_acc"]['avg']
    val_losses.append(epoch_loss)
    val_corrects_history.append(epoch_acc)
    print("Epoch validation loss/accuracy: ", epoch_loss, " / ", epoch_acc)

    return val_losses, val_corrects_history, metric_monitor.metrics['Val_loss']['sum']


def train_and_validate(model, train_dataset, val_dataset, sampler, params, optimizer, train_losses, val_losses, train_corrects_history, val_corrects_history):
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        pin_memory=True,
        # sampler = sampler,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    #criterion = nn.BCEWithLogitsLoss().to(params["device"])
    criterion = WBCE_DiceLoss().to(params["device"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=.1, verbose=True)
    print(f'Starting epoch: {params["start_epoch"]}, - Amount of epochs to train: {params["epochs"]}')
    ## Epochs Iteration
    best_loss = float("inf")
    for epoch in range(params["start_epoch"], params["start_epoch"]+params["epochs"]):
        # enumerate mini-batches
        train_losses, train_corrects_history = train(train_loader, model, criterion, optimizer, epoch, params, train_losses, train_corrects_history)
        val_losses, val_corrects_history, val_loss_ep = validate(val_loader, model, criterion, epoch, params, val_losses, val_corrects_history)
        scheduler.step(val_losses[epoch-1])
                # Save checkpoint for best validation loss model.
        if val_loss_ep < best_loss: # the accumulated loss
            print('Saving..')
            best_loss = val_loss_ep
            state = {
                'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
                'epoch': epoch
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/' + check_path_best)

    print('Saving last model after number of epochs:', epoch)
    state = {
        'model_state': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': val_losses[-1],
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    print("checkpoint saved\n")
    torch.save(state, './checkpoint/' + check_path_cont)
    return model, train_losses, val_losses, train_corrects_history, val_corrects_history


if __name__ == '__main__':

    # ========= Load settings from Config file
    # Experiment name
    name_experiment = 'sine_net'
    config = cfp.RawConfigParser()
    config.read('configuration.txt')
    data_path = config.get('data paths', 'path_local')
    path_drive = '../../data/DRIVE'
    name_experiment = config.get('experiment name', 'name')
    # training settings
    N_epochs = int(config.get('training settings', 'N_epochs'))
    batch_size = int(config.get('training settings', 'batch_size'))

    # LOAD DATA
    print('extracting patches')
    DRIVE_train_imgs_original = data_path + 'init_img_train.hdf5'
    DRIVE_train_groundTruth = data_path + 'init_gtruth_train.hdf5'
    patch_height = int(config.get('data attributes', 'patch_height'))
    patch_width = int(config.get('data attributes', 'patch_width'))
    inside_FOV = True

    image_folder = path_drive + '/training/images/'
    images = [img for img in os.listdir(image_folder) if img.endswith(".tif")]
    images = sorted(images, key=natural_sort_key)   #sorted

    gt_folder = path_drive + '/training/1st_manual/'
    gts = [img for img in os.listdir(gt_folder) if img.endswith(".gif")]
    gts = sorted(gts, key=natural_sort_key) #sorted

    train_original, train_masks = np.zeros(shape=(len(images), 584, 565), dtype=np.uint8), np.zeros(shape=(len(gts), 584, 565), dtype=np.uint8)
    for i in range(len(images)):
        train_original[i] = cv2.imread(os.path.join(image_folder, images[i]), 0)
        gif_reader = imageio.get_reader(os.path.join(gt_folder, gts[i]))
        gif_length = gif_reader.get_length()
        frame_index = 0
        if frame_index < gif_length:
            train_masks[i] = gif_reader.get_data(frame_index)

    img_h = train_original.shape[1]
    img_w = train_original.shape[2]
    print(train_original.dtype, train_masks.dtype, train_original.max(), train_masks.max())

    del gts, images, gif_reader
    gc.collect()

    ### patches_imgs_train, patches_gt_train = train_original, train_masks
    train_original, train_masks = get_data_training(
        train_original,
        train_masks,
        patch_height,
        patch_width,
        inside_FOV)  # select the patches only inside the FOV  (default == True)

    # Preprocess in batches
    # MTHT ENHANCEMENT + CLAHE
    train_original = (train_original / 255.).astype(np.float32)
    train_masks = (train_masks / 255.).astype(np.float32)
    print("End of extraction and preprocessing\n")
    print(train_original.mean(), train_masks.mean())
    print("Train test split: \n")
    ## number of patches equal the number of images
    indices = range(20)
    xtrain, xval, ytrain, yval, indices_train, indices_test = train_test_split(train_original, train_masks,
                                                                               indices, test_size=0.1, random_state=42)
    print(indices_train, indices_test)
    indices_list = [x + 1 for x in indices_train]

    # 2 augmentation steps
    augmented_path = data_path + 'Augmented_Images/'
    new_file_path = augmented_path + 'augmented_img_train'
    new_file_path_mask = augmented_path + 'augmented_gt_train'
    orig_path = augmented_path + 'img_train'
    orig_path_mask = augmented_path + 'gt_train'

    xtrain_aug, ytrain_aug = [], []
    from augmentation import aug_apply
    aug_apply(train_original, train_masks, indices_train, patch_height, data_path, 1)   # step 1 of augmentations
   
    # load saved images after step 1:
    for i in indices_train:
        xtrain_aug.append(torch.load(new_file_path + str(i) + '_0.pt'))
        ytrain_aug.append(torch.load(new_file_path_mask + str(i) + '_0.pt'))
    
    ####
    aug_apply(train_original, train_masks, indices_train, patch_height, data_path, 2)

    # load saved images after step 2:
    for i in indices_train:
        for j in range(1,9):
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
        Affine(translate_px=(0,117), p=0.5, interpolation=cv2.INTER_LINEAR),
        Rotate(limit=90, p=0.5, interpolation=cv2.INTER_LINEAR),
        OneOf([VerticalFlip(), HorizontalFlip()], p=0.5),
        Resize(patch_height, patch_width, p=1),
        ToTensorV2(p=1)
    ])

    val_transform = Compose([
        Resize(patch_height, patch_width, p=1),
        ToTensorV2(p=1)
    ])
 
    scaler = StandardScaler()
    arr_norm = scaler.fit_transform(xtrain_aug.reshape(-1, 565*565))
    arr_normv = scaler.transform(xval.reshape(-1, 565*565))
    
    # save scaler for test data
    import pickle as p
    with open("scaler.pickle", "wb") as f:
        p.dump(scaler, f)

    plt.figure()
    plt.hist(arr_norm.ravel(), bins=50, density=True)
    plt.xlabel("pixel values")
    plt.ylabel("relative frequency")
    plt.title("distribution of pixels")
    plt.show()

    train_data = AugImageDataset(arr_norm.reshape(-1, 565, 565, 1), ytrain_aug, train_transform)
    val_data = AugImageDataset(arr_normv.reshape(-1, 565, 565, 1), yval, val_transform)
    #train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size)
    #val_dataloader = DataLoader(dataset=val_data, batch_size=batch_size)
    #train_mean, train_std = get_mean_and_std(train_dataloader)
    #val_mean, val_std = get_mean_and_std(val_dataloader)
    #print(train_mean, train_std, val_mean, val_std)
    
    # ------------------- Sampler ------------------
    # ylen = len(ytrain_aug)
    # print("Augmented train data shape: ", len(xtrain_aug), ylen)
    # labels = []
    # for i in range(0, ylen):
    # # dictionary of the transformations we defined earlier
    #    labels.append(train_data[i][1].flatten())
    # labels = np.array(torch.stack(labels))
    # labels = labels.reshape(labels.shape[0]*labels.shape[1])
    # sampler = BalancedBatchSampler(train_data, labels)
    
    ## ----------------- Weighted Sampler -------------
    # weights = add_sample_weights(train_data, batch_size) 
    #sampler = WeightedRandomSampler(weights.type('torch.DoubleTensor'), len(weights), replacement=True)
    sampler = "no_sampler"

    print('Training')  
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available. Training on CPU...')
        device = torch.device('cpu')
    else:
        print('CUDA is available. Training on GPU...')
        device = torch.device('cuda:0')

    params = {
        "model_name": "Sine_Net",
        "lr": 0.0001,
        "device": device,
        "batch_size": batch_size,
        "num_workers": 4,
        "epochs": 70,
        "start_epoch": 1
    }    
    ## model initialization
    #torch.set_default_dtype(torch.float16)
    batch_size, C, H, W = int(params["batch_size"]), 1, 448, 448
    model = Sine_Net(C, 1)  # initialize with #channels & #classes
    model = model.half().to(params["device"])  #model.half()

    #Weights Initialization
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    model.apply(weights_init)

    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()

    resume = False
    if device == 'cuda':
        model.cuda()
    if resume==True:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/' + check_path_best)
        model.load_state_dict(checkpoint['model_state'])
        params["start_epoch"] = checkpoint['epoch']+1
    else:
        params["start_epoch"] = 1

    print(f"Training on device {device}.")
    optimizer1 = torch.optim.SGD(model.parameters(), lr=params["lr"], momentum=0.9, nesterov=True) ##weight_decay=0.1
    params["epochs"] = 60
    # Initialization of total epochs metrics
    train_losses, val_losses = [], []   # Training Loss & Validation Loss history
    train_corrects_history, val_corrects_history = [], []   # Accuracy history
    model, train_losses, val_losses, train_corrects_history, val_corrects_history = train_and_validate(model, train_data, val_data, sampler, params, optimizer1, train_losses,val_losses, train_corrects_history, val_corrects_history)
    # save best trained model in validate function

    resume=True
    if resume==True:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        # load the model checkpoint
        checkpoint = torch.load('./checkpoint/' + check_path_cont)
        # load model weights state_dict
        model.load_state_dict(checkpoint['model_state'])
        # # load trained optimizer state_dict
        optimizer = checkpoint['optimizer']
        params["start_epoch"] = checkpoint['epoch']+1
        # print(f"Previously trained for {epochs} number of epochs...")
        print('Loading epoch {} successful! '.format(params["start_epoch"]))
         # If there is a saved model, load the model and continue training based on it
    else:
        params["start_epoch"] = 1
        print('No saving model, training from scratch! ')


    print(f"Training on device {device}.")
    #params["lr"] = optimizer['param_groups'][0]['lr']
    params["lr"] = 1e-06
    print("Continue from learning rate: ", params["lr"])
    optimizer2 = torch.optim.Adam(model.parameters(), lr=params["lr"], betas=(0.9, 0.999), eps=1e-04)   ##weight_decay=0.1
    params["epochs"] = 20
    # Initialization of total epochs metrics
    model, train_losses, val_losses, train_corrects_history, val_corrects_history = train_and_validate(model, train_data, val_data, sampler, params, optimizer2, train_losses,val_losses, train_corrects_history, val_corrects_history)

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, color='green', label='train loss')
    plt.plot(val_losses, color='blue', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')
    plt.savefig('./outputs/loss_batch_onlineTverskyLoss.png')
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
    plt.savefig('./outputs/accuracy_batch_onlineTverskyLoss.png')
    plt.show()
