import numpy as np
import random
import h5py
import matplotlib.pyplot as plt
import torch
from preprocessing import *


# function to judge pixel(x,y) in FOV or not
def pixel_inside_FOV(i, x, y, FOVs):
    assert (len(FOVs.shape)==4)  #4D arrays
    assert (FOVs.shape[1]==1)
    if (x >= FOVs.shape[3] or y >= FOVs.shape[2]): # Pixel position is out of range
        return False
    return FOVs[i,:,y,x]>0 #0==black pixels


# Set the pixel value outside FOV to 0, only for visualization
def kill_border(data, FOVs):
    assert (len(data.shape)==4)  #4D arrays
    assert (data.shape[1]==1 or data.shape[1]==3)  #check the channel is 1 or 3
    height = data.shape[2]
    width = data.shape[3]
    for i in range(data.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                if not pixel_inside_FOV(i,x,y,FOVs):
                    data[i,:,y,x]=0.0


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" closes the file after
        return f["image"][()]


# data consinstency check
def data_consistency_check(imgs, masks):
    print('Checking consistency...\n', imgs.shape, masks.shape)
    assert (len(imgs.shape) == len(masks.shape))
    assert (imgs.shape[0] == masks.shape[0])
    assert (imgs.shape[1] == masks.shape[1])
    assert (imgs.shape[2] == masks.shape[2])
    assert (masks.shape[3] == 1)
    assert (imgs.shape[3] == 1 or imgs.shape[3] == 3)


# def inside_FOV_DRIVE(i, x, y, DRIVE_masks):
#     # (20, 565, 565, 1)
#     # assert (DRIVE_masks.shape[3]==1)  #DRIVE masks is black and white but in last axis
#     # DRIVE_masks = DRIVE_masks/255.  #NOOO!! otherwise with float numbers takes forever!!

#     if (x > DRIVE_masks.shape[2] or y > DRIVE_masks.shape[1]): #my image bigger than the original
#         return False

#     if (DRIVE_masks[i,y,x,0]>0):  #0==black pixels
#         # print DRIVE_masks[i,0,y,x]  #verify it is working right
#         return True
#     else:
#         return False


# # check if the patch is fully contained in the FOV
# def is_patch_inside_FOV(x, y, img_w, img_h, patch_h):
#     x_ = x - int(img_w / 2)  # origin (0,0) shifted to image center
#     y_ = y - int(img_h / 2)  # origin (0,0) shifted to image center
#     R_inside = 270 - int(patch_h * np.sqrt(2.0) / 2.0)  # radius is 270 (from DRIVE db docs),
#     # minus the patch diagonal (assumed it is a square) #this is the limit to contain the full patch in the FOV
#     radius = np.sqrt((x_ * x_) + (y_ * y_))
#     if radius < R_inside:
#         return True
#     else:
#         return False


#return only the pixels contained in the FOV, for both images and masks
def pred_only_FOV(data_imgs, data_masks, border_masks):
    # (1,1,584,565)
    print(data_imgs.max(), data_masks.max(), border_masks.max(), border_masks.mean())
    assert (len(data_imgs.shape)==4 and len(data_masks.shape)==4)  #4D arrays
    assert (data_imgs.shape[0]==data_masks.shape[0])
    assert (data_imgs.shape[2]==data_masks.shape[2])
    assert (data_imgs.shape[3]==data_masks.shape[3])
    assert (data_imgs.shape[1]==1 and data_masks.shape[1]==1)  #check the channel is 1
    height = data_imgs.shape[2]
    width = data_imgs.shape[3]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                if pixel_inside_FOV(i,x,y,border_masks)==True:
                    new_pred_imgs.append(data_imgs[i,:,y,x])
                    new_pred_masks.append(data_masks[i,:,y,x])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    return new_pred_imgs.reshape(new_pred_imgs.shape[0]), new_pred_masks.reshape(new_pred_masks.shape[0])
    # -----------------------
    # data_imgs = data_imgs.astype(np.uint8)
    # data_masks = data_masks.astype(np.uint8)
    # border_masks = border_masks.astype(np.uint8)
    # new_pred_imgs = np.zeros(data_imgs.shape)
    # new_pred_masks = np.zeros((border_masks.shape[0], border_masks.shape[2], border_masks.shape[3]))
    # for i in range(data_imgs.shape[0]):  #loop over the full images
    #     new_pred_imgs[i,:,:,:] = cv2.bitwise_and(data_imgs[i], data_imgs[i], mask = border_masks[i])
    #     new_pred_masks[i,:,:] = cv2.bitwise_and(data_masks[i], data_masks[i], mask = border_masks[i])
    # del data_imgs, data_masks, border_masks 
    # return np.array(new_pred_imgs).astype(np.float32), np.array(new_pred_masks).astype(np.float32)


#Divide all the full_imgs in pacthes
def extract_ordered_overlap(full_imgs, patch_h, patch_w,stride_h,stride_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  #// --> division between integers
    N_patches_tot = N_patches_img*full_imgs.shape[0]
    print("Number of patches on h : " +str(((img_h-patch_h)//stride_h+1)))
    print("Number of patches on w : " +str(((img_w-patch_w)//stride_w+1)))
    print("number of patches per image: " +str(N_patches_img) +", totally for this dataset: " +str(N_patches_tot))
    patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                patch = full_imgs[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return patches  #array with all the full_imgs divided in patches


# masks = ground truth patches
def extract_patches(full_imgs, full_masks, patch_h, patch_w):
    assert (len(full_imgs.shape) == 4 and len(full_masks.shape) == 4)  # 4D arrays

    patches, patches_masks = [], []
    _, y_len, x_len, _ = full_imgs[:, :, :, :].shape
    windowSize = (patch_h, patch_w)
    stride_h = y_len - windowSize[0]
    stride_w = x_len - windowSize[1]

    # find window inside full image where the 448*448 patch can be extracted from 565*565 image
    # Sliding window of size 448*448 over image
    for i in range(full_imgs.shape[0]):
        for y in range(0, y_len, stride_h):
            for x in range(0, x_len, stride_w):
                # print(y, x)
                if y + windowSize[0] > y_len or x + windowSize[1] > x_len:
                    continue
                patch = full_imgs[i, y:y + windowSize[1], x:x + windowSize[0], :]
                patch_mask = full_masks[i, y:y + windowSize[1], x:x + windowSize[0], :]
                patches.append(patch)
                patches_masks.append(patch_mask)

    patches = np.asarray(patches)
    patches_masks = np.asarray(patches_masks)
    return patches, patches_masks    


def get_data_training(train_original, train_masks, patch_height, patch_width, inside_FOV):
    # visual_image(group_images(train_imgs_original[0:20,:,:,:],5),'imgs_train')#.show()  #check original imgs train

    # ---------------------------- PATCHES EXTRACTION + PREPROCESS ------------------------------
    # 1. extract patches
    train_imgs = np.expand_dims(train_original[:, 10:575, :], axis=3)  # cut bottom and top so now it is 565*565
    train_masks = np.expand_dims(train_masks[:, 10:575, :], axis=3)  # cut bottom and top so now it is 565*565

    print("\ntrain images/masks shape:")
    print(train_imgs.shape, train_masks.shape)
    print("train images range (min-max): " + str(np.min(train_imgs)) + ' - ' + str(np.max(train_imgs))) 
    print("train masks range(min-max):" + str(np.min(train_masks)) + ' - ' + str(np.max(train_masks)) )
    # extract the TRAINING patches from the full images
    # patches_imgs_train, patches_masks_train = extract_random_patches(train_imgs, train_masks, patch_height, patch_width, inside_FOV)
    patches_imgs_train, patches_masks_train = train_imgs, train_masks
    patches_imgs_train = pre_proc(patches_imgs_train, patches_imgs_train.shape[1], patches_imgs_train.shape[2])
    print(patches_imgs_train.mean(), patches_masks_train.mean())
    return patches_imgs_train, patches_masks_train


def get_data_testing(test_original, test_masks, test_FOVs, patch_height, patch_width):

    # Preprocessing also TEST data:
    test_imgs = np.expand_dims(test_original, axis=(0,3))
    test_imgs = pre_proc(test_imgs, test_imgs.shape[1], test_imgs.shape[2])  # go to preprocess.py
    # test_imgs = test_imgs/255.
    test_masks = np.expand_dims(test_masks, axis=(0,3))
    print(test_FOVs.max())
    test_FOVs = np.expand_dims(test_FOVs, axis=(0,3))

    print("\ntest images/masks shape:")
    print(test_imgs.shape, test_masks.shape)
    print("test images range (min-max): " + str(np.min(test_imgs)) + ' - ' + str(np.max(test_imgs)))
    print("test masks are within 0-1 " + str(np.min(test_masks)) + ' - ' + str(np.max(test_masks)))
    print("test FOVs range (min-max): " + str(np.min(test_FOVs)) + ' - ' + str(np.max(test_FOVs)))

    ## saved the day <3
    import pickle as p
    with open("scaler.pickle", "rb") as f:
        scaler = p.load(f, encoding='bytes')
    test_imgs = scaler.transform(test_imgs.reshape(-1, 565*565))
    test_imgs = test_imgs.reshape(-1, 565, 565, 1)

    # extract the TEST patches from the full images
    patches_imgs_test, patches_masks_test = extract_patches(test_imgs, test_masks, patch_height, patch_width)
    #patches_imgs_test, patches_masks_test = test_imgs, test_masks
    return patches_imgs_test, patches_masks_test, test_imgs, test_masks, test_FOVs
