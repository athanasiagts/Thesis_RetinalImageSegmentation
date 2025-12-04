import numpy as np
import random
import h5py
from augmentation_l import dataset_normalized, dataset_whitening
import matplotlib.pyplot as plt
import cv2

def crop_center(img, cropx, cropy):
    y, x = img.shape    
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


# convert RGB image in black and white
def rgb2gray(rgb):
    assert (len(rgb.shape) == 4)  # 4D arrays
    assert (rgb.shape[3] == 3)
    bn_imgs = rgb[:, :, :, 0] * 0.299 + rgb[:, :, :, 1] * 0.587 + rgb[:, :, :, 2] * 0.114
    bn_imgs = np.reshape(bn_imgs, (rgb.shape[0], rgb.shape[1], rgb.shape[2], 1))
    # plt.figure(figsize=(1.5, 1.5))
    # plt.imshow(bn_imgs[19, :, :, :].reshape(584, 565), cmap="gray",  vmin=0, vmax=255)
    # plt.show()
    # plt.close()
    return bn_imgs


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" closes the file after
        return f["image"][()].astype(float)


def pre_proc(dataset):  # preprocess per patch (27x27) -> 400,000 patches for all RGB images (before augmentation)
    print('Num of patches before preprocessing', dataset.shape)  # (400, 27, 27, 3)
    assert(len(dataset.shape) == 4)
    assert (dataset.shape[3] == 3)  # Use the original images with 3 channels

    # train_imgs = rgb2gray(dataset)    # black-white conversion
    train_imgs = dataset

    # pre-processing starts for all patches:
    train_imgs = dataset_normalized(train_imgs)
    # train_imgs = dataset_standardized(train_imgs)
    train_imgs = train_imgs.reshape(dataset.shape[0], 27, 27, 3)

    # Normalization implemented in whitening ?
    # train_imgs2 = dataset_whitening2(train_imgs)
    train_imgs = dataset_whitening(train_imgs)
    train_imgs = train_imgs.reshape(dataset.shape[0], 27, 27, 3)

    return train_imgs


# data consinstency check
def data_consistency_check(imgs, masks):
    print('Checking consistency...\n', imgs.shape, masks.shape)
    assert (len(imgs.shape) == len(masks.shape))
    assert (imgs.shape[0] == masks.shape[0])
    assert (imgs.shape[1] == masks.shape[1])
    assert (imgs.shape[2] == masks.shape[2])
    assert (masks.shape[3] == 1)
    assert (imgs.shape[3] == 1 or imgs.shape[3] == 3)


def extract_ordered_SP(full_imgs, patch_h, patch_w, SP):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[3] == 1 or full_imgs.shape[3] == 3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[1]  # height of the full image
    img_w = full_imgs.shape[2]  # width of the full image
    N_patches_h = int(img_h / patch_h)  # round to lowest int
    if (img_h % patch_h != 0):
        print("warning: " + str(N_patches_h) + " patches in height, with about " + str(
            img_h % patch_h) + " pixels left over")
    N_patches_w = int(img_w / patch_w)  # round to lowest int
    if (img_h % patch_h != 0):
        print("warning: " + str(N_patches_w) + " patches in width, with about " + str(
            img_w % patch_w) + " pixels left over")
    print(N_patches_h)
    print("number of patches per image: " + str(N_patches_h * N_patches_w))
    N_patches_tot = (N_patches_h * N_patches_w) * full_imgs.shape[0]
    s = 5
    patches = np.empty((N_patches_tot, patch_h, patch_w, full_imgs.shape[3]))
    windows = np.empty([N_patches_tot, s, s, full_imgs.shape[3]])

    iter_tot = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                patch = full_imgs[i, h*patch_h:(h*patch_h)+patch_h, w*patch_w:(w * patch_w)+patch_w, :]
                patches[iter_tot] = patch
                if SP==True:
                    win = patch[13 - int(s / 2):13 + int(s / 2) + 1, 13 - int(s / 2):13 + int(s / 2) + 1, :]
                    windows[iter_tot] = win
                iter_tot += 1  # total
    assert (iter_tot == N_patches_tot)
    return patches, windows  # array with all the full_imgs divided in patches


# extract the TEST patches from the full images - Divide all the full TEST images in patches
def extract_ordered(full_imgs, patch_h, patch_w):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[3] == 1 or full_imgs.shape[3] == 3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[1]  # height of the full image
    img_w = full_imgs.shape[2]  # width of the full image
    N_patches_h = int(img_h / patch_h)  # round to lowest int
    if (img_h % patch_h != 0):
        print("warning: " + str(N_patches_h) + " patches in height, with about " + str(
            img_h % patch_h) + " pixels left over")
    N_patches_w = int(img_w / patch_w)  # round to lowest int
    if (img_h % patch_h != 0):
        print("warning: " + str(N_patches_w) + " patches in width, with about " + str(
            img_w % patch_w) + " pixels left over")
    print("number of patches per image: " + str(N_patches_h * N_patches_w))
    N_patches_tot = (N_patches_h * N_patches_w) * full_imgs.shape[0]
    patches = np.empty((N_patches_tot, patch_h, patch_w, full_imgs.shape[3]))

    iter_tot = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                patch = full_imgs[i, h*patch_h:(h*patch_h)+patch_h, w*patch_w:(w * patch_w)+patch_w, :]
                patches.append(patch)
                iter_tot += 1   # total
    return np.array(patches)  # array with all the full_imgs divided in patches


# extract TEST patches for all pixels in full images
def extract_all(full_imgs, full_masks, patch_h, patch_w, border_masks):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[3] == 1 or full_imgs.shape[3] == 3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[1]  # height of the full image
    img_w = full_imgs.shape[2]  # width of the full image
    patches = []

    print("masks shape: ", full_masks.shape)
    full_imgs = paint_border_all(full_imgs, patch_h, patch_w)
    new_mask = np.x = np.full((full_masks.shape[1], full_masks.shape[2]), -1)
    print("New extended images shape: ", full_imgs.shape)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range(img_h):
            for w in range(img_w):
                # check whether the pixel is fully contained in the FOV     # ADDED
                if not inside_FOV_DRIVE(i,h,w,border_masks):  # is pixel inside FOV?
                    continue
                patch = full_imgs[i, h:h+patch_h, w:w+patch_w, :]
                patches.append(patch)
                new_mask[h,w] = full_masks[i, h, w]
                
    print("new mask 0/1/-1: ", new_mask.shape)
    mask_flat = new_mask[(new_mask == 0) | (new_mask == 1)]
    print("flat mask: ", mask_flat.shape)
    print("Number of patches in all images inside FOV: ", np.array(patches).shape, new_mask.shape)
    return np.array(patches), new_mask, mask_flat  # array with all the full_imgs divided in patches


# masks = ground truth patches
def extract_random_patches(full_imgs, full_masks, patch_h, patch_w, n_patches, inside_FOV, SP):
    # extract patches randomly in the full training images -- Inside OR in full image
    if n_patches % full_imgs.shape[0] != 0:
        print("n_patches: please enter a multiple of 20")
        exit()
    assert (len(full_imgs.shape) == 4 and len(full_masks.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[3] == 1 or full_imgs.shape[3] == 3)  # check the channel is 1 or 3
    assert (full_masks.shape[3] == 1)  # masks only black and white
    assert (full_imgs.shape[1] == full_masks.shape[1] and full_imgs.shape[2] == full_masks.shape[2])
    s = 5
    windows, patches, patches_masks = [], [], []
    img_h = full_imgs.shape[1]  # height of the full image
    img_w = full_imgs.shape[2]  # width of the full image  # 565 x 565

    # (0,0) in the center of the image
    patch_per_img = int(n_patches / full_imgs.shape[0])  # n_patches equally divided in the full images
    print("Number of patches per full image: " + str(patch_per_img))
    iter_tot = 0  # iter over the total number of patches (n_patches)

    for i in range(full_imgs.shape[0]):  # loop over the full images (20)
        # ------ create BALANCED data - #pos patches = #neg patches
        indices_pos = np.where(full_masks[i,:,:,:]==1)  # positive=white=1 / negative=black=0
        indices_neg = np.where(full_masks[i,:,:,:]==0)
        print("image:", i)
        balanced=True    # balanced dataset extraction also for non-SP problem
        if balanced==True:
            count, length = 0, len(indices_pos[0])  # if SP => #positive patches in image == #vessel pixels in image
            print(length, len(indices_neg[0]))
            # patches per image = number of positive pixels in image (*2)
            while count < (patch_per_img//2) and count < length:    # iter_tot < n_patches:
                position = random.randint(0, length-1)  #take as position of pixel a random index in array
                x_center = indices_pos[0][position]
                y_center = indices_pos[1][position]     #take same position index in y-axis of pixels
                # print("center pixel:  " + str(full_imgs[i, y_center, x_center, :]))
                # check whether the patch is fully contained in the FOV
                if not is_patch_inside_FOV(x_center, y_center, img_w, img_h, patch_h):
                    continue
                patch = full_imgs[i, y_center - int(patch_h / 2):y_center + int(patch_h / 2) + 1,
                        x_center - int(patch_w / 2):x_center + int(patch_w / 2) + 1, :]
                patch_mask = full_masks[i, y_center - int(patch_h / 2):y_center + int(patch_h / 2) + 1,
                             x_center - int(patch_w / 2):x_center + int(patch_w / 2) + 1, :]

                # extract centered window SxS inside patches if SP==True (centered pixel = (13,13))
                if SP==True:
                    win = patch_mask[13 - int(s / 2):13 + int(s / 2) + 1,
                        13 - int(s / 2):13 + int(s / 2) + 1, :]
                    windows.append(win)

                patches.append(patch)
                patches_masks.append(patch_mask)
                iter_tot += 1  # total
                count += 1  # per full_img

            no_negs = count
            print(no_negs)
            count, length = 0, len(indices_neg[0])  # if SP => #positive patches in image == #vessel pixels in image
            while count < no_negs:
                position = random.randint(0, length-1)
                x_center = indices_neg[0][position]
                # print("x_center " + str(x_center))
                y_center = indices_neg[1][position]
                # print("y_center " + str(y_center))
                # check whether the patch is fully contained in the FOV
                if not is_patch_inside_FOV(x_center, y_center, img_w, img_h, patch_h):
                    continue
                patch = full_imgs[i, y_center - int(patch_h / 2):y_center + int(patch_h / 2) + 1,
                        x_center - int(patch_w / 2):x_center + int(patch_w / 2) + 1, :]
                patch_mask = full_masks[i, y_center - int(patch_h / 2):y_center + int(patch_h / 2) + 1,
                             x_center - int(patch_w / 2):x_center + int(patch_w / 2) + 1, :]

                # extract centered window SxS inside patches if SP==True
                if SP==True:
                    win = patch_mask[13 - int(s / 2):13 + int(s / 2) + 1,
                        13 - int(s / 2):13 + int(s / 2) + 1, :]
                    windows.append(win)

                patches.append(patch)
                patches_masks.append(patch_mask)
                iter_tot += 1  # total
                count += 1  # per full_img

        else:  # random imbalanced extraction
            k = 0
            while k < patch_per_img:
                x_center = random.randint(0 + int(patch_w / 2), img_w - int(patch_w / 2) - 1)
                # print("x_center " + str(x_center))
                # print(0 + int(patch_w / 2), img_w - int(patch_w / 2))
                y_center = random.randint(0 + int(patch_h / 2), img_h - int(patch_h / 2) - 1)
                # print(0 + int(patch_h / 2), img_h - int(patch_h / 2))
                # print("y_center " + str(y_center))
                # check whether the patch is fully contained in the FOV
                if not is_patch_inside_FOV(x_center, y_center, img_w, img_h, patch_h):
                    continue
                patch = full_imgs[i, y_center - int(patch_h / 2):y_center + int(patch_h / 2) + 1,
                        x_center - int(patch_w / 2):x_center + int(patch_w / 2) + 1, :]
                patch_mask = full_masks[i, y_center - int(patch_h / 2):y_center + int(patch_h / 2) + 1,
                             x_center - int(patch_w / 2):x_center + int(patch_w / 2) + 1, :]

                # patches[iter_tot] = patch
                # patches_masks[iter_tot] = patch_mask
                patches.append(patch)
                patches_masks.append(patch_mask)
                iter_tot += 1  # total
                k += 1  # per full_img

    return np.array(patches), np.array(patches_masks), np.array(windows)


# check if the patch is fully contained in the FOV
def is_patch_inside_FOV(x, y, img_w, img_h, patch_h):
    x_ = x - int(img_w / 2)  # origin (0,0) shifted to image center
    y_ = y - int(img_h / 2)  # origin (0,0) shifted to image center
    R_inside = 270 - int(patch_h * np.sqrt(2.0) / 2.0)  # radius is 270 (from DRIVE db docs),
    # minus the patch diagonal (assumed it is a square) #this is the limit to contain the full patch in the FOV
    radius = np.sqrt((x_ * x_) + (y_ * y_))
    if radius < R_inside:
        return True
    else:
        return False


# extract preprocessed images' patches for training / testing
# masks == ground Truth images
def get_data_training1(train_img_orig, train_masks, patch_height, patch_width, Nsubim, inside_FOV, SP):
    train_original = load_hdf5(train_img_orig)[:10]
    train_masks = load_hdf5(train_masks)[:10]

    print('masks mean: ', train_masks.mean())
    # visual_image(group_images(train_imgs_original[0:20,:,:,:],5),'imgs_train')#.show()  #check original imgs train

    # ---------------------------- PATCHES EXTRACTION + PREPROCESS ------------------------------
    train_masks = train_masks / 255.
    # 1. extract patches
    train_imgs = train_original[:, 9:574, :, :]  # cut bottom and top so now it is 565*565
    train_masks = train_masks[:, 9:574, :, :]  # cut bottom and top so now it is 565*565
    # data consistency check:
    data_consistency_check(train_imgs, train_masks)

    # check masks are within 0-1
    assert (np.min(train_masks) == 0 and np.max(train_masks) == 1)

    print("\ntrain images/masks shape:")
    print(train_imgs.shape)
    print("train images range (min-max): " + str(np.min(train_imgs)) + ' - ' + str(np.max(train_imgs)))
    print("train masks are within 0-1\n")

    # extract the TRAINING patches from the full images
    return extract_random_patches(train_imgs, train_masks, patch_height, patch_width, Nsubim, inside_FOV, SP)

    # data_consistency_check(patches_imgs_train, patches_masks_train)
    # print("\ntrain PATCHES images/masks shape:")
    # print(patches_imgs_train.shape)
    # print("train PATCHES range (min-max): " + str(np.min(patches_imgs_train)) + ' - ' + str(np.max(patches_imgs_train)))

    # 2. GCN/ZCA pre-processing
    # patches_imgs_train = pre_proc(patches_imgs_train)  # go to preprocess.py [400, 27, 27, 3]
    # print(patches_imgs_train.shape)

    # return patches_imgs_train, patches_masks_train


# ------------------------------------ FOR EVALUATION ----------------------------------
# # Extend the full images because patch division is not exact
# def paint_border(data, patch_h, patch_w):
#     assert (len(data.shape) == 4)  # 4D arrays
#     assert (data.shape[3] == 1 or data.shape[3] == 3)  # check the channel is 1 or 3
#     img_h = data.shape[1]
#     img_w = data.shape[2]
#     new_img_h = 0
#     new_img_w = 0
#     if (img_h % patch_h) == 0:
#         new_img_h = img_h
#     else:
#         new_img_h = ((int(img_h) / int(patch_h)) + 1) * patch_h
#     if (img_w % patch_w) == 0:
#         new_img_w = img_w
#     else:
#         new_img_w = ((int(img_w) / int(patch_w)) + 1) * patch_w
#     new_data = np.zeros((data.shape[0], int(new_img_h), int(new_img_w), data.shape[3]), dtype=np.uint8)
#     new_data[:, 0:img_h, 0:img_w, :] = data[:, :, :, :]
#     return new_data


def paint_border_all(data, patch_h, patch_w):
    assert (len(data.shape) == 4)
    assert (data.shape[3] == 1 or data.shape[3] == 3)  # check the channel is 1 or 3
    img_h = data.shape[1]
    img_w = data.shape[2]
    new_data = np.zeros((data.shape[0], int(img_h)+int(patch_h//2)*2, int(img_w)+int(patch_w//2)*2, data.shape[3]), dtype=np.uint8)
    new_data[:, 0+int(patch_h//2):img_h+int(patch_h//2), 0+int(patch_w//2):img_w+int(patch_w//2), :] = data[:, :, :, :]
    return new_data


# Load the original data and return the extracted patches for testing
def get_data_testing1(test_imgs, test_masks, test_border_masks, patch_height, patch_width, SP):

    test_masks = test_masks/255.
    print('imgs mean: ', test_imgs.mean())
    
    if SP==True:
        patches_imgs_test, windows = extract_ordered_SP(test_imgs, patch_height, patch_width)
        patches_masks_test, windows = extract_ordered_SP(test_masks, patch_height, patch_width, SP)
        data_consistency_check(patches_imgs_test, patches_masks_test)
        return patches_imgs_test, patches_masks_test, windows
    else:
        # extract the TEST patches from the full images
        patches_imgs_test, test_masks, flat_masks = extract_all(test_imgs, test_masks, patch_height, patch_width, test_border_masks)
        # plt.figure(figsize=(10, 10))
        # plt.imshow(test_masks, cmap='gray')
        # plt.show()
        # plt.close()
        return patches_imgs_test, flat_masks


# Recompone the full images with the patches
def recompone(data, N_h, N_w):
    assert (data.shape[3]==1 or data.shape[3]==3)  #check the channel is 1 or 3
    assert(len(data.shape)==4)
    N_patch_per_img = N_w*N_h
    assert(data.shape[0]%N_patch_per_img == 0)
    N_full_imgs = data.shape[0]/N_patch_per_img
    patch_h = data.shape[1]
    patch_w = data.shape[2]
    N_patch_per_img = N_w*N_h
    # define and start full recompone
    full_recomp = np.empty((N_full_imgs, N_h*patch_h, N_w*patch_w, data.shape[3]))
    k = 0  # iter full img
    s = 0  # iter single patch
    while s < data.shape[0]:
        #recompone one:
        single_recon = np.empty((N_h*patch_h, N_w*patch_w, data.shape[3]))
        for h in range(N_h):
            for w in range(N_w):
                single_recon[h*patch_h:(h*patch_h)+patch_h, w*patch_w:(w*patch_w)+patch_w, :] = data[s]
                s+=1
        full_recomp[k]=single_recon
        k+=1
    assert (k==N_full_imgs)
    return full_recomp


#return only the pixels contained in the FOV, for both images and masks
def pred_only_FOV_l(data_imgs, data_masks, border_masks):
    assert (len(data_imgs.shape)==4 and len(data_masks.shape)==4)  #4D arrays
    assert (data_imgs.shape[0]==data_masks.shape[0])
    assert (data_imgs.shape[1]==data_masks.shape[1])
    assert (data_imgs.shape[2]==data_masks.shape[2])
    assert (data_imgs.shape[3]==3 and data_masks.shape[3]==1)  #check the channel is 1
    height = data_imgs.shape[1]
    width = data_imgs.shape[2]
    data_imgs = data_imgs.astype(np.uint8)
    data_masks = data_masks.astype(np.uint8)
    border_masks = border_masks.astype(np.uint8)
    new_pred_imgs = np.zeros(data_imgs.shape)
    new_pred_masks = np.zeros((border_masks.shape[0], border_masks.shape[1], border_masks.shape[2]))
    for i in range(data_imgs.shape[0]):  #loop over the full images
        new_pred_imgs[i,:,:,:] = cv2.bitwise_and(data_imgs[i], data_imgs[i], mask = border_masks[i])
        new_pred_masks[i,:,:] = cv2.bitwise_and(data_masks[i], data_masks[i], mask = border_masks[i])
    #     for x in range(width):
    #         for y in range(height):
    #             if inside_FOV_DRIVE(i,y,x,original_imgs_border_masks)==True:
    #                 new_pred_imgs[i,y,x,:] = (data_imgs[i,y,x,:])
    #                 new_pred_masks[i,y,x,:] = (data_masks[i,y,x,:])
    #             else:
    #                 new_pred_imgs[i,y,x,:] = 0.0
    #                 new_pred_masks[i,y,x,:] = 0.0

    del data_imgs, data_masks, border_masks
    # plt.figure(figsize=(10, 10))
    # plt.imshow((new_pred_imgs[0]/255).reshape(584,565,3))
    # plt.show()
    # plt.close()
    # plt.figure(figsize=(10, 10))
    # plt.imshow((data_imgs[0]/255).reshape(584,565,3))
    # plt.show()
    # plt.close()    
    return np.array(new_pred_imgs).astype(np.float32), np.array(new_pred_masks).astype(np.float32)


# function to set to black everything outside the FOV, in a full image
def kill_border(data, original_imgs_border_masks):
    assert (len(data.shape)==4)  #4D arrays
    assert (data.shape[3]==1 or data.shape[3]==3)  #check the channel is 1 or 3
    height = data.shape[1]
    width = data.shape[2]
    for i in range(data.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                if inside_FOV_DRIVE(i,y,x,original_imgs_border_masks)==False:
                    data[i,y,x,:]=0.0


def inside_FOV_DRIVE(i, y, x, DRIVE_masks):
    assert (len(DRIVE_masks.shape)==4)  #4D arrays
    assert (DRIVE_masks.shape[3]==1)  #DRIVE masks is black and white
    # DRIVE_masks = DRIVE_masks/255.  #NOOO!! otherwise with float numbers takes forever!!

    if x >= DRIVE_masks.shape[2] or y >= DRIVE_masks.shape[1]: #my image bigger than the original (or x->2 & y->1)
        return False

    if DRIVE_masks[i, y, x, 0]>0:  #0==black pixels
        # print DRIVE_masks[i,0,y,x]  #verify it is working right
        return True
    else:
        return False