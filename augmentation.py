#import albumentations as Al
#from albumentations.augmentations.geometric import rotate, transforms
#from albumentations.augmentations.transforms import Flip
#from albumentations.pytorch.transforms import ToTensorV2
from albumentations import OneOf, Rotate, Flip, HorizontalFlip, VerticalFlip, Compose, CenterCrop, Affine
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os, cv2
from torch.utils.data import Dataset


def aug_apply(patches_imgs_train, patches_gT_train, idx_train, patch_dim, data_path, step):
    # ================================= AUGMENTATION ======================================

    # print('images patches shape: ', patches_imgs_train.shape)
    result_dir = data_path + 'Augmented_Images/'
    # print("\n1. Create directory for the results (if not already existing)")
    if os.path.exists(result_dir):
        print("Augemnted path exists!\n")
        augmented_path = result_dir
    else:
        print("Create augmented path:\n")
        os.system('mkdir ' + result_dir)
        augmented_path = data_path + 'Augmented_Images/'

    transformations_step1 = Compose([
        Rotate(limit=45, p=1, interpolation=cv2.INTER_LINEAR),
        Affine(translate_px=(0,117), p=1, interpolation=cv2.INTER_LINEAR)
    ])

    transformations_step2 = Compose([
        Rotate(limit=90, p=1, interpolation=cv2.INTER_LINEAR),
        OneOf([VerticalFlip(), HorizontalFlip()], p=1)
    ])

    # number of sequential transformation combinations to apply to each patch > with ALL 2 steps: 1 OR 8
    if step == 1:
        num_transformations_to_apply = 1
        available_transformations = transformations_step1
    else:
        num_transformations_to_apply = 8
        available_transformations = transformations_step2

    # Number of patches desired is equal to the number of original patches for the 1st Augmentation step
    # num_patches_desired = len(patches_imgs_train)-0.1*len(patches_imgs_train)
    num_patches_desired = len(idx_train)

    num_generated = 1
    while num_generated <= num_patches_desired:
        # select 1st patch to augment
        curr_id = idx_train[num_generated-1]
        print(curr_id, patches_imgs_train.shape)
        image = patches_imgs_train[curr_id]   # Random choice of a single patch
        groundTruth = patches_gT_train[curr_id]

        image_to_transform = np.asarray(image.reshape(image.shape[0], image.shape[1], image.shape[2])).astype(np.float32)
        gt_to_transform = np.asarray(groundTruth.reshape(groundTruth.shape[0], groundTruth.shape[1], groundTruth.shape[2])).astype(np.float32)
        # print(gt_to_transform)
        #fig, axes = plt.subplots(1,2)
        #axes[0].imshow(image_to_transform.reshape(565,565), cmap='gray')
        #axes[0].set_title("Original image")
        #axes[1].imshow(gt_to_transform.reshape(565,565), cmap='gray')
        #axes[1].set_title("Original ground truth")
        #plt.tight_layout()
        #plt.savefig("1transf.png")

        num_transformations = 1
        while num_transformations <= num_transformations_to_apply:  # 1 OR 8

            if step == 1:
                new_file_path = augmented_path + 'augmented_img_train' + str(curr_id) + '_' + str(0) + '.pt'
                new_file_path_mask = augmented_path + 'augmented_gt_train' + str(curr_id) + '_' + str(0) + '.pt'
            elif step == 2:
                new_file_path = augmented_path + 'augmented_img_train' + str(curr_id) + '_' + str(num_transformations) + '.pt'
                new_file_path_mask = augmented_path + 'augmented_gt_train' + str(curr_id) + '_' + str(num_transformations) + '.pt'

            # remember to also save original/un-transformed ones in file ONCE
            if step == 1:
                # print('Save originals to augmented path:\n')
                #to = transform_original(image=image_to_transform, mask=gt_to_transform)
                torch.save(image_to_transform,
                           augmented_path + 'img_train' + str(curr_id) + '.pt')
                torch.save(gt_to_transform,
                           augmented_path + 'gt_train' + str(curr_id) + '.pt')
                

            # initialization of the transformed image is the image to transform
            transformed_image = image_to_transform
            transformed_gt = gt_to_transform
            # random transformation to apply for a single image patch -> save 1 or 8 transformations for each
            transformed = available_transformations(image=transformed_image, mask=transformed_gt)
            transformed_image, transformed_gt = transformed["image"], transformed["mask"]
            
            # fig, axes = plt.subplots(1, 2)
            # axes[0].imshow(transformed_image.data.clone().permute(1,2,0))
            # axes[0].set_title('Transformed image # ' + str(num_transformations))
            # axes[1].imshow(transformed_gt.data.clone().permute(1,2,0), cmap='gray')
            # axes[1].set_title('Transformed ground Truth # ' + str(num_transformations))
            # plt.tight_layout()
            # plt.show()

            torch.save(transformed_image, new_file_path)
            torch.save(transformed_gt, new_file_path_mask)

            num_transformations += 1

        num_generated += 1

    # write also untransformed original images to disk

    # try:
    #     torch.save(training_x.data.clone(), augmented_path + 'augmented_imgs_train.pt')
    #     torch.save(training_y.data.clone(), augmented_path + 'augmented_groundTruth_train.pt')
    # except Exception as e:
    #     print("Failed to save augmented images.")
    #     print(e)
    return

