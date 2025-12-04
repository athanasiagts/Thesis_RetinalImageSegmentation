# here define the pre-processing functions & the functions used for data augmentation
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from keras_preprocessing import image
import cv2
import sklearn
from PIL import Image

def plotImage(X):
    plt.figure()
    im = X.reshape(27, 27, 3)
    im = Image.fromarray(im, mode='RGB')
    im.show()
    plt.close()


# Global Contrast Normalization -> local PER PATCH brightness and contrast normalization (independently in the R, G, B)
def dataset_normalized(X_patch):  # (all patches)
    assert (len(X_patch.shape) == 3)  # 3D arrays  (or 4D?)
    assert (X_patch.shape[2] == 3)  # check the 3rd dimension for channel is 1 (grayscale) or 3 (RGB)
    # print('Image patch shape for normalization: ', X_patch.shape)

    # mean = np.mean(X_patch, axis=(0,1), keepdims=True)
    # std = np.sqrt(((X_patch - mean) ** 2).mean(axis=(0,1), keepdims=True))
    # # img_normalized = np.divide(np.subtract(X_patch, mean, out=X_patch), std, dtype=np.float64)
    # img_normalized = X_patch - mean / std

    x_min = np.min(X_patch, keepdims=True) #axis=(0, 1),
    x_max = np.max(X_patch, keepdims=True)
    denom =  x_max - x_min  # shape=[1,1,3]
    del x_max
    denom = np.array([max(dif, 0.00001) for dif in denom[0,0,:]])

    img_normalized = (X_patch - x_min) / denom

    return img_normalized


def dataset_standardized(X):  # (per patch standardization)
    assert (len(X.shape) == 3)  # 4D arrays
    assert (X.shape[2] == 3)  # check the 3rd dimension for channel is 3
    # images_normalized = np.empty(X.shape)

    # for i in range(X.shape[0]):  # or imgs.shape -> for all images
    pixels = np.asarray(X[:,:,:])
    pixels = pixels.astype('float32')   # [27,27,3]
    means, stds = pixels.mean(axis=(0,1), dtype='float64'), pixels.std(axis=(0,1), dtype='float64')
    stds = np.array([max(s, 0.00001) for s in stds])
    img_normalized = (pixels - means) / stds
    # plt.figure(figsize=(1.5, 1.5))
    # plt.imshow(images_normalized[i].reshape(27, 27, 3))
    # plt.show()
    # plt.close()

    return img_normalized


# Zero-phase Component Analysis (ZCA Whitening) - classical PCA whitening transformation: W_PCA = Λ^(−1/2)*U^T
def dataset_whitening(X_patch):  # X_patch already normalized
    # shape: (n, 27, 27, 3)
    first_dim = X_patch.shape[0]
    second_dim = X_patch.shape[1] * X_patch.shape[2] * X_patch.shape[3]
    shape = (first_dim, second_dim)
    X_patch = X_patch.reshape(shape)
    # X_patch_norm = X_patch / 255

    X_patch = X_patch - np.mean(X_patch,0)  # zero-center the data to 1st dimension [400, 2187]

    cov = np.cov(X_patch, rowvar=False)  # get the data covariance matrix (True)
    U, S, _ = np.linalg.svd(cov, full_matrices=False)
    
    epsilon = 1e-5
    zca_matrix = U.dot(np.diag(1.0 / np.sqrt(S + epsilon))).dot(U.T)
    X_patch = zca_matrix.dot(X_patch.T).T
    del U
    del S
    del cov

    # rescale
    xzca_min = np.min(X_patch, keepdims=True)
    xzca_max = np.max(X_patch, keepdims=True)
    denom = xzca_max - xzca_min
    denom = np.array([max(dif, 0.00001) for dif in denom])
    X_patch = (X_patch - xzca_min) / denom
    
    del xzca_min, xzca_max, denom
    # print(X_ZCA.shape)
    # print(X_patch_norm.shape)
    # print("NOW PLOT1")
    # plt.subplot(1, 2, 1)
    # plt.imshow(X_patch_norm[108,:].reshape(27,27,3))
    # plt.subplot(1, 2, 2)
    # plt.imshow(X_ZCA[108,:].reshape(27,27,3), cmap="gray")
    # plt.show()
    # plt.close()

    # print('min:', X_ZCA.min())
    # print('max:', X_ZCA.max())
    return X_patch.reshape(first_dim, 27, 27, 3), zca_matrix


def dataset_whitening2(X_patch):
    from sklearn.decomposition import PCA

    first_dim = X_patch.shape[0]
    second_dim = X_patch.shape[1] * X_patch.shape[2] * X_patch.shape[3]
    shape = (first_dim, second_dim)
    X_patch2d = X_patch.reshape(shape)
    X_patch2d = X_patch2d / 255

    pca = PCA(n_components=200, random_state=0, svd_solver='randomized')
    # pca = PCA(.95, random_state=0)

    X_pca = pca.fit_transform(X_patch2d)
    # print(pca.n_components_) # for scikit-learn==0.19 version
    # else: find components_ by creating array [pca.n_components_, pca.n_features_]

    X_ZCA = X_pca.dot(pca.components_)

    # print("NOW PLOT")
    # plt.subplot(1, 2, 1)
    # plt.imshow(X_patch[55,:,:,:].reshape(27,27,3))
    # plt.subplot(1, 2, 2)
    # plt.imshow(X_ZCA[55,:].reshape(27,27,3))
    # plt.show()
    # plt.close()

    return X_ZCA


# ============== OFFLINE DATA AUGMENTATION (PRE-PROCESSING) ================
# Augmentations: each patch, normalized and whitened (10 independent transformations):
# • Scaling by a factor between 0.7 and 1.2
# • Rotation by an angle from [−90, 90]
# • Flipping horizontally or vertically
# • Gamma correction of Saturation and Value (of the HSV colorspace) by raising pixels to a power in [0.25, 4]
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_WRAP)


def scaling(image_array: ndarray):
    random_factor = random.uniform(0.7, 1.2)
    return sk.transform.rescale(image_array, random_factor, multichannel=True)   # may retun (height, width, 2)?


def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 90% on the left and 90% on the right
    random_degree = random.uniform(-90, 90)
    # return sk.transform.rotate(image_array, random_degree, mode='wrap')   # does not handle the "cut-off"
    return rotate_bound(image_array, random_degree)


def horizontal_vertical_flip(image_array: ndarray):
    # # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    # return image_array[:, ::-1]
    choices = ['hflip', 'vflip']
    choice = random.choice(choices)

    if choice == 'hflip':
        flipped_image = np.fliplr(image_array)  # fliplr reverse the order of columns of pixels in matrix
    else:
        flipped_image = np.flipud(image_array)  # flipud reverse the order of rows of pixels in matrix

    # fig, axes = plt.subplots(nrows=1, ncols=3)
    # ax = axes.ravel()
    #
    # ax[0].imshow(image_array, cmap='gray')
    # ax[0].set_title("Original image")
    #
    # ax[1].imshow(hflipped_image, cmap='gray')
    # ax[1].set_title("Horizontally flipped")
    #
    # ax[2].imshow(vflipped_image, cmap='gray')
    # ax[2].set_title("Vertically flipped")
    # plt.tight_layout()
    # plt.show()
    return flipped_image


def adjust_gamma(image_array: ndarray):
    print(image_array.shape)
    # assert (len(image_array.shape) == 3)  # 2D arrays
    random_gamma = random.uniform(0.25, 4)
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / random_gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    table = table.reshape(1, 256)
    image_array = (image_array * 255).astype(np.uint8)
    assert image_array.dtype == np.uint8, 'must be np.uint8 image'
    # apply gamma correction using the lookup table
    new_imgs = cv2.LUT(image_array, table)

    # lookUpTable = np.empty((1,256), np.uint8)
    # for i in range(256):
    #     lookUpTable[0,i] = np.clip(pow(i / 255.0, random_gamma) * 255.0, 0, 255)
    # new_imgs = cv2.LUT(image_array, lookUpTable)
    print('Gamma transformed max: ', new_imgs.max())
    print('Gamma transformed min: ', new_imgs.min())

    return new_imgs
