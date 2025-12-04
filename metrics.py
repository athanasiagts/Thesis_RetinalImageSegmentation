"""
This part contains functions related to the calculation of performance indicators
"""
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
import os
import torch
from os.path import join
import numpy as np
from collections import OrderedDict
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import seaborn as sns

params = {'legend.fontsize': 13,
         'axes.labelsize': 15,
         'axes.titlesize':15,
         'xtick.labelsize':15,
         'ytick.labelsize':15} # define pyplot parameters
pylab.rcParams.update(params)
#Area under the ROC curve
import h5py
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from copy import deepcopy
import cv2

#group a set of img patches 
def group_images(data,per_row):
    assert data.shape[0]%per_row==0
    assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg

# Prediction result splicing (original img, predicted probability, binary img, groundtruth)
def concat_result(ori_img,pred_res,gt):
    ori_img = np.transpose(ori_img,(1,2,0))
    pred_res = np.transpose(pred_res,(1,2,0))
    gt = np.transpose(gt,(1,2,0))

    # fpr, tpr, thresholds = roc_curve(gt.flatten(), pred_res.flatten())
    # score = np.sqrt(tpr * (1 - fpr))
    # np.seterr(divide='ignore', invalid='ignore')
    # index = np.argmax(score)
    # th = thresholds[index]
    # print('Best Threshold=%f, G-mean=%.3f' % (th, score[index]))
    pred_res = (pred_res*255).astype(np.uint8)
    th, imgt = cv2.threshold(pred_res, 0, 255, cv2.THRESH_OTSU)
    print('Otsu Threshold= ', th)

    print("binary mean: ",  imgt.mean())
    if ori_img.shape[2]==3:
        pred_res = np.repeat((pred_res*255).astype(np.uint8),repeats=3,axis=2)
        binary = np.repeat((binary*255).astype(np.uint8),repeats=3,axis=2)
        gt = np.repeat((gt*255).astype(np.uint8),repeats=3,axis=2)
    #total_img = np.concatenate((ori_img,pred_res,imgt.reshape(565,565,1),gt*255),axis=1)
    total_img = imgt.reshape(565,565,1)
    return total_img


#visualize image, save as PIL image
def save_img(data,filename):
    assert (len(data.shape)==3) #height*width*channels
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    img = Image.fromarray(data.astype(np.uint8))  #the image is between 0-1
    img.save(filename)
    return img


class Evaluate():
    def __init__(self, save_path=None, i=None):

        import seaborn as sns

        self.target = None
        self.output = None
        self.save_path = save_path
        self.idx = i
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        self.threshold_confusion = 0.5

    # Add data pair (target and predicted value)
    def add_batch(self,batch_tar,batch_out):
        batch_tar = batch_tar.flatten()
        batch_out = batch_out.flatten()

        self.target = batch_tar if self.target is None else np.concatenate((self.target,batch_tar))
        self.output = batch_out if self.output is None else np.concatenate((self.output,batch_out))
    
    # Plot ROC and calculate AUC of ROC
    def auc_roc(self,plot=False):
        print(self.target.shape, self.output.shape)
        AUC_ROC = roc_auc_score(self.output,  self.target)
        # print("\nAUC of ROC curve: " + str(AUC_ROC))
        if plot and self.save_path is not None:
            fpr, tpr, thresholds = roc_curve(self.target, self.output)
            # print("\nArea under the ROC curve: " + str(AUC_ROC))
            plt.figure()
            plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
            plt.title('ROC curve')
            plt.xlabel("FPR (False Positive Rate)")
            plt.ylabel("TPR (True Positive Rate)")
            plt.legend(loc="lower right")
            plt.savefig(join(self.save_path , "ROC"+str(self.idx)+".png"))
        return AUC_ROC

    # Plot PR curve and calculate AUC of PR curve
    def auc_pr(self,plot=False):
        precision, recall, thresholds = precision_recall_curve(self.target, self.output)
        precision = np.fliplr([precision])[0]
        recall = np.fliplr([recall])[0]
        AUC_pr = np.trapz(precision, recall)
        # print("\nAUC of P-R curve: " + str(AUC_pr))
        if plot and self.save_path is not None:

            plt.figure()
            plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_pr)
            plt.title('Precision - Recall curve')
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(loc="lower right")
            plt.savefig(join(self.save_path ,"Precision_recall"+str(self.idx)+".png"))
        return AUC_pr

    # Accuracy, specificity, sensitivity, precision can be obtained by calculating the confusion matrix
    def confusion_matrix(self):
        #Confusion matrix
        y_pred = self.output>=self.threshold_confusion
        confusion = confusion_matrix(self.target, y_pred)
        # print(confusion)
        accuracy = 0
        if float(np.sum(confusion))!=0:
            accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
        # print("Global Accuracy: " +str(accuracy))
        specificity = 0
        if float(confusion[0,0]+confusion[0,1])!=0:
            specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
        # print("Specificity: " +str(specificity))
        sensitivity = 0
        if float(confusion[1,1]+confusion[1,0])!=0:
            sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
        # print("Sensitivity: " +str(sensitivity))
        precision = 0
        if float(confusion[1,1]+confusion[0,1])!=0:
            precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
        # print("Precision: " +str(precision))
        return confusion,accuracy,specificity,sensitivity,precision

    # Jaccard similarity index
    def jaccard_index(self):
        pass
        # jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
        # print("\nJaccard similarity score: " +str(jaccard_index))

    # calculating f1_score
    def f1_score(self):
        pred = self.output>=self.threshold_confusion
        F1_score = f1_score(self.target, pred, labels=None, average='binary', sample_weight=None)
        # print("F1 score (F-measure): " +str(F1_score))
        return F1_score

    # apply threshold to probabilities to create labels
    def to_labels(pos_probs, threshold):
        over = (pos_probs >= threshold).astype('float')
        return over

    def optimal_thres_tuning(self):       ## y_pred->output, y_test->target
        probs = self.output.flatten()
        y_test = self.target.flatten()
        
        precision, recall, thresholds = precision_recall_curve(y_test, probs)
        precisnan = precision[np.isnan(precision)]
        recallnan = recall[np.isnan(recall)]
        print("NaN values: ", precisnan, recallnan)
        
        import seaborn as sns
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        plt.figure(figsize=(5, 5), dpi=100)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        sns.lineplot(fpr, tpr)
        plt.savefig("roc_curve.png")

        probs[probs>=0.5] = 1.0
        probs[probs<0.5] = 0.0
        confusion = confusion_matrix(y_test, probs)
        print("With threshold=0.5: ", confusion)

        #gmeans
        score = np.sqrt(tpr * (1 - fpr))
        #precision, recall, thresholds = precision_recall_curve(y_test, probs)
        np.seterr(divide='ignore', invalid='ignore')
        #score = np.where(precision+recall!=0, (2 * precision * recall) / (precision + recall), -1)
        
        index = np.argmax(score)
        thresholdOpt = thresholds[index]    #round(thresholds[index], ndigits = 4)
        print('Best Threshold=%f, G-mean=%.3f' % (thresholdOpt, score[index]))
        
        #thresholds = np.arange(0, 1, 0.001)
        #scores = [f1_score(y_test,to_labels(probs, t)) for t in thresholds]
        #ix = np.argmax(scores)
        #print('Threshold=%.5f, F-measure=%.5f' % (thresholds[ix], scores[ix]))

        return thresholdOpt


    def evaluation_metrics(self, th): ## y_scores->output, y_true->target
        print(th) 
        from sklearn.utils.multiclass import type_of_target
        self.output[self.output >= th] = 1.0
        self.output[self.output < th] = 0.0
        print("Number of positives: ", np.sum(self.output))
        
        output = self.output.flatten()
        target = self.target.flatten()
        confusion = confusion_matrix(target, output)
        print(confusion)

        AUC_ROC = roc_auc_score(target, output)
        print("\nArea under the ROC curve: " + str(AUC_ROC))
        
        import seaborn as sns
        fpr_points, tpr_points, thr = roc_curve(target, output)
        plt.figure(figsize=(5, 5), dpi=100)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        sns.lineplot(fpr_points, tpr_points)
        plt.savefig("roc_after.png")
        accuracy = 0
        if float(np.sum(confusion)) != 0:
            accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
        print("Global Accuracy: " + str(accuracy))
        specificity = 0
        if float(confusion[0, 0] + confusion[0, 1]) != 0:
            specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
        print("Specificity: " + str(specificity))
        sensitivity = 0
        if float(confusion[1, 1] + confusion[1, 0]) != 0:
            sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
        print("Sensitivity: " + str(sensitivity))
        precision = 0
        if float(confusion[1, 1] + confusion[0, 1]) != 0:
            precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
        print("Precision: " + str(precision))
        jaccard = 0
        if float(confusion[1, 1] + confusion[0, 1] + confusion[1, 0]) != 0:
            jaccard = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1] + confusion[1, 0])
        print("Jaccard Index: " + str(jaccard))

        return output, target

    def pred2image(self, preds_bin, im):
        # s = int(np.sqrt(test_masks.shape[0]))
        # test_masks = test_masks.reshape(s, s)
        print("mask shape: ", self.target.shape)
        print("preds shape:", preds_bin.shape)
        height = self.target.shape[0]
        width = self.target.shape[1]
        new_image = np.zeros(shape=(height, width))
        j = 0
        for h in range(height):
            for w in range(width):
                if self.target[h,w] == -1:
                    new_image[h,w] = 0
                    continue
                else:
                    new_image[h,w] = preds_bin[j]
                j += 1

        plt.figure(figsize=(30, 30))
        plt.imshow(new_image, cmap="gray")
        plt.savefig("predicted"+str(im)+".png")
        plt.close()
        return

    def performance_metrics(self, i):
            # self.output = np.moveaxis(self.output, 1, -1)
            threshold = self.optimal_thres_tuning()
            preds_over, y_test = self.evaluation_metrics(threshold)
            # Convert predictions to image and plot:
            self.pred2image(preds_over, i)

    # Save performance results to specified file
    def save_all_result(self,i, plot_curve=True,save_name=None):

        self.performance_metrics(i)

        #Save the results
        AUC_ROC = self.auc_roc(plot=plot_curve)
        AUC_pr  = self.auc_pr(plot=plot_curve)
        F1_score = self.f1_score()
        confusion,accuracy, specificity, sensitivity, precision = self.confusion_matrix()
        if save_name is not None:
            file_perf = open(join(self.save_path, save_name), 'w')
            file_perf.write("AUC ROC curve: "+str(AUC_ROC)
                            + "\nAUC PR curve: " +str(AUC_pr)
                            # + "\nJaccard similarity score: " +str(jaccard_index)
                            + "\nF1 score: " +str(F1_score)
                            +"\nAccuracy: " +str(accuracy)
                            +"\nSensitivity(SE): " +str(sensitivity)
                            +"\nSpecificity(SP): " +str(specificity)
                            +"\nPrecision: " +str(precision)
                            + "\n\nConfusion matrix:"
                            + str(confusion)
                            )
            file_perf.close()
        return OrderedDict([("AUC_ROC",AUC_ROC),("AUC_PR",AUC_pr),
                            ("f1-score",F1_score),("Acc",accuracy),
                            ("SE",sensitivity),("SP",specificity),
                            ("precision",precision)
                            ])
