#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    
    # Compute intersection area 
    tl_row_I = max(box_1[0],box_2[0])
    tl_col_I = max(box_1[1],box_2[1])
    br_row_I = min(box_1[2],box_2[2])
    br_col_I = min(box_1[3],box_2[3])
    Area_I = max((br_row_I-tl_row_I, 0)) * max((br_col_I - tl_col_I), 0)
    
    Area_1 = (box_1[2]-box_1[0]) * (box_1[3]-box_1[1])
    Area_2 = (box_2[2]-box_2[0]) * (box_2[3]-box_2[1])
    
    iou = Area_I / (Area_1+Area_2-Area_I)
    
    assert (iou >= 0) and (iou <= 1.0)

    return iou


# In[3]:


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        
        pred_t = []
        # threshold predictions, "t" stands for threshold
        for pred_i in range(len(pred)):
            if pred[pred_i][4] >= conf_thr:
                pred_t.append(pred[pred_i])
        
        # There are predictions
        if len(pred_t)>0:
            # predictions that are associated successfully
            pred_associated = []

            for i in range(len(gt)):

                FN += 1 #regard as false negative by default
                iou = np.zeros((len(pred_t),))

                for j in range(len(pred_t)):
                    iou[j] = compute_iou(pred[j][:4], gt[i])

                iou_max = np.amax(iou)
                idx_max = np.where(iou == iou_max)

                # whether associated successfully
                if iou_max > iou_thr:
                    if idx_max not in pred_associated:
                        pred_associated.append(idx_max)
                        TP += 1
                        FN -= 1

            FP += (len(pred_t)-len(pred_associated))
        
        # There is no prediction
        else:
            FN = len(gt) 
    '''
    END YOUR CODE
    '''

    return TP, FP, FN


# In[4]:


def PR_curve (preds, gts, train, weaken):
    scores = []
    for fname in preds:
        for pred in range(len(preds[fname])):
            scores.append(preds[fname][pred][4])
    confidence_thrs = np.sort(scores)
    #confidence_thrs = np.sort(np.array([preds_train[fname][4] for fname in preds_train if len(preds_train[fname]) !=0] ,dtype=float))
    iou_thrs = [0.25,0.5,0.75]
    tp = np.zeros((3,len(confidence_thrs)))
    fp = np.zeros((3,len(confidence_thrs)))
    fn = np.zeros((3,len(confidence_thrs)))
    P = np.zeros((3,len(confidence_thrs)))
    R = np.zeros((3,len(confidence_thrs)))
    for i, conf_Thr in enumerate(confidence_thrs):
        for j in range(3):
            tp[j][i], fp[j][i], fn[j][i] = compute_counts(preds, gts, iou_thr=iou_thrs[j], conf_thr=conf_Thr)
            
            if tp[j][i] + fp[j][i] == 0: # no predictions
                P[j][i] = 1
            else:
                P[j][i] = tp[j][i]/(tp[j][i]+fp[j][i])

            if tp[j][i] + fn[j][i] == 0: 
                R[j][i] = 1
            else:
                R[j][i] = tp[j][i]/(tp[j][i]+fn[j][i])
                
    if train:
        if weaken:
            plt_title = "PR curves of the training set, weakened algorithm"
            plt_save = "Traing_weakened_student.png"
        else:
            plt_title = "PR curves of the training set, original algorithm"
            plt_save = "Traing_original_student.png"
    else:
        if weaken:
            plt_title = "PR curves of the test set, weakened algorithm"
            plt_save = "Test_weakened_student.png"
        else:
            plt_title = "PR curves of the test set, original algorithm"
            plt_save = "Test_original_student.png"
            
    # Plot training set PR curves
    plt.figure()
    plt.plot(R[0],P[0], label = 'IoU = 0.25')
    plt.plot(R[1],P[1], label = 'IoU = 0.50')
    plt.plot(R[2],P[2], label = 'IoU = 0.75')
    plt.xlim(0,1)
    plt.ylim(0,1)
    
    plt.title(plt_title)
    plt.legend()
    plt.savefig(plt_save)


# In[5]:


# set a path for predictions and annotations:
preds_path = 'data/hw02_preds'
gts_path = 'data/hw02_annotations'

# load splits:
split_path = 'data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data original and weakened 
'''
with open(os.path.join(preds_path,'preds_train04202321.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(preds_path,'preds_train_weakened04210852.json'),'r') as f:
    preds_train_we = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_students_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data original and weakened
    '''
    
    with open(os.path.join(preds_path,'preds_test04202321.json'),'r') as f:
        preds_test = json.load(f)
    
    with open(os.path.join(preds_path,'preds_test_weakened04210852.json'),'r') as f:
        preds_test_we = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_students_test.json'),'r') as f:
        gts_test = json.load(f)


# In[6]:


# Create PR curves

PR_curve(preds_train, gts_train, train=True, weaken=False)
PR_curve(preds_train_we, gts_train, train=True, weaken=True)

if done_tweaking:
    PR_curve(preds_test, gts_test, train=False, weaken=False)
    PR_curve(preds_test_we, gts_test, train=False, weaken=True)


# In[ ]:




