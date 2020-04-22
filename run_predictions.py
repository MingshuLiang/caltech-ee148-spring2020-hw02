import os
import numpy as np
import json
from PIL import Image, ImageDraw
from datetime import datetime



def compute_convolution(I_o, T, stride=1):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I_o)
    (n_rows_T,n_cols_T,n_channels_T) = np.shape(T) # "k" for "kernel"
    
    # zero padding
    nrTh = int((n_rows_T-1)/2) # for n_rows_T_h
    ncTh = int((n_cols_T-1)/2) # for n_cols_T_h   
    I = np.pad(I_o, ((nrTh,nrTh), (ncTh,ncTh), (0, 0)), 'constant')
    
    heatmap_rgb = np.zeros((n_rows,n_cols,n_channels))

    # normalize T
    # "ch" for channel
    T_n = np.zeros((n_rows_T,n_cols_T,n_channels_T))
    for ch in range(n_channels_T): 
        norm_T = np.linalg.norm(T[:,:,ch])
        T_n[:,:,ch] = T[:,:,ch]/norm_T
    
    for i in range(n_rows):
        for j in range(n_cols):
            for ch in range(n_channels):
                # normalize cropped image 
                norm_I = np.linalg.norm(I[i:i+n_rows_T,j:j+n_cols_T,ch])
                I_cropped_n = I[i:i+n_rows_T,j:j+n_cols_T,ch]/norm_I
                heatmap_rgb[i][j][ch] = np.sum(T_n[:,:,ch]*I_cropped_n)
    
    # Weighted combinatioin of RGB channels
    heatmap = 0.8*heatmap_rgb[:,:,0]+0.10*heatmap_rgb[:,:,1]+0.10*heatmap_rgb[:,:,2]

    return heatmap


def predict_boxes(heatmap_o, n_rows_T, n_cols_T):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''
    
    n_rows = np.shape(heatmap_o)[0]
    n_cols = np.shape(heatmap_o)[1]
    
    output = []

    '''
    BEGIN YOUR CODE
    '''
    
    nrTh = int((n_rows_T-1)/2) # for n_rows_T_half
    ncTh = int((n_cols_T-1)/2) # for n_cols_T_half
    
    heatmap = np.where(heatmap_o > 0.92, heatmap_o ,0)
    
    while np.any(heatmap != 0):
        score = np.amax(heatmap)
        idx = np.where(heatmap == score)
        c_row = idx[0].item(0) # c for center
        c_col = idx[1].item(0) # c for center
        tl_row = c_row-nrTh
        tl_col = c_col-ncTh
        #print(tl_row,tl_col)
        br_row = tl_row + 7 #n_rows_k
        br_col = tl_col + 7 #n_cols_k
        output.append([tl_row,tl_col,br_row,br_col, score])
        
        top = np.max([c_row-n_rows_T,0])
        bottom = np.min([c_row+n_rows_T,n_rows])
        left = np.max([c_col-n_cols_T,0])
        right = np.min([c_col+n_cols_T,n_cols])
        
        heatmap[top:bottom,left:right] = 0
        

    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    # set the path to kernel: 
    T_path = 'data/kernel'

    # get sorted list of kernels: 
    T_names = sorted(os.listdir(T_path)) 

    # remove any non-JPEG kernels: 
    T_names = [f for f in T_names if '.jpg' in f] 

    T = np.zeros((21,9,3))
    # load kernel
    for i in range(len(T_names)):

        # read image using PIL:
        T_c = Image.open(os.path.join(T_path,T_names[i]))

        # convert to numpy array:
        T += np.asarray(T_c)

    T /= len(T_names)

    heatmap = compute_convolution(I, T)
    
    (n_rows_T,n_cols_T,n_channels_T) = np.shape(T) 
    
    output = predict_boxes(heatmap,n_rows_T,n_cols_T)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output


# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = 'data/RedLights2011_Medium'

# load splits: 
split_path = 'data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = 'data/hw02_preds'
#os.makedirs(preds_path, exist_ok=True) # create directory if needed

# save preds (named with date and time)
now = datetime.now()
month = now.strftime("%m")
day = now.strftime("%d")
time = now.strftime("%H%M")
date_time = now.strftime("%m%d%H%M")

preds_name_train = 'preds_train_tweaked' + date_time + '.json'
preds_name_test = 'preds_test_tweaked' + date_time + '.json'

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):
    
    print(i)
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)
    

with open(os.path.join(preds_path,preds_name_train),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,preds_name_test),'w') as f:
        json.dump(preds_test,f)





