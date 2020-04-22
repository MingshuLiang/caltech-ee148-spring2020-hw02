import os
import numpy as np
import json
from PIL import Image, ImageDraw





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
    print(nrTh)
    ncTh = int((n_cols_T-1)/2) # for n_cols_T_h
    print(ncTh)
    
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
    heatmap = 0.8*heatmap_rgb[:,:,0]+0.1*heatmap_rgb[:,:,1]+0.1*heatmap_rgb[:,:,2]

    return heatmap





def predict_boxes(heatmap_o, n_rows_T, n_cols_T, filename):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''
    
    n_rows = np.shape(heatmap_o)[0]
    n_cols = np.shape(heatmap_o)[1]
    
    output = []
    bounding_boxes = [] # for visualization

    '''
    BEGIN YOUR CODE
    '''
    
    nrTh = int((n_rows_T-1)/2) # for n_rows_T_half
    ncTh = int((n_cols_T-1)/2) # for n_cols_T_half
    
    heatmap = np.where(heatmap_o > 0.92, heatmap_o ,0)
    
    # Draw threshold heatmap
    Ht_name = filename.replace('.jpg','heatmap_t.png')
    
    I_heatmap = (heatmap * 255 / np.max(heatmap)).astype('uint8')
    I_heatmap = Image.fromarray(I_heatmap)
    #I_heatmap.show()
    I_heatmap.save(os.path.join('data/visualization',Ht_name))
    
    
    while np.any(heatmap != 0):
        score = np.amax(heatmap)
        idx = np.where(heatmap == score)
        c_row = idx[0].item(0) # c for center
        c_col = idx[1].item(0) # c for center
        tl_row = c_row-nrTh
        tl_col = c_col-ncTh
        #print(tl_row,tl_col)
        br_row = tl_row + 7#n_rows_k
        br_col = tl_col + 7#n_cols_k
        output.append([tl_row,tl_col,br_row,br_col, score])
        
        # for visualization
        bounding_boxes.append([tl_col,tl_row,br_col,br_row])
        
        top = np.max([c_row-n_rows_T,0])
        bottom = np.min([c_row+n_rows_T,n_rows])
        left = np.max([c_col-n_cols_T,0])
        right = np.min([c_col+n_cols_T,n_cols])
        
        heatmap[top:bottom,left:right] = 0
        

    '''
    END YOUR CODE
    '''

    return output, bounding_boxes





def detect_red_light_mf(I, filename):
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
    # set the path to heatmap:
    H_path = 'data/visualization'
    H_name = filename.replace('.jpg','heatmap.png')
    
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
    
    # Draw heatmap
    heatmap = compute_convolution(I, T)
    I_heatmap = (heatmap * 255 / np.max(heatmap)).astype('uint8')
    I_heatmap = Image.fromarray(I_heatmap)
    #I_heatmap.show()
    I_heatmap.save(os.path.join(H_path,H_name))
    
    (n_rows_T,n_cols_T,n_channels_T) = np.shape(T) 
    
    output, bounding_boxes = predict_boxes(heatmap,n_rows_T,n_cols_T,filename)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output, bounding_boxes





# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = 'data/RedLights2011_Medium'

# load splits: 
split_path = 'data/hw02_splits'
file_names_train = ["RL-076.jpg","RL-107.jpg","RL-012.jpg","RL-278.jpg","RL-167.jpg","RL-268.jpg","RL-213.jpg"] #failed examples
#succeeded examples: ["RL-016.jpg","RL-096.jpg","RL-217.jpg","RL-285.jpg","RL-199.jpg","RL-217.jpg"]
#file_names_test = np.load(os.path.join(split_Path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = 'data/hw02_preds'
#os.makedirs(preds_path, exist_ok=True) # create directory if needed

gts_path = 'data/hw02_annotations'

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

with open(os.path.join(gts_path, 'formatted_annotations_students.json'),'r') as f:
    gts_train = json.load(f)

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]], bounding_box = detect_red_light_mf(I,file_names_train[i])
    

    I = Image.fromarray(I)
    
    # Draw prediction
    for j in range(len(preds_train[file_names_train[i]])):
        boundingbox = ImageDraw.Draw(I)   
        boundingbox.rectangle(bounding_box[j], fill = None, outline ="yellow")
        print(type(bounding_box[j]))
    
    # Draw ground truth
    for j_gts in range(len(gts_train[file_names_train[i]])):
        gts = gts_train[file_names_train[i]][j_gts].copy()
        gts[0] = gts_train[file_names_train[i]][j_gts][1]
        gts[1] = gts_train[file_names_train[i]][j_gts][0]
        gts[2] = gts_train[file_names_train[i]][j_gts][3]
        gts[3] = gts_train[file_names_train[i]][j_gts][2]
        gts_int = [int(item) for item in gts]
        
        boundingbox = ImageDraw.Draw(I)   
        boundingbox.rectangle(gts_int, fill = None, outline ="green")
    
    #I.show()
    I.save(os.path.join('data/visualization',file_names_train[i].replace('.jpg','.png')))
    
# save preds (overwrites any previous predictions!)
#with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
#    json.dump(preds_train,f)

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
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
