import cv2
import numpy as np
from PIL import Image
from skimage.measure import label
from features import *


def convert_colorspace(img, color_space='RGB'):
    if color_space == 'RGB':
        feature_image = np.copy(img)      
    elif color_space == 'HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif color_space == 'YCrCb':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        raise ValueError('Color space is not found.')
        
    return feature_image


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap < threshold] = 0
    heatmap[heatmap > 0] = 1
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    box_list = []
    
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        box_list.append(bbox)

    # Draw boxes on the image
    out_img = draw_boxes(img, box_list, color=(0, 0, 255), thick=6)
        
    return out_img


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    
    # Create a list to append feature vectors to
    features = []
    
    # Iterate through the list of images
    for image in imgs:
        file_features = []
        
        # Read in each one by one
        #image = np.asarray(Image.open(file))
        
        # apply color conversion if other than 'RGB'
        feature_image = convert_colorspace(image, color_space)
            
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
            
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
            
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        
        if len(file_features) == 0:
            raise ValueError('Feature vector is empty.')
            
        features.append(np.concatenate(file_features))
        
    # Return list of feature vectors
    return features


def find_cars(img, svc, X_scaler, y_start_stops, scales, window, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat):
    
    box_list = []
    
    for scale, (y_start, y_stop) in zip(scales, y_start_stops):
        img_tosearch = img[y_start:y_stop,:,:]
        #ctrans_tosearch = convert_colorspace(img_tosearch, color_space=color_space)

        if scale != 1:
            imshape = img_tosearch.shape
            img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        nxsteps = (img_tosearch.shape[1] - window) // pix_per_cell + 1
        nysteps = (img_tosearch.shape[0] - window) // pix_per_cell + 1

        for xc in range(nxsteps):
            for yc in range(nysteps):
                xleft = xc * pix_per_cell
                ytop = yc * pix_per_cell

                # Extract the image patch
                subimg = img_tosearch[ytop:ytop+window, xleft:xleft+window]

                # Extract features and make a prediction
                test_features = X_scaler.transform(extract_features([subimg], 
                                                                    color_space, spatial_size,
                                                                    hist_bins, orient,
                                                                    pix_per_cell, cell_per_block,
                                                                    hog_channel, spatial_feat,
                                                                    hist_feat, hog_feat))
                
                test_prediction = svc.predict(test_features)

                if test_prediction[0] == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    bbox = ((xbox_left, ytop_draw+y_start), (xbox_left+win_draw, ytop_draw+win_draw+y_start))
                    box_list.append(bbox)
                    
        out_img = draw_boxes(img, box_list, color=(0, 0, 255), thick=6)
                    
    return out_img, box_list