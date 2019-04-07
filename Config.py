class Config():
    window = 64 # Sliding window width and height
    
    # Hyperparameters for feature extraction:
    color_space = 'YCrCb' # Can be RGB, HSV or YCrCb
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16   # Number of histogram bins
    orient = 9 # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

    # Whether to extract certain features:
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    
    # Search ranges and scale factors:
    y_start_stops = [(400, 464), (400, 496), (432, 560), (448, 644)] # Min and max in y to search in sliding window
    scales = [1, 1.5, 2., 3.]