#
# run_aiirr.py
#
#  Example of using the aiirr module (aiirr.py)
#
# Last modified: 27 Aug 2026; by Masahiko TANAHASHI

import aiirr

##########################################################################
# It is recommended to set these pseudo-constant values ​​to their default,
# but you can change them by uncommenting either of the follwing lines.
#
# aiirr.MASK_THRESHOLD = 127      # When mask is a grayscale image, pixels less than this will be treated as zero (default: 127).
# aiirr.OUTER_GREY_VALUE = 127    # Paint outside ROI in this color (default: 127).
# aiirr.GAUSSIAN_FILTER_SIZE = 5  # Size of Gaussian filter (default: 5).
# aiirr.NOISE_THRESHOLD = 0.001   # Ignore small region less than [area_roi * NOISE_THRESHOLD] (default: 0.001). Use 0 (or None) will improve the calculation speed.
#

##########################################################################
# Read multiple image files in the current directory and compute AIIRR
#
# 0?.jpg : an image that contains target object (e.g. insect).
# 0?.png : a binary mask image that specifies the region of the target object within the image. 

# Number of files to read
N_FILES = 2
for i in range(N_FILES):
    
    # Print a blank line before the run
    print()
    
    # Generate image file names 
    img_file  = f"{i+1:02}.jpg" # '01.jpg', '02.jpg', ...
    mask_file = f"{i+1:02}.png" # '01.png', '02.png', ...
    
    # Compute AIIRR
    results = aiirr.compute_aiirr(
        img_file=img_file, 
        mask_file=mask_file,
        region_select=aiirr.SELECT_DARKER,
        num_gaussian_filters=25,
        shake_amplitude=20,
        num_bootstrap_iterations=100,
        animation=True,
        animation_wait=100,
        console=False,
    )
    
    # Output results.
    # The return value ('results') from compute_aiirr() is a dictionary
    # that contains both the mean statistics of the bootstrap iterations
    # and numpy arrays of the raw data values.
    print(f"Input image file  : {img_file}")
    print(f"Mean AII          : {results['mean_aii']:.4f}")
    print(f"Mean region count : {results['mean_region_count']:.2f}")
    