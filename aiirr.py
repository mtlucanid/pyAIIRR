#
# aiirr.py
#
#   Python imprementation for computation of AIIRR
#   (Area Integrity Index with Random Rearrangement)
#
#   Copyright (C) 2026, Masahiko Tanahashi
#
#   Last modified: 18 Aug 2026
#   First release: xx Aug 2026
#
#   E-mail: m.tanahashi.lucanid@gmail.com
#
# ------------------------------------------------------------------------
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://gnu.org>.
#
# ------------------------------------------------------------------------
#
# The latest version of this file will be avairable from GitHub repository of the author:
#  https://github.com/mtlucanid/AIIRR_python
#
# Before use, please install Python libraries from the command shell:
#  $ pip install numpy opencv-python screeninfo
#
# Example Usage (see also 'run_aiirr.py'):
# [Create 'YOUR_PYTHON_FILE.py' under the same path of 'aiirr.py' and type the following code]
#   import aiirr
#   results = compute_aiirr(
#       img_file = "your_image.jpg",
#       mask_file = "your_mask_pattern.png",
#       select_darker_pixel = True,
#       num_gaussian_filters = 25,
#       shake_amplitude = 20,
#       num_bootstrap_iterations = 50,
#   )
#   print(f"Mean Area Integrity Index (AII): {results['mean_aii']:.4f}")
#   print(f"Average Regions: {results['mean_nregions']:.1f}")
#
# For the details of AIIRR, please refer to our original papers:
#  Tanahashi M, Lin M-C, Lin C-P (2025). Area Integration Index with Random Rearrangement
#     (AIIRR): a new concept for quantifying disruptive colorations. Methods in Ecology
#     and Evolution, 16(8), 1781–1795. https://doi.org/10.1111/2041-210X.70085
#
#  Tanahashi M, Huang J-P (2026). pyAIIRR: A Python program for Area Integration Index with
#     Random Rearrangement (AIIRR). To be submitted to XXX.

import cv2
import sys
import math
import random
import numpy as np
from screeninfo import get_monitors

##########################################################
## Constants (*which does not need to change frequently)
MASK_THRESHOLD = 127     # When mask is a grayscale image, pixels less than this will be treated as zero (default: 127) 
OUTER_GREY_VALUE = 127   # Paint outside ROI in this color (default: 127)
GAUSSIAN_FILTER_SIZE = 5 # Size of Gaussian filter (default: 5)
NOISE_THRESHOLD = 0.001  # Ignore small region less than [area_roi * NOISE_THRESHOLD] (default: 0.001)
                         #  Use 0 (or None) will improve the calculation speed

##########################################################
## Enumerative Constants
SELECT_LIGHTER = 0
SELECT_DARKER = 1
SELECT_BOTH = 2

##########################################################
## The main function of this module
def compute_aiirr(
    img: np.ndarray = None,
    mask: np.ndarray = None,
    img_file: str = None,
    mask_file: str = None,
    region_select: int = SELECT_DARKER,
    num_gaussian_filters: int = 25,
    shake_amplitude: float = 20,
    shake_aspect_ratio: float = None,
    shake_deflection_angle: float = None,
    num_bootstrap_iterations: int = 50,
    random_seed: int = None,
    output_prefix: str = None,
    output_images: bool = False,
    animation: bool = True,
    animation_wait: int = 100,
    console: bool = True,
) -> dict:
    """Calculates AIIRR (Area Integrity Index with Random Rearrangement) 

    Parameters:
    -----------
    img : np.ndarray
        Target image with camouflage pattern.
    mask : np.ndarray
        Binary mask outlining the subject/object boundary.    
    img_file : str (optional)
        Path to the target input image with camouflage pattern.
    mask_file : str (optional)
        Path to binary mask outlining the subject/object boundary.    
    select_darker_pixel : bool
        A flag whether the program use darker area as the background that is separated by colorations.  
    num_gaussian_filters : int (n)
        Number of Gaussian blur iterations applied (pre/post blurring).
    shake_amplitude : float (d)
        Maximum pixel displacement during random rearrangement.
    shake_aspect_ratio : float (or, None)
        Aspect ratio (0 to inf) in the direction of deflection angle. Set this to None disable the directional shake.
    shake_deflection_angle : float (or, None)
        Deflection angle (0 to 180) for directional shake operation. Set this to None disable the directional shake.
    num_bootstrap_iterations : int (N)
        Number of random rearrangement bootstrapping iterations.
    random_seed : int (or, None)
        Random number seed. Setting this to None will use the system default (initialized by time at every run) 
    animation : bool
        Enable animation during the bootstrap calculations.   
    animation_wait : int
        Duration (milliseconds) to show each image in the animation. 
    console : bool
        Enable console output during the bootstrap calculations. Setting this to False will show a simple progress.

    Returns:
    --------
    dict containing:
        mean_aii      : Mean AII value across iterations (Range: 0 to 1)
        mean_nregions : Average number of valid separated regions
    """

    # 0. Initialize the random seed
    if random_seed is not None:
        random.seed(random_seed)

    # 1. Load image in grayscale
    if img is None:
        # cv2.imread() with cv2.IMREAD_GRAYSCALE flag returns a numpy.ndarray
        # with the data type (dtype) of numpy.uint8 (0 to 255)
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_file}")
    else:
        if img_file is not None:
            print(f"Ignoring img_file={img_file} as img is specified")

    # -> Start animation
    if animation:
        imshow_init(img)
        imshow(img, animation_wait * 10)

    # 2. Mask handling
    if mask is None:
        if mask_file is not None:
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        mask = np.full_like(img, 255, np.uint8)
    else:
        # cv2.threshold() returns a np.ndarray of the same type of input array (np.uint8)     
        _, mask = cv2.threshold(mask, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
      

    # 3. Area and RSS of the entire ROI
    _y, _x = np.where(mask != 0)
    if len(_x) == 0:
        return {
            "aiirr_mean_aii": None,
            "aiirr_disruption_score": None,
            "mean_num_regions": 0,
        }
    area_roi = len(_x)
    rss_roi = compute_rss(_y, _x)

    # 4. Set the minimum region size
    if NOISE_THRESHOLD and NOISE_THRESHOLD > 0:
        min_region_size = int(area_roi * NOISE_THRESHOLD)
    else:
        min_region_size = None

    # -> Console output
    if console:
        print(
            f"Common_stats\n"
            f"area_roi   : {area_roi}\n"
            f"min_region : {min_region_size}\n"
            f"rss_roi    : {rss_roi:.2f}\n"
        )

    # 5. Threshold the image inside the mask (generate a 8-bit grayscale image (np.uint8))
    binary_pattern = threshold_inside_mask(img, mask, (region_select == SELECT_DARKER))

    # 6. Paint the outside area in gray
    binary_pattern[mask == 0] = OUTER_GREY_VALUE

    # -> Show the animation
    if animation:
        imshow(binary_pattern, animation_wait * 10)

    # 7. Pre-blur filtering
    blurred_img = binary_pattern.copy()
    for _ in range(num_gaussian_filters):
        blurred_img = cv2.GaussianBlur(
            blurred_img,
            (GAUSSIAN_FILTER_SIZE, GAUSSIAN_FILTER_SIZE),
            0,
        )

    # 8. Bootstrap Loop (Random Rearrangement)
    aii_list = []
    nregion_list = []
    for bt in range(num_bootstrap_iterations):

        # 9. Generate two random numbers: u and z
        #  u : Uniform distribution, [0, 1)
        #  z : Normal distribution, N(0, 1)
        u = random.random()
        z = random.gauss(0, 1)
        
        # 10. Determine the shake vector, (dx, dy)
        #  theta : Random shake angle (= 2PIu)
        #  phi   : Deflection angle
        #  r     : Aspect ratio
        #
        #      | dx |        | cos(phi) -sin(phi) |  | r * cos(theta) |
        #  v = |    | = |zd| |                    |  |                |
        #      | dy |        | sin(phi)  cos(phi) |  |     sin(theta) |
        #
        theta = u * math.pi * 2
        length = abs(z * shake_amplitude)
        
        if (shake_deflection_angle is not None) and (shake_aspect_ratio is not None):
            phi = shake_deflection_angle / 180 * math.pi
            r = shake_aspect_ratio
        else:
            phi = 0
            r = 1
            
        cp = math.cos(phi)
        sp = math.sin(phi)
        ct = math.cos(theta)
        st = math.sin(theta)
    
        dx = round(length * (cp * r * ct - sp * st))
        dy = round(length * (sp * r * ct + cp * st))

        #* Alternatively, you can simply generate random pixel shift like as:
        #*  dx = np.random.randint(-shake_amplitude, shake_amplitude + 1)
        #*  dy = np.random.randint(-shake_amplitude, shake_amplitude + 1)

        # 10. Generate a shifted image
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_img = cv2.warpAffine(
            blurred_img, M, (blurred_img.shape[1], blurred_img.shape[0])
        )
        
        # 11. Blend the original and shifted images
        blended = cv2.addWeighted(blurred_img, 0.5, shifted_img, 0.5, 0)

        # 12. Generate a shifted mask (intermediate of the original and shifted images)
        M = np.float32([[1, 0, int(dx/2)], [0, 1, int (dy/2)]])
        shifted_mask = cv2.warpAffine(
            mask, M, (mask.shape[1], mask.shape[0]),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        # 13. Thresholding inside the mask
        resegmented = threshold_inside_mask(blended, shifted_mask)

        # 14. Area thresholdings: internally runs the connected component
        #     analysis twice (see connectedComponents2())
        if min_region_size:
            resegmented = areaThreshold(
                *connectedComponents2(
                    resegmented,
                    shifted_mask,
                    add_bg = True, # ALWAYS be True to remove small background regions
                ), # Four return values will be unpacked here
                min_region_size,
                min_region_size,
            ) 
        
        # 15. Connected component analysis
        num_labels, labels, stats, _ = connectedComponents2(
            resegmented,
            shifted_mask,
            add_bg = (region_select == SELECT_BOTH),
        )

        # 16. Calculate AII for this filtered image
        area_total, nregions, rss_total, rss_within, aii = compute_aii(num_labels, labels)

        # 17. Add values of this bootstrap iteration to the lists
        aii_list.append(aii)
        nregion_list.append(nregions)

        # -> Console output
        if console:
            print(
                f"Iteration  : {bt + 1}\n"
                f"area_total : {area_total}\n"
                f"nregions   : {nregions}\n"
                f"rss_total  : {rss_total:.2f}\n"
                f"rss_within : {rss_within:.2f}\n"
                f"AII        : {aii:.4f}\n"
            )
        # otherwise, show a progress bar if the standard output is not a file or a pipe
        elif sys.stdout.isatty():
            if bt+1 < num_bootstrap_iterations:
                print(f"\rIteration: {bt+1}/{num_bootstrap_iterations}", end="", flush=True)
            else:
                print(f"\rIteration: {bt+1}/{num_bootstrap_iterations}")

        # -> Show the animation
        if animation:

            # Get indices of labels sorted by area (cv2.CC_STAT_AREA = 4), skipping background (label 0)
            label_indices = list(range(1, num_labels))
            label_indices.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)  # Largest first

            # Create an empty BGR color output image
            output_img = np.zeros((*resegmented.shape, 3), dtype=np.uint8)

            # Number of colors used
            ncolors = 8 # or, max(1, len(label_indices))

            # Assign colors based on size rank (e.g., using a colormap or custom palette)
            for rank, label in enumerate(label_indices):
                # Generate a distinct color (e.g., scaling hue or picking from a colormap)
                # Here we map the rank to a 0-255 hue value for HSV, or pick distinct BGR colors
                hue = int(180 * (rank % ncolors) / ncolors)
                color_hsv = np.uint8([[[hue, 255, 255]]])
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
                # Paint the component pixels
                output_img[labels == label] = color_bgr

            # Show the colored output
            imshow(output_img, animation_wait)

    # -> Finish the output after waiting for a while
    if animation:
        imshow_wait(animation_wait * 10)
        imshow_exit()

    # 18. Summarize final metrics
    mean_aii = float(np.mean(aii_list))
    return {
        "mean_aii"      : mean_aii,
        "mean_nregions" : float(np.mean(nregion_list)),
    }

##############################################################
## Core routine for AII computation
##  (labels of background regions must be set to 0) 
def compute_aii(num_labels: int, labels: np.ndarray):
    """Calculates Area Integrity Index (AII) from a binary image."""
    # Area and RSS of the total regions
    _y, _x = np.where(labels != 0)
    area_total = len(_x)
    rss_total = compute_rss(_y, _x)

    # Sum of RSS for separated regions (within-region RSS)
    rss_within = 0.0
    nregions = 0
    for i in range(1, num_labels):  # Skip background (label 0)
        _y, _x = np.where(labels == i)
        if len(_x):
            nregions += 1
            rss_within += compute_rss(_y, _x)

    # Compute AII for this bootstrap iteration
    aii = rss_within / rss_total if rss_total > 0 else 1.0

    # Returns
    return area_total, nregions, rss_total, rss_within, aii

###########################################
## Helper functions for image outputs
def imshow_init(img):
    """Prepare a fixed-sized window to show animations"""
    # Close previous windows
    cv2.destroyAllWindows()
    # Fetch the primary monitor's resolution
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height
    height, width = img.shape
    # Resize the image so that it does not exceed the screen size
    if width > screen_width:
        scale = screen_width / width
        width *= scale
        height *= scale
    if height > screen_height:
        scale = screen_height / height
        width *= scale
        height *= scale
    # Precreate the named window with a fixed size (to avoid window flashing)
    width = int(width)
    height = int(height)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", width, height)

def imshow(img, wait):
    cv2.imshow("Image", img)
    if wait > 0:
        cv2.waitKey(wait)

def imshow_wait(wait):
    if wait > 0:
        cv2.waitKey(wait)

def imshow_exit():
    cv2.destroyAllWindows()

############################################
## Helper functions for stats
def compute_rss(y_coords: np.ndarray, x_coords: np.ndarray) -> float:
    """Calculates spatial Residual Sum of Squares (Sxx + Syy) for a set of pixel coordinates."""
    if len(x_coords) == 0:
        return 0.0

    mean_x = np.mean(x_coords)
    mean_y = np.mean(y_coords)

    # RSS = Sxx + Syy = sum((x - mean_x)^2) + sum((y - mean_y)^2)
    s_xx = np.sum((x_coords - mean_x) ** 2)
    s_yy = np.sum((y_coords - mean_y) ** 2)

    return float(s_xx + s_yy)
    
#############################################
## Helper functions for image operations
def threshold_inside_mask(
    img: np.ndarray,
    mask: np.ndarray,
    select_darker_pixel: bool = False
) -> np.ndarray:
    """Apply Otsu's threshold to the image inside the mask"""

    # Extract only the pixel values inside the masked area
    # This flattens the masked pixels into a 1D NumPy array
    masked_pixels = img[mask != 0]

    # Calculate Otsu's threshold value from these pixels
    # Since it's a 1D array, cv2.threshold will only look at the region's histogram
    thresh_val, _ = cv2.threshold(
        masked_pixels, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # Apply the calculated threshold to the entire image
    if select_darker_pixel:
        _, binary_pattern = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY_INV)
    else:
        _, binary_pattern = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)

    # Crop the image
    binary_pattern = cv2.bitwise_and(binary_pattern, mask)

    # Convert binary pattern into grayscale
    binary_pattern = binary_pattern.astype(np.uint8) #* 255

    # Return the np.uint8 binary pattern
    return binary_pattern

######################################################################
## Run cv2.ConnectedComponents for foreground and background regions
def connectedComponents2(
    binary: np.ndarray,
    mask  : np.ndarray = None,
    add_bg: bool = False,
):
    """Run cv2.ConnectedComponents for foreground and background regions
    Returns:
      num_labels : int
      labels     : 2D numpy.ndarray with a data type of signed 32-bit integer (int32 / cv2.CV_32S) by default.
      stats      : 2D numpy.ndarray with a data type of int32
      num_fg     : int
    """
    # Apply mask if avairable
    if mask is not None:
        binary = cv2.bitwise_and(binary, mask)

    # Separate foreground regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    n = num_labels

    # Also separate background regions and add them after the end of foreground data
    if add_bg:

        # Invert image (inside the mask, if avairable)
        binary_i = cv2.bitwise_not(binary, mask = mask)

        # Separate inverted image
        num_labels_i, labels_i, stats_i, _ = cv2.connectedComponentsWithStats(binary_i)

        # Total num of regions
        n += num_labels_i - 1

        # Merge labels
        for i in range(1, num_labels_i):
            j = num_labels + i - 1
            labels[labels_i == i] = j

        # Concatenate two stats without the first one of inverted image
        stats = np.vstack([stats, stats_i[1:]])

    # Returns
    return n, labels, stats, num_labels

######################################################################
## Run cv2.ConnectedComponents for foreground and background regions
def areaThreshold(
    num_labels: int,
    labels    : np.ndarray,
    stats     : np.ndarray,
    num_fg    : int = None,
    thresh_fg : int = None,
    thresh_bg : int = None,
) -> np.ndarray:
    """ Regions indicated by '0' to 'num_fg - 1' in labels are the foreground patterns
    """
    # Create a new image
    binary_pattern = np.zeros_like(labels, dtype = np.uint8)

    # When ommiting the last 3 arguments
    if num_fg is None:
        num_fg = num_labels
        thresh_bg = None

    # When background operation is not required
    if num_fg == num_labels:
        thresh_bg = None

    # Apply foreground area threshold
    if thresh_fg:
        for i in range(1, num_fg):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < thresh_fg:
                pass
            else:
                binary_pattern[labels == i] = 255
    else:
        for i in range(1, num_fg):
            binary_pattern[labels == i] = 255

    # Apply background area threshold
    if thresh_bg:
        for i in range(num_fg, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < thresh_bg:
                binary_pattern[labels == i] = 255
            else:
                pass

    # Return the binary image
    return binary_pattern
        
##################################################
## Message when user directlu run this module
if __name__ == '__main__':
    input(
        "\n"
        "--------------------------------------------------\n"
        " aiirr.py                                         \n"
        "                                                  \n"
        "  Python imprementation for computation of AIIRR  \n"
        " (Area Integrity Index with Random Rearrangement) \n"
        "                                                  \n"
        "     Copyright (C) 2026, Masahiko Tanahashi       \n"
        "                                                  \n"
        "     PLEASE DO NOT DIRECTLY RUN THIS FILE!        \n"
        "     (see the Example Usage in this file)         \n"
        "--------------------------------------------------\n"
        "Hit any key"
    )