#
# run_aiirr.py
#
#  Example usage of aiirr module (aiirr.py)
#

import aiirr

# These values are usually fixed to the defalut values
# aiirr.OUTER_GREY_VALUE = 127  # Change if noisy regions appear near the object edge
# aiirr.NOISE_THRESHOLD = 0.001 # Set this to 0 (or None) will improve the computation speed

# Read 0?.jpg / 0?.png in the current directory and compute AIIRR
for i in range(1, 3):
    print()
    filebody = f"{i:02}"
    results = aiirr.compute_aiirr(
        img_file=f"{filebody}.jpg", 
        mask_file=f"{filebody}.png",
        region_select=aiirr.SELECT_DARKER,
        num_gaussian_filters=25,
        shake_amplitude=20,
        #shake_aspect_ratio=1,
        #shake_deflection_angle=0,        
        num_bootstrap_iterations=100,
        animation=True,
        animation_wait=100,
        #console=False,
    )
    print(f"Input image file                : {filebody}.jpg")
    print(f"Mean Area Integrity Index (AII) : {results['mean_aii']:.4f}")
    print(f"Mean number of regions          : {results['mean_nregions']:.2f}")
    