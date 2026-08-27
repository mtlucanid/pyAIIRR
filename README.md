# pyAIIRR

Python implement for computation of Area Integuity Index for Random Rearrangement (AIIRR)

## Requirements

- Python 3.1+

## Package requirements

- numpy
- cv2 (OpenCV for Python)
- screeninfo (optional, used in `imagehelper.py`)

## This repository contains
* aiirr.py

  The main python module for AIIRR computation. You can import this module in your python code and call `compute_aiirr()` function, as follows:
  ```python
  import aiirr # be sure that 'aiirr.py' exists in your project folder
  results = aiirr.compute_aiirr(
    img_file="01.jpg", 
    mask_file="01.png",
    region_select=aiirr.SELECT_DARKER,
    num_gaussian_filters=25,
    shake_amplitude=20,
    num_bootstrap_iterations=100,
    animation=True,
    animation_wait=100,
    console=False,
  )
  print(f"Input image file                : {filebody}.jpg")
  print(f"Mean Area Integrity Index (AII) : {results['mean_aii']:.4f}")
  print(f"Mean number of regions          : {results['mean_nregions']:.2f}")
  '''
* imagehelper.py
  
  A helper module for the graphical output of AIIRR process.
* run_aiirr.py
  
  A sample code to use `aiirr` module.
* 01.jpg / 02.jpg
  
  Sample images (insect specimen photographs).
* 01.png / 02.png
  
  Sample mask images (binary masks to specify the foreground regions).

## See also
* Original publication of AIIRR

  https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.70085
* GitHub repository for the original AIIRR program (for Windows)

  https://github.com/mtlucanid/AIIRR/
