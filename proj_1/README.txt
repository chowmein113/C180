name: Kevin Chow

SID: 3036803547

email: chowmein113@berkeley.edu



### How to run code ###
# SOME Global Variables
    DEBUG - will use matplotlib to plot the channels, the processed channels if set to true
    RESCALE - will use rescaling instead of resizing to preserve quality when downscaling if set to true
    PARALLEL_PRO - will use multithreading to speed up long NCC calculations if true
# Go down to main() in main.py:
    - Can edit FUNC_METHOD to print out to final plot comparison of img with and without alignments
    - CROP_PERCENT: percent to crop off of image when computing quality
    - can uncomment 1 of 4 functions:
        - sklearns canny edge detection
        - gaussian blur subtraction edge detection
        - color based detection
        - EDGE detection with Color based aligning
            - this is the default algorithm and is already uncommented
# RUN main.py
    - type in the name of the image you want to run and press enter, should be placed in a directory 
    called data as a child directory to root folder(dir should be placed along side
    images)
    - type in the lvl u want to do for pyramid scaling and press enter:
        - 1: no pyramid scaling
        - 0: automatic pyramid scaling until image is scaled to be leq to 100 pixels width and height
        - n: specify desired custom level to pyramid
    - if debug is true will see multiple plot graphs
    - other wise will see one final graph of aligned and non aligned image with the transition displacements
    and can be saved as a png file
    -after exiting final matplot will save aligned image in out directory with same file type as initially provided
    

