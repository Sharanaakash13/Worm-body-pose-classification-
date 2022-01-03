import os
import glob
import cv2
import numpy as np

# Input image directory
input_path = 'f:\\Term 2\\DUP\\New Labels refined\\New folder'
# Creating path for all images in all folders
print("[INFO] loading images...")
p = os.path.sep.join([input_path, '**', '*.png'])
print(p)

# Returns file list
file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]
print(f"[INFO] images found: {len(file_list)}")
g = len(file_list)


# Defining image processing function
def imagePreProcessing(path):
    # Loading an image from  the path
    img = cv2.imread(path)

    # Converting to gray scale image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive Thresholding
    # Blocksize value is 33 and constant as 8
    thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 8)

    # Removing noise using Connected component with stats
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, None, None, None, 8, cv2.CV_32S)

    # get CC_STAT_AREA component as stats[label, COLUMN]
    areas = stats[1:, cv2.CC_STAT_AREA]

    # Removing smaller area in the image
    result = np.zeros(labels.shape, np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >= 200:  # Here area below 200 is considered as noise
            result[labels == i + 1] = 255
    # Morphological operation
    kernel_dil = np.ones((2, 1), np.uint8)
    erode = cv2.erode(result, kernel_dil, iterations=1)

    # Resizing the image to 80*80
    img_resize = cv2.resize(erode, (80, 80), interpolation=cv2.INTER_AREA)

    return img_resize


# loop over the image paths
for filename in file_list:
    # Loads the image
    image = imagePreProcessing(filename)

    # Replace the image
    cv2.imwrite(filename, image)

    # Countdown of remaining images
    f = g - 1
    print(f'Remaining images :{f}')
    g = f