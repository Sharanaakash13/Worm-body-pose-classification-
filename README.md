# Worm-body-pose-classification-

- The goal of this project is to classify the worms based on the body shape.
- In this project Convolutional Neural Network is trained and validated using 15K images. 

### Following steps are carried out, 
* Labelling the worm manually making it Supervised learning.
* Pre-processing the image using the python script from the Image processing folder.
* Once the images are pre-processed the Convolutional Neural Network (CNN) for image classification was implemented in MATLAB's Deep Learning Toolbox. 

#### Below is a brief summary of the Python script's(image processing) functionality:
* This script performs image processing on a set of PNG images located in a specified directory and its subdirectories. 
* It imports the necessary libraries such as os, glob, cv2, and numpy for working with files, processing images, and handling arrays.

* The input directory path is specified as input_path. 
* The script uses the os module to join the input path with the filename pattern, i.e., "**/*.png", and creates a list of all PNG image file paths in the input directory and its subdirectories. This is done using glob.iglob() function.

* The script then defines a function imagePreProcessing() that takes a file path of an image as input and processes the image using various image processing techniques. * The function reads the image using cv2.imread(), converts it to grayscale using cv2.cvtColor(), applies adaptive thresholding using cv2.adaptiveThreshold(), removes noise using connected component analysis with statistics using cv2.connectedComponentsWithStats(), removes smaller area components using a threshold value of 200 pixels, applies morphological operations using cv2.erode() function, and finally resizes the image to a fixed size of 80x80 pixels using cv2.resize() function. The processed image is returned by the function.

* The main loop of the script iterates over all image file paths in the list, applies the imagePreProcessing() function to each image, overwrites the original image file with the processed image using cv2.imwrite() function, and displays the remaining number of images to be processed.

#### Below is a brief summary of the MATLAB script's functionality:
* The script starts by preparing the workspace, clearing it of any existing variables and closing all open figures.
*It then loads the image dataset using the imageDatastore function from the Deep Learning Toolbox. The images are organized into subfolders according to their respective categories.
* The dataset is shuffled using the shuffle function to randomize the order of the images.
* The script sets various network parameters, such as input image size, number of classes, mini-batch size, and number of epochs.
* It then performs hyperparameter tuning by splitting the dataset into three parts: a training set, a validation set, and a test set. The training set is augmented using random rotations. The validation set is used to evaluate the network's performance during training, while the test set is used for final evaluation after training is complete.
* The network architecture is defined using a series of layers, including convolutional, batch normalization, relu, max pooling, fully connected, softmax, and classification layers. The layers are defined using the layer functions from the Deep Learning Toolbox.
* The network is trained using the trainNetwork function from the Deep Learning Toolbox, with the defined layers, training options, and augmented training set.
* The trained network is used to make predictions on the training and test sets.
* The accuracy and loss of the network on the training and test sets are calculated and plotted using MATLAB's plotting functions
