# Worm-body-pose-classification-

- The goal of this project is to classify the worms based on the body shape.
- In this project Convolutional Neural Network is trained and validated using 15K images. 

### Following steps are carried out, 
* Labelling the worm manually making it Supervised learning.
* Pre-processing the image using the python script from the Image processing folder.
* Once the images are pre-processed the Convolutional Neural Network (CNN) for image classification was implemented in MATLAB's Deep Learning Toolbox. 

### Below is a brief summary of the MATLAB script's functionality:
* The script starts by preparing the workspace, clearing it of any existing variables and closing all open figures.
*It then loads the image dataset using the imageDatastore function from the Deep Learning Toolbox. The images are organized into subfolders according to their respective categories.
* The dataset is shuffled using the shuffle function to randomize the order of the images.
* The script sets various network parameters, such as input image size, number of classes, mini-batch size, and number of epochs.
* It then performs hyperparameter tuning by splitting the dataset into three parts: a training set, a validation set, and a test set. The training set is augmented using random rotations. The validation set is used to evaluate the network's performance during training, while the test set is used for final evaluation after training is complete.
* The network architecture is defined using a series of layers, including convolutional, batch normalization, relu, max pooling, fully connected, softmax, and classification layers. The layers are defined using the layer functions from the Deep Learning Toolbox.
* The network is trained using the trainNetwork function from the Deep Learning Toolbox, with the defined layers, training options, and augmented training set.
* The trained network is used to make predictions on the training and test sets.
* The accuracy and loss of the network on the training and test sets are calculated and plotted using MATLAB's plotting functions
