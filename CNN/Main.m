%% Preparing the workspace
close all
clear
clc

%% Loading the image dataset
% as .mat file %%%%%% later
% Directory setup
rootFolder = fullfile('Training set');

% Categories
categories = {'c','delta', 'i', 'noworm','s'};

% Create ImageDatastore
imds = imageDatastore(fullfile(rootFolder,categories),...
    'LabelSource','foldernames');

% %  Splitting the image dataset into
%  [imdsTrain, imdsTest] = splitEachLabel(imds,0.7,'randomize');

%% Shuffling the the entire dataset
% This will give us some randomization adds to better generalization
imdsTrain = shuffle(imds);

%% Network parameters
inputSize = [80 80 1];      % Image size with single channel
numClasses = 5;             % Number of labels
miniBatchSize  = 128;        % Mini Batch size
validationFrequency = 64;   %floor(numel(YTrain)/miniBatchSize);
kfold = 2;                  % Number of equal partition of dataset
epochs_new = 15;                % Number of epochs
Learning_rate_reg = 0.0001;
L2_regularization_reg = 1;      % initial learning rate
%momentum = 1;               % parameter updates previous iteration to current iteration
kernel_conv_mlenet = 5;         % Size of the filter
numfilt_mlenet =64;             % Number of filters in the convolutional layer
kernel_pool_mlenet = 2;         % Filter size of pooling
kernel_conv_mlenet2 = 5;        % [layer 2] Size of the filter
numfilt_mlenet2 = 128;          % [layer 2] Number of filters in the convolutional layer
kernel_pool_mlenet2 = 2;        % [layer 2] Filter size of pooling

%% Hyperparameter tuning
%Splitting the dataset
rng(1)
[imdsTrain1, imdsValhyp, imdsTesthyp] = splitEachLabel(imds,0.6,0.2,'randomize');

% This will give us some randomization adds to better generalization
imdsTrain = shuffle(imdsTrain1);

augmenter = imageDataAugmenter('RandRotation',@()randi([0,1],1)*90);
imdsTrainhyp= augmentedImageDatastore(inputSize,imdsTrain,'DataAugmentation',augmenter);

%% Evaluation with tuned parameters
% Layer
layers_eval = [
    imageInputLayer(inputSize,'Normalization','rescale-symmetric')
    convolution2dLayer(kernel_conv_mlenet,numfilt_mlenet,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(kernel_pool_mlenet,'Stride',2)
    convolution2dLayer(kernel_conv_mlenet2,numfilt_mlenet2)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(kernel_pool_mlenet2,'Stride',2)
    fullyConnectedLayer(numClasses, 'WeightsInitializer','he')
    softmaxLayer
    classificationLayer
    ];

% Training options
options = trainingOptions(...
    'adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',epochs_new, ...
    'InitialLearnRate',Learning_rate_reg, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',validationFrequency, ...
    'ValidationData',imdsTesthyp, ...
    'Verbose',true, ...
    'executionEnvironment','parallel', ...
    'L2Regularization', L2_regularization_reg);
%    'L2Regularization',L2_regularization_reg, ...

% % Training the network
[net, info_evl1] = trainNetwork(imdsTrainhyp, layers_eval, options);


%% Evaluate
Number_iteration_evl = info_evl1.TrainingAccuracy;
Num_ele_evl = numel(Number_iteration_evl);
index_evl =(Num_ele_evl/epochs_new)*(1:epochs_new);

% storing parameter for ploting each epoch (accuracy and loss)
epoch_train_acc = (info_evl1.TrainingAccuracy(index_evl))';
epoch_test_acc = (info_evl1.ValidationAccuracy(index_evl))';
epoch_train_loss = (info_evl1.TrainingLoss(index_evl))';
epoch_test_loss = (info_evl1.ValidationLoss(index_evl))';

% Predicting
pred_train = classify(net, imdsTrainhyp);
[pred_test, probs] = classify(net, imdsTesthyp);

% Training accuracy
YTrain = imdsTrain.Labels;
eval_acc_train = mean(pred_train == YTrain);

% Test accuracy
YTest = imdsTesthyp.Labels;
eval_acc_test = mean(pred_test == YTest);

% Training loss
eval__loss_train = 1 - mean(pred_train == YTrain);

% Test loss
eval_loss_test = 1 - mean(pred_test == YTest);

% Plotting
figure()
title('Accuracy with tunned parameter')
plot(1:epochs_new, [epoch_train_acc epoch_test_acc])
xlabel('No. of epochs')
ylabel('Accuarcy')
legend('Training', 'Testing')

figure()
title('Loss with tunned parameter')
plot(1:epochs_new, [epoch_train_loss epoch_test_loss])
xlabel('No. of epochs')
ylabel('Loss')
legend('Training', 'Testing')

% Training accuracy
X1 = ['Training accuracy = ',num2str(eval_acc_train)];
disp(X1)
% Testing accuracy
X1 = ['Training accuracy = ',num2str(eval_acc_test)];
disp(X1)
% Training loss
X1 = ['Training loss = ',num2str(eval__loss_train)];
disp(X1)
% Testing loss
X1 = ['Testing loss = ',num2str(eval_loss_test)];
disp(X1)

%%
confusionMat = confusionmat(YTrain, pred_train);

precision = diag(confusionMat)./sum(confusionMat,2)

recall = diag(confusionMat)./sum(confusionMat,1)'

f1Scores = 2*(precision.*recall)./(precision+recall)

meanF1 =  mean(f1Scores)

%% confusion plot
% Confusion matrix for training
figure(13)
plotconfusion(YTrain,pred_train)
title('Confusion plot train set')

% Confusion matrix for validation
figure(14)
plotconfusion(YTest,pred_test)
title('Confusion plot test set')