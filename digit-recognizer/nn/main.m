%% Initialization
clear ; close all; clc

DEBUG = false;

%% Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 28x28 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

if DEBUG
  train = csvread('../data/train_200.csv');
else
  train = csvread('../data/train.csv');
end

train = train(2:end, :);
X = train(:, 2:end);
y = train(:, 1);

m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(m);
sel = sel(1:100);

displayData(X(sel, :));

if DEBUG
  fprintf('Program paused. Press enter to continue.\n');
  pause;
end


%% ================ Part 2: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =================== Part 3: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
if DEBUG
  options = optimset('MaxIter', 50);
else
  options = optimset('MaxIter', 100);
end

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

if DEBUG
  fprintf('Program paused. Press enter to continue.\n');
  pause;
end


%% ================= Part 4: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

if DEBUG
  fprintf('Program paused. Press enter to continue.\n');
  pause;
end


%% ================= Part 5: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

if DEBUG
  fprintf('Program paused. Press enter to continue.\n');
  pause;
end


%% ================= Part 6: Implement Predict (on test set) ===
%  Predict on test set.

if DEBUG
  test_X = csvread('../data/test_100.csv');
else
  test_X = csvread('../data/test.csv');
end

test_X = test_X(2:end, :);

test_pred = predict(Theta1, Theta2, test_X);
result = [(1:size(test_X, 1))' test_pred];

fid = fopen('submission.csv', 'w');
fdisp(fid, 'ImageId,Label');
csvwrite(fid, result, '-append');
fclose(fid);

fprintf('Finished writing to submission file.\n');
