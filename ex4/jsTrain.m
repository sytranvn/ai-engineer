args = argv();
if nargin != 2
    fprintf("%s <input> <output>", program_name());
    exit(1);
end

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

data = csvread(args{1});
X = data(:, 1:400);
y = data(:, 401);
m = size(X, 1);


initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

options = optimset('MaxIter', 50);

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

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));
csvwrite([args{2} '-weights-1.csv'], Theta1);
csvwrite([args{2} '-weights-2.csv'], Theta2);

fprintf("Output saved to:\n- %s\n- %s", [args{2} '-weights-1.csv'], [args{2} '-weights-2.csv']);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;