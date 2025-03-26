function [J_vec, w_vec, b] = hLogisticRegression(X_train, y_train, w_init, b_init, opts)
% This function plots the logistic regression fitting
%   Inputs:
%       X_train (m-by-n)        : Data, m examples with n features
%       y_train (m-by-1)        : target values
%       w_init (n-by-1)         : Initial values of model weights  
%       b_init (scalar)         : Initial value of model bias
%       opts.LearnRate (scalar) : Learning rate
%       opts.MinCost (scalar)   : Minimum cost value to stop training
%       opts.MaxIter (scalar)   : Max number of iterations to stop training
%       opts.Verbose (logical)  : Display the training progress
%
%   Outputs:
%       w_vec (nx1)             : Final weights
%       b (scalar)              : Final bias
arguments
    X_train {mustBeNumeric}
    y_train {mustBeNumeric}
    w_init {mustBeNumeric}
    b_init {mustBeNumeric}
    opts.LearnRate = 0.01
    opts.ParameterInit = 'random'
    opts.MinCost = 1e-8
    opts.MaxIter = 1e6
    opts.Verbose = true
    opts.DispFreq = []
end

if isempty(opts.DispFreq)
    opts.DispFreq = fix(opts.MaxIter/10);
end
alpha = opts.LearnRate;
numDispDigits = length(num2str(opts.MaxIter));

% Step 0: Initialization
m = size(X_train, 1); % number of samples
n = size(X_train, 2); % number of features

% Step 1: Parameter Initialization
w_vec = w_init;
b = b_init;
stopTraining = false;
iter = 0;
J_vec = [];

% Step 2: Loop until the termination condition has been satisfied
while ~stopTraining
    J_tmp = 0;
    dJdw = 0;
    dJdb = 0;
    for i = 1:m
        % Step 2.1: Calculate the cumulative outputs
        X_norm_i = X_train(i,:); % input normalization
        y_i = y_train(i);
        f_wb_i = 1/(1 + exp( -(X_norm_i*w_vec + b) ));
        error_i = f_wb_i - y_i;
        
        % Step 2.2: Calculate the cumulative gradients
        dJdw = dJdw + (1/m)*error_i.*X_norm_i;
        dJdb = dJdb + (1/m)*error_i;

        % Step 2.3: Calculate the cumulative cost
        L = -( y_i*log(f_wb_i) ) - ( (1-y_i)*log(1-f_wb_i) ); % binary cross-entropy loss
        J_tmp = J_tmp + (1/m)*L;
    end
    J_vec = cat(2, J_vec, J_tmp);

    % Step 2.2: Update the parameters w_vec and b
    w_vec = w_vec - alpha*dJdw.';
    b = b - alpha*dJdb;

    % Step 2.3: Display progress
    if opts.Verbose
        if mod(iter, opts.DispFreq) == 0
            fprintf(['Iteration: %' num2str(numDispDigits) 'd -- w_vec: [ ' repmat('%+8.4f ',1,length(w_vec)) ' ] | b: %+8.4f | Cost (J): %11.8f \n'], iter, w_vec.', b, J_vec(end));
        end
    end
    if (J_vec(end) > 0 && J_vec(end) <= opts.MinCost)
        stopTraining = true;
        if opts.Verbose && mod(iter, opts.DispFreq) ~= 0
            fprintf(['Iteration: %' num2str(numDispDigits) 'd -- w_vec: [ ' repmat('%+8.4f ',1,length(w_vec)) ' ] | b: %+8.4f | Cost (J): %11.8f \n'], iter, w_vec.', b, J_vec(end));
        end
        fprintf('\n----------\n');
        fprintf('Termination: Minimum cost reached.\n');
    elseif iter == opts.MaxIter - 1
        stopTraining = true;
        if opts.Verbose && mod(iter, opts.DispFreq) ~= 0
            fprintf(['Iteration: %' num2str(numDispDigits) 'd -- w_vec: [ ' repmat('%+8.4f ',1,length(w_vec)) ' ] | b: %+8.4f | Cost (J): %11.8f \n'], iter, w_vec.', b, J_vec(end));
        end
        fprintf('\n----------\n');
        fprintf('Termination: Maximum iterations reached.\n');
    end
    
    % Increment the iteration counter
    iter = iter + 1;
end
end