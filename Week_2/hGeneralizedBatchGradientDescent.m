function [J_vec,w_vec,b_vec] = hGeneralizedBatchGradientDescent(X,y,alpha,options)
% This function implements multi-variate stochastic gradient descent (SGD)
% with a single batch.
% 
% The inputs are as follows:
% X_train: An (m-by-n) matrix of training features, where 'm' is the number
% of training examples and 'n' is the number of features.
% 
% y_train: An (m-by-1) vector of training labels or targets.
% 
% alpha: The learning rate, a scalar value between 0 and 1.
%
% Optional inputs:
% initMethod: Determines the weight initialization method. Could be "zero"
% or "random" for unit-symmetric random initialization.
% 
% J_stop: The stopping cost value for the SGD algorithm. The algorithm
% terminates if the cost J is less than or equal to J_stop.
% 
% iter_stop: The maximum number of iterations for the SGD algorithm. The
% algorithm terminates if the number of iterations reaches iter_stop.
% 
% verbose: A boolean flag that determines whether to display the status of
% the SGD algorithm.
% 
% verboseFreq: An integer that specifies the frequency of status updates
% (if verbose is true).
% 
% acceleration: A boolean flag that controls the output history of the SGD.
% If true, only the results from the last iteration will be returned.
% Otherwise, the complete history of each iteration's parameters will be
% returned.

arguments
    X % features
    y % labels
    alpha % learning rate
    options.initMethod = "zero";
    options.J_stop = 1e-3;
    options.iter_stop = 1e3;
    options.verbose = true;
    options.verboseFreq = 100;
    options.accelaration = false;
end

% Global parameters
stopFlag = false;
iter = 0; % Iteration number
J_vec = [];
w_vec = [];
b_vec = [];

% Derived parameters
m = length(X); % number of samples
n = width(X); % number of features

% Step 1: Initialize the parameters w and b
if strcmpi(options.initMethod,"unit-symmetric")
    w = randsrc(n,1,-1:0.001:1);
    b = randsrc(1,1,-1:0.001:1);
elseif strcmpi(options.initMethod,"zero")
    w = zeros(n,1);
    b = 0;
end

% Step 2 - Repeat until the stopping criteria is met
while ~stopFlag

    % Update the weights and the bias
    dJdwj = zeros(m,n); % partial derivative w.r.t w
    dJdb = 0; % partial derivative w.r.t b
    for i = 1:m % rows (samples)
        for j = 1:n % columns (features)
            dJdwj(i,j) = (1/m)*( ( (X(i,:)*w)+b-y(i) )*X(i,j) );
        end
        dJdb = dJdb + (1/m)*( (X(i,:)*w)+b-y(i) );
    end
    dJdwj = sum(dJdwj); % Sum in i or example dimension, yields (1xn) vector

    % Simultaneous parameter update
    for j = 1:n
        w(j) = w(j) - alpha*dJdwj(j);
    end
    b = b - alpha*dJdb;
    
    % Record the updated parameters
    if ~options.accelaration % keep the history
        w_vec = cat(1,w_vec,w.');
        b_vec = cat(1,b_vec,b);
    end

    % Step 3 - Calculate the cost function and decide if the stopping
    % criteria is met
    J = 0;
    for i = 1:m
        J = J + (1/(2*m))*( ( (X(i,:)*w)+b-y(i) )^2 );
    end
    if ~options.accelaration % keep the history
        J_vec = cat(1,J_vec,J);
    end

    % Status display
    if options.verbose
        if mod(iter,options.verboseFreq) == 0
            disp(['Iteration #',num2str(iter),' | Cost: ',num2str(J)])
        end
    end

    % Check for the stopping condition, if met terminate the loop
    if iter == options.iter_stop - 1 || J <= options.J_stop 
        stopFlag = true;
        disp(['* BGD stopped: max number of iterations ',sprintf('(%d)',iter)])
        disp(repmat('=',1,40))
    elseif J <= options.J_stop
        stopFlag = true;
        disp(['* BGD stopped: minimized cost function ',sprintf('(%f.2)',J)])
        disp(repmat('=',1,40))
    end

    iter = iter + 1; % increase the iteration number
end

% Return the final values
if options.accelaration
    w_vec = w.';
    b_vec = b;
    J_vec = J;
end

% Display the final values
disp(['Parameters found using BGD ',sprintf('w: [ %.4f ], ',w_vec(end,:)),sprintf('b: %.4f',b_vec(end))])
end