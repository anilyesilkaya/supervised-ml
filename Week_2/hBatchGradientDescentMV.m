function [w_vec,b_vec,J_vec] = hBatchGradientDescentMV(X_train,y_train,alpha,condStop,maxIter,verbose,verboseFreq)
% This function implements multi-variate stochastic gradient descent (SGD)
% with a single batch. The inputs are as follows:
%
% X_train: An (m-by-n) matrix of training features, where 'm' is the number
% of training examples and 'n' is the number of features. 
% 
% y_train: An (m-by-1) vector of training labels or targets.
% 
% alpha: The learning rate, a scalar value between 0 and 1.
% 
% condStop: The stopping cost value for the SGD algorithm. The algorithm
% terminates if the cost J is less than or equal to condStop.
% 
% maxIter: The maximum number of iterations for the SGD algorithm. The
% algorithm terminates if the number of iterations reaches maxIter.
% 
% verbose: A boolean flag that determines whether to display the
% status of the SGD algorithm.
% 
% verboseFreq: An integer that specifies the
% frequency of status updates (if verbose is true).

% Step (1) - Initialization
stopFlag = false;
m = size(X_train,1); % number of examples
n = size(X_train,2); % number of features
w = zeros(n,1);
b = 0;
iter = 0;
w_vec = [];
b_vec = [];
J_vec = [];

% Let's implement random initialization in order not to get stuck in a
% local minima, however, the problem is we have no idea about the range of
% those parameters. We can either randomly pick a value or initiate them to
% be zero

% Step (2) - Repeat until the stopping criteria is met
while ~stopFlag

    % Update the weights
    dJdwj = zeros(m,n); % partial derivative
    for j = 1:n
        for i = 1:m
            dJdwj(i,j) = (1/m)*( ( w.'*X_train(i,:).'+b-y_train(i) )*X_train(i,j) );
        end
    end
    dJdwj = sum(dJdwj); % Sum in i (example) dimension

    % Update the bias
    dJdb = 0; % partial derivative
    for i = 1:m
        dJdb = dJdb + (1/m)*( w.'*X_train(i,:).'+b-y_train(i) );
    end

    % Simultaneous update
    for j = 1:n
        w(j) = w(j) - alpha*dJdwj(j);
    end
    b = b - alpha*dJdb;
    
    % Record the updated parameters
    w_vec = cat(2,w_vec,w);
    b_vec = cat(1,b_vec,b);

    % Step (3) - Calculate the cost function and decide if the stopping
    % criteria is met
    J_tmp = 0;
    for i = 1:m
        J_tmp = J_tmp + (w.'*X_train(i,:).'+b-y_train(i))^2;
    end
    J = (1/(2*m))*J_tmp;
    J_vec = cat(1,J_vec,J);

    % Status display
    if verbose
        if mod(iter,verboseFreq) == 0
            dispData = [];
            dispData = cat(2,dispData,sprintf('Iteration #%d\t\t',iter));
            dispData = cat(2,dispData,sprintf('J = %.2f\t',J));
            for j = 1:n
                dispData = cat(2,dispData,sprintf('w_%d = %.2e\t',j,w(j)));
            end
            dispData = cat(2,dispData,sprintf('b = %.2e',b));
            disp(dispData)
        end
    end

    % Check for the stopping condition
    if iter == maxIter-1 || J <= condStop
        stopFlag = true;
        disp("==================================")
        if iter == maxIter - 1
            disp(['BGD Stopped: max number of iterations ',sprintf('(%d)',iter)])
        else
            disp(['BGD Stopped: minimized cost function ',sprintf('(%.2f)',J)])
        end
    end

    iter = iter + 1; % increase the iteration number
end

end