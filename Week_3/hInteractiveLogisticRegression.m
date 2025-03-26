function [cost_vec, w_vec, b_vec] = hInteractiveLogisticRegression(X_train, y_train, w_init, b_init, opts)
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
%       opts.Plot (logical)     : Plot the training progress
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
    opts.Plot = true
end
alpha = opts.LearnRate;

% Step 0: Initialization
m = size(X_train, 1); % number of samples
n = size(X_train, 2); % number of features

% Step 1: Parameter Initialization
w = w_init;
b = b_init;
stopTraining = false;
iter = 0;
w_vec = [];
b_vec = [];
cost_vec = [];

% Step 2: Loop until the termination condition has been satisfied
while ~stopTraining
    cost = 0;
    dJdw = 0;
    dJdb = 0;

    % ---------------------------------------------------------------------
    % Step 2.1: Calculate the partial derivatives: dJ/dw and dJ/db
    % ---------------------------------------------------------------------
    for i = 1:m
        X_norm_i = X_train(i,:); % no input normalization
        y_i = y_train(i);
        f_wb_i = sigmoidFunc( (X_norm_i*w + b) );
        error_i = f_wb_i - y_i;
        
        % Step 2.2: Calculate the cumulative gradients
        dJdw = dJdw + (1/m)*error_i.*X_norm_i;
        dJdb = dJdb + (1/m)*error_i;
    end

    % ---------------------------------------------------------------------
    % Step 2.2: Update the parameters: w_vec and b using the derivatives
    % ---------------------------------------------------------------------
    w = w - alpha*dJdw.';
    b = b - alpha*dJdb;

    w_vec = cat(1, w_vec, w);
    b_vec = cat(1, b_vec, b);

    % ---------------------------------------------------------------------
    % Step 2.3: Calculate the loss and cost functions based on w_vec and b
    % ---------------------------------------------------------------------
    for i = 1:m
        X_norm_i = X_train(i,:); % no input normalization
        y_i = y_train(i);
        f_wb_i = sigmoidFunc( X_norm_i*w + b );

        % Calculate the cost and loss
        loss = ( -y_i*log(f_wb_i) - (1 - y_i)*log(1 - f_wb_i) ); % binary cross-entropy loss
        cost = cost + (1/m)*loss;
    end
    cost_vec = cat(1, cost_vec, cost);

    % Step 2.4: Display the progress
    if opts.Verbose
        if mod(iter, 10) == 0
            fprintf('Iteration: %d -- w_vec: [ %s ] | b: %.4f | Cost (J): %.8f \n', iter, num2str(w.'), b, cost_vec(end));
        end
    end
    if (cost_vec(end) > 0 && cost_vec(end) <= opts.MinCost)
        stopTraining = true;
        if opts.Verbose && mod(iter, 10) ~= 0
            fprintf('Iteration: %d -- w_vec: [ %s ] | b: %.4f | Cost (J): %.8f \n', iter, num2str(w.'), b, cost_vec(end));
        end
        fprintf('----------\n');
        fprintf('Termination: Minimum cost reached.\n');
    elseif iter == opts.MaxIter - 1
        stopTraining = true;
        if opts.Verbose && mod(iter, 10) ~= 0
            fprintf('Iteration: %d -- w_vec: [ %s ] | b: %.4f | Cost (J): %.8f \n', iter, num2str(w.'), b, cost_vec(end));
        end
        fprintf('----------\n');
        fprintf('Termination: Maximum iterations reached.\n');
    end
    
    % Increment the iteration counter
    iter = iter + 1;
end

% Display the interactive training progress plots
if opts.Plot
    f = figure;
    tiledlayout(2, 2, "TileSpacing", "compact");
    
    % Plot 1: Plot data points and the fitted sigmoid function
    nexttile
    hBinaryScatter(X_train, y_train, {'o','x'},{'b','r'}, 10)
    xlabel('Tumor size')
    ylabel('y')
    hold on
    plot_x = linspace(min(X_train),max(X_train),100);
    plot(plot_x, sigmoidFunc( (w.'*plot_x + b) ), 'g')
    hold off
    legend('Benign','','','Malignant','','',['y = \sigma(f(x)), w: ' num2str(w) ', b: ' num2str(b)], "Location","southeast")
    
    % Plot 2: Cost vs iteration plot
    nexttile
    plot(0:opts.MaxIter - 1, cost_vec, 'LineWidth', 2)
    xlabel('Number of iterations')
    ylabel('Cost, J(w, b)')
    grid on
    legend(['J(end) = ' num2str(cost_vec(end))])

    % Plot 3: Contour cost vs w and b
    w_space = linspace(-1, 7, 50);
    b_space = linspace(-14, 1, 50);
    [tmp_b, tmp_w] = meshgrid(b_space, w_space);
    
    % Calculate cost matrix
    cost_mtx = zeros(size(tmp_w));
    for i = 1:size(tmp_w, 1)
        for j = 1:size(tmp_w, 2)
            cost_mtx(i, j) = computeCostMatrix(X_train, y_train, tmp_w(i,j), tmp_b(i,j));
        end
    end
    
    % Plot 3: Contour cost vs w and b
    nexttile
    contour(w_space, b_space, cost_mtx, 12)
    hold on
    for idx = 1:100:length(cost_vec)
        plot3(w_vec(idx), b_vec(idx), cost_vec(idx), 'o', 'MarkerSize', 5, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'LineWidth', 0.5)
    end
    hold off
    xlabel('w\_vec')
    ylabel('b\_vec')
    colorbar

    % Plot 4: Surf cost vs w and b
    nexttile
    surf(w_space, b_space, cost_mtx, 'FaceAlpha', 0.4)
    hold on
    for idx = 1:100:length(cost_vec)
        plot3(w_vec(idx), b_vec(idx), cost_vec(idx), 'o', 'MarkerSize', 5, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'LineWidth', 0.5)
    end
    hold off
    xlabel('w\_vec')
    ylabel('b\_vec')
    colorbar
    view([-95 37])
    colormap turbo

    f.Position = [0 0 850 600];
end
end

%% Local Functions
function cost = computeCostMatrix(X_train, y_train, w, b)
    m = size(X_train, 1); % number of samples
    cost = 0;
    for idx = 1:m
        X_norm_i = X_train(idx,:); % no input normalization
        y_i = y_train(idx);
        f_wb_i = sigmoidFunc( X_norm_i*w + b );
        loss = ( -y_i*log(f_wb_i) - (1 - y_i)*log(1 - f_wb_i) ); % binary cross-entropy loss
        cost = cost + (1/m)*loss;
    end
end

function y = sigmoidFunc(x)

    % Protect against overflow
    if any(x > 500)
        x(x > 500) = 500;
    end
    if any(x < -500)
        x(x < -500) = -500;
    end
    y = 1 ./ (1 + exp(-x));
end