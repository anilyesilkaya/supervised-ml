function hInteractiveLinearRegression(x,y,bgd_w,bgd_b,bgd_J,idxIter,plotMode)
% This function plots the linear regression fitting for a given iteration
% step. Accordingly, each individual step of the SGD can be presented if plotMode is
% set to "STEP". Similarly, the entire process with each step highlighted can be
% plotted if plotMode is "CUMULATIVE". The function generates four figures:
%
% Figure 1: Final regression plot using the momentary or final weight (w)
% and bias (b) values.
%
% Figure 2: Cost function (J) versus iteration step.
%
% Figure 3: Cost function (J) surface versus the weight (w) and bias
% value (b) in three dimensions.
%
% Figure 4: Top view of the cost function (J) contour versus the weight (w) and
% bias (b), showing each iteration step.

idxIter = idxIter + 1;
switch upper(plotMode)
    case "STEP"
        linearRegressionStep(x,y,bgd_w,bgd_b,bgd_J,idxIter);
    case "CUMULATIVE"
        linearRegressionCumulative(x,y,bgd_w,bgd_b,bgd_J,idxIter);
    otherwise
        error("Undefined plotting mode.")
end

end

%% Linear regression visualization for each iteration step
function linearRegressionStep(x,y,bgd_w,bgd_b,bgd_J,idxIter)

weight_vec = -10:0.9:10;
bias_vec = -10:0.9:10;
J_mtx = costFunctionSurf(x,y,weight_vec,bias_vec);

% Figure 1: Regression plot for each step of gradient descent
figure
tiledlayout(2,2)
nexttile
scatter(x,y,'ro','filled');
grid on
xlabel('x')
ylabel('y')
hold on
plot(x,bgd_w(idxIter)*x+bgd_b(idxIter),'b-')
hold off
if bgd_b(idxIter) > 0
    legend('data points',[num2str(bgd_w(idxIter)),'$x+$',num2str(bgd_b(idxIter))],"Interpreter","latex","Location","northwest");
else
    legend('data points',[num2str(bgd_w(idxIter)),'$x-$',num2str(abs(bgd_b(idxIter)))],"Interpreter","latex","Location","northwest");
end

% Figure 2: Cost function plot against number of iterations
nexttile
plot(0:idxIter-1,bgd_J(1:idxIter),'b-','Marker','.','LineWidth',3)
xlabel('Number of iterations')
ylabel('Cost Function (J)')

% Figure 3: 3D cost function plot against w and b values
nexttile
surf(bias_vec,weight_vec,J_mtx,'FaceAlpha',0.5)
hold on
plot(bgd_b(idxIter),bgd_w(idxIter),'ro','MarkerFaceColor','red','MarkerSize',7)
hold off
xlabel('b')
ylabel('w')
zlabel('J(w,b)')
title('J(w,b) versus w and b')
view([404 31])

% Figure 4: Top view contour cost function plot against w and b
nexttile
contour(bias_vec,weight_vec,J_mtx,35);
hold on
plot(bgd_b(idxIter),bgd_w(idxIter),'ro','MarkerFaceColor','r','MarkerSize',6,'MarkerEdgeColor','k')
hold off
xlabel('b')
ylabel('w')
colorbar

end

%% Linear regression visualization for cumulative time steps
function linearRegressionCumulative(x,y,bgd_w,bgd_b,bgd_J,idxIter)

weight_vec = -10:0.9:10;
bias_vec = -10:0.9:10;
J_mtx = costFunctionSurf(x,y,weight_vec,bias_vec);

% Figure 1: Regression plot for cumulative gradient descent
figure
tiledlayout(2,2)
nexttile
scatter(x,y,'ro','filled');
grid on
xlabel('x')
ylabel('y')
hold on
plot(x,bgd_w(idxIter)*x+bgd_b(idxIter),'b-')
hold off
if bgd_b(idxIter) > 0
    legend('data points',[num2str(bgd_w(idxIter)),'$x+$',num2str(bgd_b(idxIter))],"Interpreter","latex","Location","northwest");
else
    legend('data points',[num2str(bgd_w(idxIter)),'$x-$',num2str(abs(bgd_b(idxIter)))],"Interpreter","latex","Location","northwest");
end

weight_vec = -10:0.9:10;
bias_vec = -10:0.9:10;

for idxw = 1:length(weight_vec)
    for idxb = 1:length(bias_vec)
        w = weight_vec(idxw);
        b = bias_vec(idxb);

        J(idxw,idxb) = sum((w*x+b-y).^2);
    end
end

% Figure 2: Cumulative cost function plot
nexttile
plot(0:idxIter-1,bgd_J(1:idxIter),'b-','Marker','.','LineWidth',3)
xlabel('Number of iterations')
ylabel('Cost Function (J)')

% Figure 3: 3D cumulative cost function plot against w and b values
nexttile
surf(bias_vec,weight_vec,J,'FaceAlpha',0.5)
hold on
plot(bgd_b(1:idxIter),bgd_w(1:idxIter),'ro','MarkerFaceColor','red','MarkerSize',7)
hold off
xlabel('b')
ylabel('w')
zlabel('J(w,b)')
title('J(w,b) versus w and b')
view([404 31])

% Figure 4: Top view cumulative contour cost function plot against w and b
nexttile
contour(bias_vec,weight_vec,J_mtx,35);
hold on
plot(bgd_b(1:idxIter),bgd_w(1:idxIter),'ro','MarkerFaceColor','r','MarkerSize',6,'MarkerEdgeColor','k')
hold off
xlabel('b')
ylabel('w')
colorbar
drawnow

end


%% Local function definition

function J_mtx = costFunctionSurf(x,y,weight_vec,bias_vec)
m = length(x);
J_mtx = [];

for idxw = 1:length(weight_vec)
    for idxb = 1:length(bias_vec)
        w = weight_vec(idxw);
        b = bias_vec(idxb);

        J = 0;
        for i = 1:m
            J = J + (w*x(i)+b-y(i)).^2;
        end
        J_mtx(idxw,idxb) = (1/(2*m))*J;
    end
end

end