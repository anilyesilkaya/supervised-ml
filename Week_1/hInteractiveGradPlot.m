function hInteractiveGradPlot(w_selected)
% This function plots the cost function (J) against user-defined values of
% the weight parameter (w_selected). The plot is dynamically updated to
% visualize the results in real-time using the drawnow function.

% Let's plot the cost function against the weights w
w_vec = -5:0.1:5; % weights
x = 0:5; % features
y = x; % targets
J = [];
dJdw = [];

for j = 1:numel(w_vec)
    w = w_vec(j);
    
    f = w.*x;
    
    % Calculate the cost function
    J = cat(1,J,(1/(2*length(x)))*sum( (f-y).^2 ));
    
    % Calculate the gradient w.r.t w
    dJdw = cat(1,dJdw,(1/(length(x)))*sum( (f-y).*x ));
end

% Visualize
tolarence = 1e-12; % tolerance value

figure
tiledlayout(1,2)

nexttile
plot(w_vec,J,'k.-','LineWidth',0.7)
xlabel('$w$','Interpreter','latex');
ylabel('$J(w,b)$','Interpreter','latex');
hold on
plot(w_selected,J(abs(w_vec-w_selected)<tolarence),'o','MarkerFaceColor','r','MarkerEdgeColor','b','MarkerSize',8,'LineWidth',1.5)
hold off
legend('',['x: ',num2str(w_selected),', y: ',num2str(J(abs(w_vec-w_selected)<tolarence))],'Location','north')

nexttile
plot(w_vec,dJdw,'k','LineWidth',1.2)
xlabel('$w$','Interpreter','latex');
ylabel('$\frac{\partial}{\partial w}J(w,b)$','Interpreter','latex');
hold on
plot(w_selected,dJdw(abs(w_vec-w_selected)<tolarence),'o','MarkerFaceColor','r','MarkerEdgeColor','b','MarkerSize',8,'LineWidth',1.5)
plot(w_vec,zeros(length(w_vec),1),'k:','LineWidth',1.2)
hold off
legend('',['x: ',num2str(w_selected),', y: ',num2str(dJdw(abs(w_vec-w_selected)<tolarence))],'Location','northwest')

if dJdw(abs(w_vec-w_selected)<tolarence) < 0
    x = [0 0.08] + 0.7;
    y = [0 0] + 0.3;
    annotation('textarrow',x,y,'Color','red')
elseif dJdw(abs(w_vec-w_selected)<tolarence) == 0
    x = [0 0] + 0.75;
    y = [0 0.08] + 0.3;
    annotation('textarrow',x,y,'Color','red')
else % dJdw(abs(w_vec-w_selected)<tolarence) > 0
    x = [0.08 0] + 0.7;
    y = [0 0] + 0.3;
    annotation('textarrow',x,y,'Color','red')
end
drawnow

end