function hBinaryScatter3D(input, labels, markerColorList, markerSize)

arguments
% The input is an m-by-n dataset
% m: number of examples
% n: number of features
input
labels
markerColorList
markerSize
end

m = height(input);
n = width(input);

if n == 1 % 1D
    for j=1:length(input)
        if labels(j) % positive class
            marker = 'x';
            markerColor = 'r';
        else % negative class
            marker = 'o';
            markerColor = 'b';
        end
        plot(input(j), labels(j), "Marker", marker, "MarkerEdgeColor", markerColor, "MarkerSize", markerSize,...
            "LineWidth", 1.5, "LineStyle", "none")
        hold on
    end
    hold off

elseif n == 2 % 2D
    for i = 1:m
        plot3(input(i,1), input(i,2), labels(i), "Marker", 'o', "MarkerEdgeColor", "none", ...
            "MarkerFaceColor", markerColorList{labels(i)+1},...
            "MarkerSize", (labels(i)*markerSize) + (1-labels(i))*markerSize, ...
            "LineWidth",1.5, "LineStyle","none")
        hold on
    end
    hold off
else
    error("Undefined number of classes.")
end

end