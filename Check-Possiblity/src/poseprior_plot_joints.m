Si_inv = Si.';
x = Si_inv(:, 1);
y = Si_inv(:, 2);
z = Si_inv(:, 3);

figure;
scatter3(x, y, z);
% Label
a = (1:17).'; b = num2str(a); c = cellstr(b);
dx = 0.1; dy = 0.1; dz = 0.1; % displacement so the text does not overlay the data points
text(x+dx, y+dy, z+dz, c);
hold on
% Plot the edges between the points that are written in "edges" object
for i=1:length(edges)
    line = [Si_inv(edges(i,1),:); Si_inv(edges(i,2),:);];
    plot3(line(:, 1), line(:, 2), line(:, 3));
end
hold off



axis equal;
axis vis3d;
h = rotate3d;
h.RotateStyle = 'orbit';
h.Enable = 'on';