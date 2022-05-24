function poseprior_plot_joints(selector, pose, name, static_pose)
    arguments
        selector string = 'none';
        pose {isnumeric} = nan;
        name string = '';
        static_pose {boolean} = false;
    end
    staticPose = load("staticPose.mat");
    if strcmp(selector, '3d')
        plot3dpose(pose, staticPose.edges, name);
        if static_pose
            plot3dpose(staticPose.Si.', staticPose.edges, 'Static Pose');
        end
    elseif strcmp(selector, '2dFrom3d')
        plot2dposeFrom3d(pose, staticPose.edges, name);
        if static_pose
            plot2dposeFrom3d(staticPose.Si.', staticPose.edges, 'Static Pose');
        end
    end
end

function plot3dpose(pose, edges, name)
    x = pose(:, 1);
    y = pose(:, 2);
    z = pose(:, 3);
    
    figure('Name', name);
    scatter3(x, y, z);
    xlabel('x-axis')
    ylabel('y-axis')
    zlabel('z-axis')
    % Label
    a = (1:17).'; b = num2str(a); c = cellstr(b);
    dx = 0.1; dy = 0.1; dz = 0.1; % displacement so the text does not overlay the data points
    text(x+dx, y+dy, z+dz, c);
    hold on
    % Plot the edges between the points that are written in "edges" object
    for i=1:length(edges)
        line = [pose(edges(i,1),:); pose(edges(i,2),:);];
        plot3(line(:, 1), line(:, 2), line(:, 3));
    end
    hold off

    axis equal;
    axis vis3d;
    h = rotate3d;
    h.RotateStyle = 'orbit';
    h.Enable = 'on';
end

function plot2dposeFrom3d(pose, edges, name)
    y = pose(:, 2);
    z = pose(:, 3);

    figure('Name', name);
    scatter(z, y);
    xlabel('x-axis')
    ylabel('y-axis')
    % Label
    a = (1:17).'; b = num2str(a); c = cellstr(b);
    dy = 0.1; dz = 0.1; % displacement so the text does not overlay the data points
    text(z+dz, y+dy, c);
    hold on
    % Plot the edges between the points that are written in "edges" object
    for i=1:length(edges)
        line = [pose(edges(i,1),:); pose(edges(i,2),:);];
        plot(line(:, 3), line(:, 2));
    end
    hold off
    
    axis equal;
    axis vis3d;
    h = rotate3d;
    h.RotateStyle = 'orbit';
    h.Enable = 'on';
end