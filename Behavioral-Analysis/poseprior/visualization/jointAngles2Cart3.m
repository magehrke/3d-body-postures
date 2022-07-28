function [dS2, R2] = jointAngles2Cart3(dS, angles, typ, p)
% function dS2 = jointAngles2Cart(dS, angles, normals, angleSprd)
% dS is used to get the torso points and estimate the bone lengths

if nargin<4
    p=nan;
end

load staticPose      % load di, a, Si, edges
% global angleSprd
prnts = [2 3 5 6 2 9 10 11 13 14 15];
chlds = [3 4 6 7 8 10 11 12 14 15 16];
nprts = length(chlds);      % excluding torso

% jmp = 2;
% thEdge = -180:jmp:180;
% phEdge = -90:jmp:90;
torso = [1 2 5 9 13];
chldsT = [3 6 8 10 14];     % torso's child

bl = sqrt(sum((dS).^2, 1));
dS2 = nan(size(dS));
dS2(:, torso) = dS(:, torso);

shldr = dS2(:, 5) - dS2(:, 2);
hip = dS2(:, 13) - dS2(:, 9);

for i=1:nprts
%     u = dS2(:, prnts(i));
    if ismember(chlds(i), chldsT)
        if i==1 || i==3 || i==5
            u = shldr;
        else
            u = hip;
        end
        u = u/norm(u);
        v = dS2(:,1);
        v = v/norm(v);
    else
        u = dS2(:, prnts(i));
        u = u/norm(u);
        v = getNormal(R*a, u);    % v is perpendicular to u
%         v = getNormal(R*a1, R*a2, u);    % v is perpendicular to u
%         v = getNormal(R*di(:, prnts(i)), R*a, u);    % v is perpendicular to u
    end    
%     [R] = gramschmidt([u, v, cross(u,v)]);    
%     [R] = gramschmidt([u, cross(u,v), -v]);
    w = cross(u,v);
    w = w/norm(w);
    if typ==1
        R = gramschmidt([u, v, w]);
    elseif typ==2
        R = gramschmidt([w, u, v]);
    elseif typ==3
        R = gramschmidt([v, w, u]);
    elseif typ==4
        R = gramschmidt([u, -w, v]);
    elseif typ==5
        R = gramschmidt([-w, v, u]);
    else
        R = gramschmidt([v, u, -w]);
    end
    
    if i==p
        R2 = R;
    end
    th = angles(1,i);
    phi = angles(2,i);
    
    [x,y,z] = sph2cart(deg2rad(th), deg2rad(phi), bl(chlds(i)));
    dS2(:, chlds(i)) = R*[x;y;z];
end
end

function [n] = getNormal(a, x)
% function [n] = getNormal(a1, a2, x)
% 
% cosTh = a2'*x/(norm(a2)*norm(x));
% a = cosTh^2*a1 + (1-cosTh^2)*a2;
n = cross(a, x);
end
% function n = getNormal(x1, a, x)
% 
% nth = 1e-4;
% if norm(x-a)<nth || norm(x+a)<nth   % x and a are parallel
%     n = cross(x, x1);
% else
%     n = cross(a, x);
% end
% n = n/norm(n);
% end