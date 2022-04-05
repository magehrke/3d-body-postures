function [flags, S2] = isvalid(S)
% input S: a 3-by-17 matrix consisting of 3D coordiantes of the joints
% flags: a binary vector telling whether the corresponding bone is valid or
% not. Please see readme to know how the bones and joints are defined in
% the 3D pose S
%
% copyright: Ijaz Akhter, MPI Tuebingen
% Jan 15, 2015
% Revision (May 20, 2015)
% Output S2: The closest valid pose to S

% edges = [1 2 3 4 2 6 7 2 1 10 11 12 1 14 15 16;...
%      2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17]';
 
global prnts chlds chldsT di a % shared with global2local and local2global function

var = load('staticPose');
di = var.di;
a = var.a;

var = load('jointAngleModel_v2');
% var = load('jointAngleLimits12c_1');
jmp = var.jmp;

chlds = var.chlds;
prnts = var.prnts;
edges = var.edges;
angleSprd = var.angleSprd;
sepPlane = var.sepPlane;
E2 = var.E2;
bounds = var.bounds;

chldsT = [3 6 8 10 14];     % torso's child
nprts = length(chlds);      % excluding torso
 
dS = S(:, edges(:,1)) - S(:, edges(:,2));     % find relative depths: coordinates by taking the parent as origin

flags = true(1,size(dS,2));
angles = zeros(2,size(dS,2));
dSl = global2local(dS);     % convert relative to local coordinates

for i=1:nprts
    chldB = dSl(:, chlds(i));           % the bone to validate
    [th, phi, ri] = cart2sph(chldB(1), chldB(2), chldB(3));
    chldB = chldB/ri;
    th = rad2deg(th);
    phi = rad2deg(phi);
    t_j = floor((th+180)/jmp + 1);
    p_j = floor((phi+90)/jmp + 1);
    angles(:, chlds(i)) = [t_j; p_j];
    
    if ismember(chlds(i), chldsT)
        if ~angleSprd{i}(t_j, p_j)
            flags(chlds(i)) = false;
        end
    else
        t_p = angles(1,prnts(i));
        p_p = angles(2,prnts(i));
        
        v = squeeze(sepPlane{i}(t_p,p_p,:));
        v = v/norm(v(1:3));
        
        if any(isnan(v)) || v'*[chldB;1]>0
            flags(chlds(i)) = false;
        else
            e1 = v(1:3);
            e2 = squeeze(E2{i}(t_p,p_p,:));
            T = gramschmidt([e1,e2,cross(e1, e2)]);
            bnd = squeeze(bounds{i}(t_p,p_p,:));
            
            u = T(:,2:3)'*chldB;
            if u(1)<bnd(1) || u(1)>bnd(2) || u(2)<bnd(3) || u(2)>bnd(4)
                flags(chlds(i)) = false;
            end
        end
    end
end

%% update on Feb 27, 2015

if nargout>1
    boundries = var.boundries;
    thEdge = -180:jmp:180;
    phEdge = -90:jmp:90;
    epsilon = [-0.06 0.06 -0.06 0.06]';

    angles2 = angles;
    dSl2 = dSl;    
    for i=1:nprts
        chldB = dSl(:, chlds(i));
        ri = norm(chldB);
        chldB = chldB/ri;
        
        if ismember(chlds(i), chldsT)
            if ~flags(chlds(i))     % if angle is invalid
                angi = angles2(:, chlds(i));
                angi2 = findClosestValidPoint(boundries{i}, angi);    
                
                angles2(:, chlds(i)) = angi2;   % replace with the closest valid angle
                thetas = deg2rad(thEdge(angi2(1)));
                phis = deg2rad(phEdge(angi2(2)));
                [Xi, Yi, Zi] = sph2cart(thetas, phis, ri);
                dSl2(:, chlds(i)) = [Xi;Yi;Zi];
            end 
        else
            if ~flags(chlds(i))
                t_p = angles2(1,prnts(i));
                p_p = angles2(2,prnts(i));
                
                v = squeeze(sepPlane{i}(t_p,p_p,:));
                v = v/norm(v(1:3));
                
                if any(isnan(v))
                    disp('Invalid v!');
                end
                e1 = v(1:3);
                e2 = squeeze(E2{i}(t_p,p_p,:));
                T = gramschmidt([e1,e2,cross(e1, e2)]);
                bnd = squeeze(bounds{i}(t_p,p_p,:));
                bnd = bnd - epsilon;
                
                if v'*[chldB;1]>0
                    chldB = T(:,2:3)*T(:,2:3)'*chldB;    % project chldB to the plane
                end                
                u = T(:,2:3)'*chldB;
                
                % move u inside the bounding-box
                if u(1)<bnd(1)
                    u(1) = bnd(1);
                end
                if u(1)>bnd(2)
                    u(1) = bnd(2);
                end
                if u(2)<bnd(3)
                    u(2) = bnd(3);
                end
                if u(2)>bnd(4)
                    u(2) = bnd(4);
                end
                if norm(u)>1        % valid points only lie inside a unit circle
                    u = u/norm(u);
                end                
                
                Xh = T(:,2:3)*u;
                X2 = pointPlane2Sphere(Xh, v, ri);
                [th, phi, ~] = cart2sph(X2(1), X2(2), X2(3));                
                t_j = floor((rad2deg(th)+180)/jmp + 1);
                p_j = floor((rad2deg(phi)+90)/jmp + 1);
                
                if (i==7 || i==10) && ~angleSprd{i}(t_j, p_j)
                    % filter boundary points and find the closest one
                    
                    bndry = boundries{i};
                    thetas = deg2rad(thEdge(bndry(1,:)));
                    phis = deg2rad(phEdge(bndry(2,:)));
                    [Xs, Ys, Zs] = sph2cart(thetas, phis, 1);
                    bndryPts = [Xs; Ys; Zs];
                    
                    %----
%                     imagesc(angleSprd{i})
%                     colormap gray
%                     hold on
%                     plot(bndry(2,:), bndry(1,:), 'r-')

                    
                    % filter points on the wrong side of the sepPlane
                    ind = v'*[bndryPts; ones(1,length(Xs))]<0;
                    bndryU = T(:,2:3)'*bndryPts(:, ind);                    
                    
                    %----
%                     tmp = T(:,2:3)*bndryU;
%                     tmp = pointFromPlane2Sphere(tmp, v, 1);                    
%                     [t1, p1, ~] = cart2sph(tmp(1,:), tmp(2,:), tmp(3,:));                    
%                     t_1 = floor((radtodeg(t1)+180)/jmp + 1);
%                     p_1 = floor((radtodeg(p1)+90)/jmp + 1);
%                     plot(p_1, t_1, 'g.')
                
                    % adjust boundaries based on the bounding box
                    ind =  bndryU(1,:)<bnd(1);
                    bndryU(1,ind) = bnd(1);                    
                    
                    ind =  bndryU(1,:)>bnd(2);
                    bndryU(1,ind) = bnd(2);
                    
                    ind =  bndryU(2,:)<bnd(3);
                    bndryU(2,ind) = bnd(3);
                    
                    ind =  bndryU(2,:)>bnd(4);
                    bndryU(2,ind) = bnd(4);
                    
                    %----
%                     tmp = T(:,2:3)*bndryU;
%                     tmp = pointFromPlane2Sphere(tmp, v, 1);
%                     [t1, p1, ~] = cart2sph(tmp(1,:), tmp(2,:), tmp(3,:));                    
%                     t_1 = floor((radtodeg(t1)+180)/jmp + 1);
%                     p_1 = floor((radtodeg(p1)+90)/jmp + 1);
%                     plot(p_1, t_1, 'b.')                    
                    
                    u2 = findClosestValidPoint(bndryU, u);
%                     u2 = u2 + sign(u-u2).*epsilon2;                    
                    Xh = T(:,2:3)*u2;
                    X2 = pointPlane2Sphere(Xh, v, ri);
                    [th, phi, ~] = cart2sph(X2(1), X2(2), X2(3));
                    t_j = floor((rad2deg(th)+180)/jmp + 1);
                    p_j = floor((rad2deg(phi)+90)/jmp + 1);
                    
                    % -----
%                     plot(p_j, t_j, 'g+')
%                     hold off
                end
                
                dSl2(:, chlds(i)) = X2;
                angles2(:, chlds(i)) = [t_j; p_j];
            end
        end
    end    
    dS2 = local2global(dSl2);
    S2 = estimateZ(dS2, edges', S(:,1));
end


function pt2 = findClosestValidPoint(boundary, pt)

N = size(boundary,2);
diff = boundary - pt*ones(1,N);
dist = sum(diff.^2,1);
[~, ind] = min(dist);
pt2 = boundary(:, ind(1));

function X = pointPlane2Sphere(Xh, v, ri)
% move Xh in the direction of v until its norm become ri
% X = Xh + a*v, find a

if any(isnan(Xh))
    X = nan(3,1);
    return;
end
th = 1e-6;
vh = v(1:3);
Xv = Xh'*vh;

dis = sqrt(1+ Xv^2 - norm(Xh));
if norm(dis)<th     % dis in this case can be complex
    dis = 0;
end
a1 = -Xv + dis;
a2 = -Xv - dis;
X1 = Xh + a1*vh;
X2 = Xh + a2*vh;

if [X1',1]*v<=th
    X = ri*X1;
elseif [X2',1]*v<=th
    X = ri*X2;
else
    disp('Unable to find the point');
end