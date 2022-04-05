function [angles, flg, fi] = cart2JointAngles3(dS, typ)

load staticPose         % load di, a, Si, edges
flg = false;
fi = 0;

prnts = [2 3 5 6 2 9 10 11 13 14 15];
chlds = [3 4 6 7 8 10 11 12 14 15 16];
chldsT = [3 6 8 10 14];     % torso's child

nprts = length(chlds);      % excluding torso
angles = zeros(2, nprts);

shldr = dS(:, 5) - dS(:, 2);
hip = dS(:, 13) - dS(:, 9);

for i=1:nprts
%     u = dS(:, prnts(i));    
    if ismember(chlds(i), chldsT)
        if i==1 || i==3 || i==5
            u = shldr;
        else
            u = hip;
        end
        u = u/norm(u);
        v = dS(:,1);
        v = v/norm(v);
    else
        u = dS(:, prnts(i));
        u = u/norm(u);
%         [v] = getNormal(R*a, u);       % v is perpendicular to u
%         [v] = getNormal(R*a1, R*a2, u);       % v is perpendicular to u
        [v,flg] = getNormal(R*di(:, prnts(i)), R*a, u);       % v is perpendicular to u
        
        if flg
           fi=i; 
        end
        ang = acosd((u'*v)/norm(u));
        if ang<87 || ang>93
            disp('Normal is not Perpedicular');
        end
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
    
    chldB = R'*dS(:, chlds(i));
    [th, phi, r] = cart2sph(chldB(1), chldB(2), chldB(3));
    
    th = radtodeg(th);
    phi = radtodeg(phi);
    angles(:, i) = [th; phi];
end
end