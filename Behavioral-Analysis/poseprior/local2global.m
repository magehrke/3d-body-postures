function dS = local2global(dSl)
% convert local coordinates to  relative coordinates (with origin at the parent)
% dSl is of the same size as dS and contain the same torso coordinates (left/right shoulder and hip and belly)

global prnts chlds chldsT di a %a1 a2

nprts = length(chlds);      % excluding torso
shldr = dSl(:, 5) - dSl(:, 2);
hip = dSl(:, 13) - dSl(:, 9);

torso = [1 2 5 9 13];
dS = nan(size(dSl));
dS(:, torso) = dSl(:, torso);   % dS has the same torso corordnates as dSl

for i=1:nprts
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
        [v] = getNormal(R*di(:, prnts(i)), R*a, u);       % v is perpendicular to u
    end
    
    w = cross(u,v);
    w = w/norm(w);
    R = gramschmidt([u, v, w]);
%     if typ==1
%         R = gramschmidt([u, v, w]);
%     elseif typ==2
%         R = gramschmidt([w, u, v]);
%     elseif typ==3
%         R = gramschmidt([v, w, u]);
%     elseif typ==4
%         R = gramschmidt([u, -w, v]);
%     elseif typ==5
%         R = gramschmidt([-w, v, u]);
%     else
%         R = gramschmidt([v, u, -w]);
%     end
    dS(:, chlds(i)) = R*dSl(:, chlds(i));
end
