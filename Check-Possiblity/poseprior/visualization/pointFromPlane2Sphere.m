function X = pointFromPlane2Sphere(Xh, v, ri)
% move Xh in the direction of v until its norm become ri
% X = Xh + a*v, find a. The function allows both vector and scalar input

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