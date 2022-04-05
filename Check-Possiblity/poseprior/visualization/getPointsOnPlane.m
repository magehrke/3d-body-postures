function [points, tri] = getPointsOnPlane(X, v, e2)
% function [points, tri] = getPointsOnPlane(X, v)

v = v/norm(v(1:3));
% [U,~,~] = svd(v(1:3)*ones(1,3));
% B = U(:, 2:3);
% T = [B, v(1:3)];
% x = B'*X;
%
% bb = minBoundingBox(x);
% points = T*[bb; -v(4)*ones(1,4)];
% tri = [1, 2, 3; 1, 3, 4; 1, 4, 2; 2, 4, 3];

e1 = v(1:3);
T = gramschmidt([e1,e2,cross(e1, e2)]);
x = T(:,2:3)'*X;

bb = minBoundingBox(x);
points = T*[-v(4)*ones(1,4); bb];
tri = [1, 2, 3; 1, 3, 4; 1, 4, 2; 2, 4, 3];

%%
% points = zeros(3,4);
% 
% A = v(3)^2 + v(1)^2;
% B = 2*v(3)*v(4);
% C = v(4)^2- v(1)^2;
% 
% tm = sqrt(B^2 - 4*A*C);
% points(3,1) = (-B+tm)/(2*A);
% points(3,2) = (-B-tm)/(2*A);
% 
% points(1,1) = -(v(4)+v(3)*points(3,1))/v(1);
% points(1,2) = -(v(4)+v(3)*points(3,2))/v(1);
% 
% A = v(2)^2 + v(1)^2;
% B = 2*v(2)*v(4);
% C = v(4)^2- v(1)^2;
% 
% tm = sqrt(B^2 - 4*A*C);
% points(2,3) = (-B+tm)/(2*A);
% points(2,4) = (-B-tm)/(2*A);
% points(1,3) = -(v(4)+v(2)*points(2,3))/v(1);
% points(1,4) = -(v(4)+v(2)*points(2,4))/v(1);
% 
% tri = [1,2,3; 2,3,4];

%%
% n = 1;
% pt = zeros(3,4);
% points = zeros(3,4);
% 
% jMin = -1;
% jMax = 1;
% kMin = -1;
% kMax = 1;
% 
% for i=1:3
%     j = rem(i,3)+1;
%     k = rem(i+1,3)+1;
% %     jMin = min(X(j,:));
% %     kMin = min(X(k,:));
% %     jMax = max(X(j,:));
% %     kMax = max(X(k,:));
%     
%     pt(i,:) = 0;
%     pt(j,:) = [jMin,jMin, jMax,jMax];
%     pt(k,:) = [kMin,kMax, kMin,kMax];
%     
%     for q=1:4        
%         pt(i,q) = -(v(4) + v(j)*pt(j,q) + v(k)*pt(k,q))/v(i);
%         if pt(i,q)<=(max(X(i,:))+0.3) && pt(i,q)>=(min(X(i,:))-0.3)
%             points(:,n) = pt(:,q);
%             n = n + 1;
%         end
%     end
% end
% n = n - 1;
% 
% if n<3
%     points=pt;
%     tri = [1,2,3; 2,3,4];
% elseif n==3
%     tri = [1,2,3];
% % elseif n==4
% %     tri = [1,2,3; 2,3,4];
% else
%     pt2 = points + 0.1*rand(size(points));
%     tri = convhull(pt2(1,:), pt2(2,:), pt2(3,:));
% end
