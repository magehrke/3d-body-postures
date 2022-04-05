function AS = getSmoothAngleSpread(p, t_c, p_c, winSize)

    global angleSprd thEdge phEdge
    
    nth = 0.15;
    a = [0.9975; 0.0023; 0.0709];
    if nargin<4
        winSize = 3;
    end
    N = floor(winSize/2);
    Nt = length(thEdge);
    Np = length(phEdge);
    
    tmin = t_c - N;
    tmax = t_c + N;    
    tI = tmin:tmax;
%     tI = tI(tI>=1 & tI<=Nt);
    tI(tI<1) = tI(tI<1) + Nt;
    tI(tI>Nt) = tI(tI>Nt) - Nt;
        
    
    pmin = p_c - N;
    pmax = p_c + N;
    pI = pmin:pmax;
%     pI = pI(pI>=1 & pI<=Np);
    pI(pI<1) = pI(pI<1) + Np;
    pI(pI>Np) = pI(pI>Np) - Np;
    
    AS = squeeze(angleSprd{p}(t_c, p_c, :,:));
    for i=1:length(tI)
        for j=1:length(pI)
            t_i = tI(i);
            p_j = pI(j);
%             b =  sph2cart(deg2rad(thEdge(t_i)), deg2rad(thEdge(p_j)), 1);
            
%             if norm(b-a)>nth && norm(b+a)>nth   % x and a are not parallel            
                AS = AS | squeeze(angleSprd{p}(t_i, p_j, :,:));
%             else
%                 i;
%             end
        end
    end        
end