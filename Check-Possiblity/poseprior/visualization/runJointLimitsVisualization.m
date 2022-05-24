function runJointLimitsVisualization
%% Use system background color for GUI components   
global angleSprd thEdge phEdge

panelColor = 'w';
typ = 1;
var = load('jointAngleLimitsVisual');

X1 = var.X1;
triP = 1;
jmp = var.jmp;
% jmp = 2;
edges = var.edges;
thEdge = var.thEdge;
phEdge = var.phEdge;
Nt = length(thEdge);
Np = length(phEdge);

chlds = var.chlds;
prnts = var.prnts;
angleSprd = var.angleSprd;
ASLwrLegs = var.ASLwrLegs;
sepPlane = var.sepPlane;
E2 = var.E2;
bounds = var.bounds;
% winSize = var.winSize;
winSize = 3;
sel = [1,3; 2,3; 1,4; 2,4]';
prnts2 = [0 1 0 3 0 0 6 7 0 9 10];
% chldsT = [3 6 8 10 14];
% chldsR = [4 7 11 12 15 16];
prtName = {'L-Uarm', 'L-Larm', 'R-Uarm', 'R-Larm', 'head', ...
    'L-Uleg', 'L-Lleg', 'L-feet', 'R-Uleg', 'R-Lleg', 'R-feet'};
hs = zeros(4,1);
tl = 1;
ax = [-15 15 -15 15 -20 20];
ax2 = [-1.1 1.1 -1.1 1.1 -1.1 1.1];
% jmp = 2;
shdStop = true;
vidStop = true;
vCnt = 1;
writerObj = 0;
P = length(X1);
dS = X1(:, edges(:,1)) - X1(:, edges(:,2));

rdC = [51,160,44]'/255;
cp = [251,154,153]'/255;
grC = [166,206,227]'/255;
lnC = 0.4*[1 1 1];

%% ------------ Callback Functions ---------------

% Figure resize function
function figResize(src, evt)
    set(src, 'Units','characters');
    fpos = get(src,'Position');
    
%     [100 8 20 27]
%     [1/20 8 100 27]
    set(src,'Position',[fpos(3)*100/120 fpos(4)*5/35 fpos(3)*20/120 fpos(4)*30/35])
    set(src,'Position',[1/20 fpos(4)*5/35 fpos(3)*100/120 fpos(4)*30/35]);    
end

% Right panel resize function
function rightPanelResize(src, evt)
%     [3 2 13 24]
    rpos = get(src,'Position');
    set(src,'Position',[rpos(3)*3/20 rpos(4)*2/30 rpos(3)*13/20 rpos(4)*24/30]);
end

%% Callback for list box
function listBoxCallback(src, evt)
p = get(src, 'Value');
if prnts2(p)==sp
    t_gp = t_p;
    p_gp = p_p;
    [myPrnt, myGPrnt, t_p, p_p] = getPrntsNAnglIndcs(p, angles);
else  
    [myPrnt, myGPrnt, t_p, p_p, t_gp, p_gp] = getPrntsNAnglIndcs(p, angles);
end
sp = p;
% t_c = ceil(length(thEdge)/2)+1;
% p_c = ceil(length(phEdge)/2)+1;
t_c = 27;
p_c = 32;
% [myPrnt, myGPrnt, t_p, p_p, t_gp, p_gp] = getPrntsNAnglIndcs(sp, angles);
initlialize(sp);
end % listBoxCallback

%% ------------ GUI layout ---------------

%% Set up the figure and defaults
f = figure('Units','characters', 'Position',[10 10 237 72], 'NumberTitle','off', ...
        'Name','Joint-Angle-Limits', 'ResizeFcn',@figResize, 'KeyPressFcn', @keyPressed,'Color', 'w'); 
% set(f, 'WindowButtonDownFcn', @clicker);
set(f, 'WindowButtonUpFcn', @stopDragFcn, 'WindowButtonDownFcn', @startDragFcn);
%% Create the right side panel
rightPanel = uipanel('bordertype','etchedin','BackgroundColor',...
    panelColor, 'Units','characters', 'Position',[100 8 20 27], ...
    'Parent',f, 'ResizeFcn',@rightPanelResize);

%% Create the center panel
% bg = 0.96*[1 1 1];
centerPanel = uipanel('bordertype','etchedin','Units','characters',...
    'Position', [1/20 8 100 27], 'BackgroundColor', panelColor, 'Parent',f);
    
listBox = uicontrol(f,'Style','listbox','Units','characters',...
        'Position',[3 2 13 24], 'BackgroundColor','white','Max', ...
        10,'Min',1, 'Parent',rightPanel,'Callback',@listBoxCallback);

set(listBox,'String',prtName)
%% Add an axes to the center panel
axes('parent',centerPanel);

%% 
function startDragFcn(varargin)
    set(f, 'WindowButtonMotionFcn', @clicker);
end

function stopDragFcn(varargin)
%     pause(0.01);
    set(f, 'WindowButtonMotionFcn', '');
end

%% click function
function clicker(gcbo, eventData, handles)
%     disp(get(gca, 'CurrentPoint'));
    coordinates = get(gca, 'CurrentPoint');
    c = round(coordinates(1,1));
    r = round(coordinates(1,2));
    
    if gca==hs(1) && r>0 && c>0 && r<Np && c<Nt
        t_p = c;
        p_p = r;        
        resetAxes(sp);
    elseif gca==hs(4) && r>0 && c>0
        if myPrnt && r<Nv && c<Nu
            u_c = c;
            v_c = r;
            [t_c, p_c] = uv2polar(sp, u_c, v_c, t_p, p_p);
        elseif r<Np && c<Nt
            t_c = c;
            p_c = r;
        end
        ch = get(hs(6), 'Children');
%         set(ch(2), 'XData', [], 'YData', [], 'ZData', []);
        reset4thAxis(sp);
    end    
end

function reset4thAxis(p)
    ch = get(hs(4), 'Children');
    if myPrnt
        set(ch(length(ch)-1),  'XData', u_c, 'YData', v_c);
    else
        set(ch(length(ch)-1),  'XData', t_c, 'YData', p_c);
    end
    Sk = getLimitPose(p, t_c, p_c, t_p, p_p);
    
    ch = get(hs(5), 'Children'); 
    pn = chlds(sp) + 1;
    rn = setdiff(1:P, pn);
    
    n = length(ch);
    m = n;
    set(ch(m), 'XData', Sk(1, rn), 'YData', Sk(3, rn), 'ZData', Sk(2, rn));
    
    for b = 1:size(edges,1)
        m = m-1;
        i2 = edges(b,1); j2 = edges(b,2);
        set(ch(m), 'XData', [Sk(1,i2), Sk(1,j2)], 'YData', [Sk(3,i2), Sk(3,j2)], ...
            'ZData', [Sk(2,i2), Sk(2,j2)])
    end
    set(ch(m-1), 'XData', Sk(1, pn), 'YData', Sk(3, pn), 'ZData', Sk(2, pn));
    
    ch = get(hs(6), 'Children'); 
    [x,y,z] = sph2cart(deg2rad(thEdge(t_c)), deg2rad(phEdge(p_c)), 1);
    set(ch(1), 'XData', x, 'YData', z, 'ZData', y);

end

function resetAxes(p)
    if myPrnt==0
        return;
    end    
    ch = get(hs(1), 'Children');
    set(ch(1), 'XData', t_p, 'YData', p_p);
    
    [Sl] = getLimitPoses(p, t_p, p_p, t_gp, p_gp);    
    ch = get(hs(4), 'Children');     

    AS = getSmoothAngleSpread(p, t_p, p_p, winSize);
    [S2, s2, points, tr] = getSpericalSpread(sp, t_c, p_c,  t_p, p_p, AS);        
        
    if myPrnt
        [sprd, ind] = getProjectedSpread(sp, t_p, p_p, S2);
%         set(ch(length(ch)), 'CData', sprd);

        imgAS = zeros(size(sprd,2), size(sprd,1), 3);
        lyr = zeros(size(sprd));
        for i=1:3
            lyr(~sprd) = grC(i);
            lyr(sprd) = rdC(i);
            lyr(ind) = 0;
            imgAS(:,:,i) = lyr;
        end
        set(ch(length(ch)), 'CData', imgAS);
    else
%         set(ch(length(ch)), 'CData', AS');
        imgAS = zeros(size(AS,2), size(AS,1), 3);
        lyr = zeros(size(AS));
        for i=1:3
            lyr(~AS) = grC(i);
            lyr(AS) = rdC(i);
            imgAS(:,:,i) = lyr';
        end
        set(ch(length(ch)), 'CData', imgAS);
    end
    
    ch = get(hs(2), 'Children'); 
    n = length(ch);
    
    m = n;
    for k=1:4
        Sk = Sl(3*k-2:3*k, :);        
        set(ch(m), 'XData', Sk(1, :), 'YData', Sk(3, :), 'ZData', Sk(2, :));
        
        for b = 1:size(edges,1)
            m = m-1;
            ii = edges(b,1); jj = edges(b,2);
            set(ch(m), 'XData', [Sk(1,ii), Sk(1,jj)], 'YData', [Sk(3,ii), Sk(3,jj)], ...
            'ZData', [Sk(2,ii), Sk(2, jj)]);
        end
        i2 = prnts(p) + 1;
        j2 = chlds(p) + 1;
        if i2==3 && j2==9
            i2 = 2;
        end         
        set(ch(m-1), 'XData', [Sk(1,i2), Sk(1,j2)], 'YData', [Sk(3,i2), Sk(3,j2)], ...
            'ZData', [Sk(2,i2), Sk(2, j2)])
        m = m-2;
    end

%     str = sprintf('\\theta: %d, \\phi: %d', t_p, p_p);
%     set(tl, 'String', str);
    
    ch = get(hs(3), 'Children');
    [x,y,z] = sph2cart(deg2rad(thEdge(t_p)), deg2rad(phEdge(p_p)), 1);
    set(ch(1), 'XData', x, 'YData', z, 'ZData', y);
    
    ch = get(hs(5), 'Children'); 
    n = length(ch);
    m = n;
    pn = chlds(sp) + 1;
    rn = setdiff(1:P, pn);
    set(ch(m), 'XData', Sk(1, rn), 'YData', Sk(3, rn), 'ZData', Sk(2, rn));
    
    for b = 1:size(edges,1)
        m = m-1;
        i2 = edges(b,1); j2 = edges(b,2);
        set(ch(m), 'XData', [Sk(1,i2), Sk(1,j2)], 'YData', [Sk(3,i2), Sk(3,j2)], ...
            'ZData', [Sk(2,i2), Sk(2, j2)])
    end
    set(ch(m-1), 'XData', Sk(1, pn), 'YData', Sk(3, pn), 'ZData', Sk(2, pn));
    ch = get(hs(6), 'Children');    
    
    v = squeeze(sepPlane{sp}(t_p,p_p,:));
    v = v/norm(v(1:3));
    bnd = squeeze(bounds{sp}(t_p,p_p,:));
    
    e1 = v(1:3);
    e2 = squeeze(E2{sp}(t_p,p_p,:));
    T = gramschmidt([e1,e2,cross(e1, e2)]);
    u = T(:,2:3)'*spX;
    
    ind1 = (v'*[spX;ones(1,length(spX))]) < 0;
    ind2 = ~(u(1,:)<bnd(1) | u(1,:)>bnd(2) | u(2,:)<bnd(3) | u(2,:)>bnd(4));
    ind = ind1 & ind2;
    
%     tind = all(indn(tri), 2);
    tind = sum(~ind(tri), 2)>2;
    tri2 = tri(tind, :);
    tri1 = tri(~tind, :);
    
    Xp = points(1,:);
    Yp = points(2,:);
    Zp = points(3,:);
    x = spX(1,:);
    y = spX(2,:);
    z = spX(3,:);
    set(ch(1), 'XData', s2(1), 'YData', s2(3), 'ZData', s2(2));
%     set(ch(2), 'XData', S2(1, :), 'YData', S2(3, :), 'ZData', S2(2, :));
    n = 2;
    set(ch(n+1), 'vertices', [Xp;Zp;Yp]', 'XData', Xp(tr'), 'YData', Zp(tr'), 'ZData', Yp(tr'));
    set(ch(n+2), 'XData', x(tri2'), 'YData', z(tri2'), 'ZData', y(tri2'));
    set(ch(n+3), 'XData', x(tri1'), 'YData', z(tri1'), 'ZData', y(tri1'));

end        
%% key Press function
function keyPressed(src, evnt)
    
    if myPrnt && gca==hs(4) && ~strcmpi(evnt.Key, 'r')
        return;
    end
    if gca==hs(1)
        c = t_p;
        r = p_p;
    elseif gca==hs(4)
        c = t_c;
        r = p_c;
    else
        
    end
    if strcmpi(evnt.Key, 'r')
        if shdStop
            shdStop = false;
            rotateView;
%             start(t);
        else
            shdStop = true;
%             stop(t);
        end
    elseif strcmpi(evnt.Key, 'v')
        if vidStop
            vidStop = false;
            name = sprintf('VideoCapture%d.avi', vCnt);
            writerObj = VideoWriter(name);
            open(writerObj);            
%             captureScreen;
        else
            vCnt = vCnt + 1;
            vidStop = true;
            close(writerObj);
        end
    elseif strcmpi(evnt.Key, 'downarrow')
        if r>1
            r = r - 1;
        end
    elseif strcmpi(evnt.Key, 'uparrow')
        if r<Np
            r = r + 1;
        end
    elseif strcmpi(evnt.Key, 'leftarrow')
        if c>1
            c = c - 1;
        end
    elseif strcmpi(evnt.Key, 'rightarrow')
        if c<Nt
            c = c + 1;
        end
    elseif strcmpi(evnt.Key, 'a')
        
    end
    
    if strcmpi(evnt.Key, 'downarrow') || strcmpi(evnt.Key, 'uparrow') || ...
            strcmpi(evnt.Key, 'leftarrow') || strcmpi(evnt.Key, 'rightarrow')
        if gca==hs(1)
            t_p = c;
            p_p = r;
            resetAxes(sp);
%         elseif gca==hs(4)
%             t_c = c;
%             p_c = r;
%             reset4thAxis(sp);
        end
    end
end

% function captureScreen
%     while ~vidStop 
%         frame = getframe;
%         writeVideo(writerObj,frame);
%     end
%     close(writerObj);
% end

function rotateView
    delta = 2;
    while ~shdStop        
        set(hs(2), 'View', get(hs(2), 'View')-[delta,0]);
        set(hs(3), 'View', get(hs(3), 'View')-[delta,0]);
        set(hs(5), 'View', get(hs(5), 'View')-[delta,0]);
        set(hs(6), 'View', get(hs(6), 'View')-[delta,0]);
        if ~vidStop            
            frame = getframe(gcf);            
            writeVideo(writerObj,frame);
        end
       pause(0.0005);
    end
end

function myTimerFnc
    delta = 3;
    set(hs(2), 'View', get(hs(2), 'View')-[delta,0]);
    set(hs(3), 'View', get(hs(3), 'View')-[delta,0]);
    set(hs(5), 'View', get(hs(5), 'View')-[delta,0]);
    set(hs(6), 'View', get(hs(6), 'View')-[delta,0]);
end

function initlialize(sp)
    a = [-15, 250];
    b = [10, 30];    
    c = {'r.-', 'g.-', 'b.-', 'c.-'};
    fc = 1.4;
    scl = 1.005;
    
    if myPrnt==0
        hs(1) = subplot(2,3,1);
        cla;
        title('torso')
    else
        hs(1) = subplot(2,3,1);
        if myGPrnt
%             LT = squeeze(angleSprd{myPrnt}(t_gp, p_gp, :,:));
            if sp==8
                LT = ASLwrLegs{1}; 
            elseif sp==11
                LT = ASLwrLegs{2};
            end
        else
            LT = squeeze(angleSprd{myPrnt});            
        end
        imgAS = zeros(size(LT,2), size(LT,1), 3);
        lyr = zeros(size(LT));
        for i=1:3
            lyr(~LT) = grC(i);
            lyr(LT) = rdC(i);
            imgAS(:,:,i) = lyr';
        end
        imagesc(imgAS)
%         imagesc(LT');
        colormap(gray)
        xlabel('\theta', 'FontSize', 20);
        ylabel('\phi', 'FontSize', 20);
        xl = str2num(get(hs(1), 'XTickLabel'));
        yl = str2num(get(hs(1), 'YTickLabel'));
        set(hs(1), 'XTickLabel', jmp*xl-180, 'YTickLabel', jmp*yl-90);        
        title(prtName{myPrnt})
        hold on
        plot(t_p, p_p, 'k.', 'markersize', 30);
        hold off
    end
    
    [Sl] = getLimitPoses(sp, t_p, p_p, t_gp, p_gp);
    
    hs(4) = subplot(2,3,4);
    if myPrnt
        AS = getSmoothAngleSpread(sp, t_p, p_p, winSize);
    else
        AS = squeeze(angleSprd{sp});        
    end
    [S2, s2, points, triP] = getSpericalSpread(sp, t_c, p_c, t_p, p_p, AS);
    
    if myPrnt
        [sprd, ind] = getProjectedSpread(sp, t_p, p_p, S2);
%         imagesc(sprd);
        imgAS = zeros(size(sprd,2), size(sprd,1), 3);
        lyr = zeros(size(sprd));
        for i=1:3
            lyr(~sprd) = grC(i);
            lyr(sprd) = rdC(i);
            lyr(ind) = 0;
            imgAS(:,:,i) = lyr;
        end
        imagesc(imgAS);
        
        xl = str2num(get(hs(4), 'XTickLabel'));
        yl = str2num(get(hs(4), 'YTickLabel'));
        set(hs(4), 'XTickLabel', gap*(xl-100), 'YTickLabel', gap*(yl-100));
        xlabel('u', 'FontSize', 20);
        ylabel('v', 'FontSize', 20);
        
        hold on
        plot(u_c, v_c, 'r.', 'markersize', 30);        
        hold off        
    else
%         imagesc(AS');        
        imgAS = zeros(size(AS,2), size(AS,1), 3);
        lyr = zeros(size(AS));
        for i=1:3
            lyr(~AS) = grC(i);
            lyr(AS) = rdC(i);
            imgAS(:,:,i) = lyr';
        end
        imagesc(imgAS);
        
        colormap(gray)
        xlabel('\theta', 'FontSize', 20);
        ylabel('\phi', 'FontSize', 20);
        
        xl = str2num(get(hs(4), 'XTickLabel'));
        yl = str2num(get(hs(4), 'YTickLabel'));
        set(hs(4), 'XTickLabel', jmp*xl-180, 'YTickLabel', jmp*yl-90);
        
        hold on
        plot(t_c, p_c, 'r.', 'markersize', 30);
        hold off
    end
    title(prtName{sp});
    
    hs(2) = subplot(2,3,2);
    cla
    hold on;    
    for k=1:4
        Sk = Sl(3*k-2:3*k, :);        
        i2 = prnts(sp) + 1;
        j2 = chlds(sp) + 1;
        if i2==3 && j2==9
            i2 = 2;
        end
        plot3(Sk(1, :), Sk(3, :), Sk(2, :), 'b.', 'markersize', 30);
        drawLines(Sk, 'k-',edges);
        plot3([Sk(1,i2), Sk(1,j2)], [Sk(3,i2), Sk(3,j2)], [Sk(2,i2), Sk(2, j2)], c{k}, 'linewidth', 3);
    end

    hold off
    view(a(1),b(1));
    axis equal off
    axis(ax);
    zoom out;
    zoom(fc);
    axis vis3d    
    
%     str = sprintf('\\theta: %d, \\phi: %d', t_p, p_p);
%     tl = title(str);
    
    hs(3) = subplot(2,3,3);
%     [X,Y,Z] = sphere;
    if myPrnt
        
        [ind, s1] = getSpericalSpread2(LT, t_p, p_p);        
        %     tind = all(indn(tri), 2);
        tind = sum(~ind(tri), 2)>2;
        tri2 = tri(tind, :);
        tri1 = tri(~tind, :);
        
        h = trisurf(tri1, spX(1,:), spX(3,:), spX(2,:));
        set(h, 'FaceColor', rdC, 'EdgeColor','none','LineStyle','none','FaceLighting','phong');
        alpha(0.3);
        hold on
        
        h = trisurf(tri2, spX(1,:), spX(3,:), spX(2,:));
        set(h, 'FaceColor', grC, 'EdgeColor','none','LineStyle','none','FaceLighting','phong');
        alpha(0.4);        
        
        h = surf(scl*Sx,scl*Sz,scl*Sy);
        set(h, 'FaceColor', 'none', 'EdgeColor',lnC, 'LineStyle','-');
%         plot3(S2(1, :), S2(3, :), S2(2, :), 'k.', 'markersize', 5);
        plot3(s1(1), s1(3), s1(2), 'k.', 'markersize', 30);
        hold off        
        
        view(a(1),b(1));        
        axis equal off 
        axis(ax2);
        axis vis3d
    else
        cla;
    end
    
    hs(5) = subplot(2,3,5);    
    [Sl] = getLimitPose(sp, t_c, p_c, t_p, p_p);
    pn = chlds(sp) + 1;
    rn = setdiff(1:P, pn);
    plot3(Sl(1, rn), Sl(3, rn), Sl(2, rn), 'b.', 'markersize', 30);
    hold on
    drawLines(Sl, 'k-',edges);
    
    plot3(Sl(1, pn), Sl(3, pn), Sl(2, pn), 'r.', 'markersize', 35);
    hold off
    
    view(a(1),b(1));
    axis equal off 
    axis(ax);
    zoom(fc);
    axis vis3d
    
    hs(6) = subplot(2,3,6);
    if myPrnt
        v = squeeze(sepPlane{sp}(t_p,p_p,:));
        v = v/norm(v(1:3));
        bnd = squeeze(bounds{sp}(t_p,p_p,:));
        
        e1 = v(1:3);
        e2 = squeeze(E2{sp}(t_p,p_p,:));
        T = gramschmidt([e1,e2,cross(e1, e2)]);
        u = T(:,2:3)'*spX;
        
        ind1 = (v'*[spX;ones(1,length(spX))]) < 0;
        ind2 = ~(u(1,:)<bnd(1) | u(1,:)>bnd(2) | u(2,:)<bnd(3) | u(2,:)>bnd(4));
        ind = ind1 & ind2;
        
        tind = all(~ind(tri), 2);
%         tind = sum(~ind(tri), 2)>2;
        tri2 = tri(tind, :);
        tri1 = tri(~tind, :);
        
        h = trisurf(tri1, spX(1,:), spX(3,:), spX(2,:));
        set(h, 'FaceColor', rdC, 'EdgeColor','none','LineStyle','none','FaceLighting','phong');
        alpha(0.3);
        hold on
        
        h = trisurf(tri2, spX(1,:), spX(3,:), spX(2,:));
        set(h, 'FaceColor', grC, 'EdgeColor','none','LineStyle','none','FaceLighting','phong');
        alpha(0.4);
        h = trisurf(triP,points(1,:),points(3,:),points(2,:));
        set(h, 'FaceColor', cp, 'EdgeColor','none','LineStyle','none','FaceLighting','phong');
        alpha(0.4);
        
        h = surf(scl*Sx,scl*Sz,scl*Sy);
        set(h, 'FaceColor', 'none', 'EdgeColor',lnC, 'LineStyle','-');
%         plot3(S2(1, :), S2(3, :), S2(2, :), 'k.', 'markersize', 5);
        plot3(s2(1), s2(3), s2(2), 'r.', 'markersize', 30);
        hold off
        
        view(a(1),b(1));
        axis equal off
        axis(ax2);
        axis vis3d
    else
        [ind, s1] = getSpericalSpread2(AS, t_c, p_c);        
        %     tind = all(indn(tri), 2);
        tind = sum(~ind(tri), 2)>2;
        tri2 = tri(tind, :);
        tri1 = tri(~tind, :);
        
        h = trisurf(tri1, spX(1,:), spX(3,:), spX(2,:));
        set(h, 'FaceColor', rdC, 'EdgeColor','none','LineStyle','none','FaceLighting','phong');
        alpha(0.3);
        hold on
        
        h = trisurf(tri2, spX(1,:), spX(3,:), spX(2,:));
        set(h, 'FaceColor', grC, 'EdgeColor','none','LineStyle','none','FaceLighting','phong');
        alpha(0.4);        
        
        h = surf(scl*Sx,scl*Sz,scl*Sy);
        set(h, 'FaceColor', 'none', 'EdgeColor',lnC, 'LineStyle','-');
%         plot3(S2(1, :), S2(3, :), S2(2, :), 'k.', 'markersize', 5);
        plot3(s1(1), s1(3), s1(2), 'r.', 'markersize', 30);
        hold off        
        
        view(a(1),b(1));        
        axis equal off 
        axis(ax2);
        axis vis3d
    end
end
function [ind, s] = getSpericalSpread2(AS, t_i, p_i)

    [ti, pi] = find(AS>0);
    angVldSpc = [thEdge(ti)', phEdge(pi)'];
    
    [sTh, sPhi, ~] = cart2sph(spX(1,:), spX(2,:), spX(3,:));    
    sti = floor((rad2deg(sTh)+180)/jmp + 1);
    spi = floor((rad2deg(sPhi)+90)/jmp + 1);
    
    qntAngs = [thEdge(sti)', phEdge(spi)'];
    ind = ismember(qntAngs, angVldSpc, 'rows');
    
    [x,y,z] = sph2cart(deg2rad(thEdge(t_i)), deg2rad(phEdge(p_i)), 1);
    s = [x;y;z];    
end

function [S, s, points, tr] = getSpericalSpread(p, t_i, p_i, t_ii, p_ii, AS)

    points = nan(3,4);
    tr = triP;
    
    [ti, pi] = find(AS>0);
    thetas = thEdge(ti);
    phis = phEdge(pi);
    
    [x,y,z] = sph2cart(deg2rad(thetas), deg2rad(phis), 1);
    S = [x;y;z];
    [x,y,z] = sph2cart(deg2rad(thEdge(t_i)), deg2rad(phEdge(p_i)), 1);
    s = [x;y;z];
    
    if nargout==4 && t_ii>0 && p_ii>0
        v = squeeze(sepPlane{p}(t_ii,p_ii,:));
        if size(S,2)>3 && ~all(v(1:3)==0) && all(~isnan(v))
            e2 = squeeze(E2{p}(t_ii,p_ii,:));
            [points, tr] = getPointsOnPlane(S, v, e2);
        end
    end
end

function [sprd, ind, T, prjS] = getProjectedSpread(p, t_ii, p_ii, S)
    
    prjS = nan;
    T = nan;
    sprd = false(Nv, Nu);
    uEdge = -1:gap:1;
    vEdge = -1:gap:1;
    ind = [];
    
    if t_ii>0 && p_ii>0
        v = squeeze(sepPlane{p}(t_ii,p_ii,:));
        
%         if size(S,2)>1 && ~all(v(1:3)==0) && all(~isnan(v))
        if ~all(v(1:3)==0) && ~any(isnan(v))
            v = v/norm(v(1:3));
            e1 = v(1:3);
            e2 = squeeze(E2{p}(t_ii,p_ii,:));
            T = gramschmidt([e1,e2,cross(e1, e2)]);
            
            bnd = squeeze(bounds{p}(t_ii, p_ii,:));
            v_i = floor((bnd(1:2)+1)/gap + 1);
            u_i = floor((bnd(3:4)+1)/gap + 1);            
            sprd(u_i(1):u_i(2), v_i(1):v_i(2)) = true;

            prjS = T(:,2:3)'*S;     % T(:,1)=b1;            
            ud = floor((prjS(1,:)+1)/gap) + 1;
            vd = floor((prjS(2,:)+1)/gap) + 1;
            ind = (ud-1)*Nv + vd;
            sprd(ind) = false;
        end
    end
end

function [t_i, p_i] = uv2polar(p, u_i, v_i, t_i, p_i)
    
    uEdge = -1:gap:1;
    vEdge = -1:gap:1;
    
    v = squeeze(sepPlane{p}(t_i,p_i,:));
    v = v/norm(v(1:3));
    if ~any(isnan(v))
        e1 = v(1:3);
        e2 = squeeze(E2{p}(t_i,p_i,:));
        Ti = gramschmidt([e1,e2,cross(e1, e2)]);
        
        ui = uEdge(u_i);
        vi = vEdge(v_i);
        
%         Xh = Ti(:,2:3)*[ui; vi];
        Xh = Ti*[-v(4); ui; vi];
        if norm(Xh)>1
            Xh = Xh/norm(Xh);
        end
        X_i = pointFromPlane2Sphere(Xh, v, 1);
        
        if ([X_i', 1]*v)>0.001
            disp('Seperating Plane constraint violated');
        end
    end
    
    [th, phi, ~] = cart2sph(X_i(1), X_i(2), X_i(3));
    th = rad2deg(th);
    phi = rad2deg(phi);
    
    t_i = floor((th+180)/jmp + 1);
    p_i = floor((phi+90)/jmp + 1);
end
  
function [Sl] = getLimitPose(p, t_i, p_i, t_ii, p_ii)    
    anglsi = angles;    
    if t_ii>0 && p_ii>0
        anglsi(:, prnts2(p)) = [thEdge(t_ii); phEdge(p_ii)];
    end
    anglsi(:, p) = [thEdge(t_i); phEdge(p_i)];      % This doesn't matter for X
    
    [dSi] = jointAngles2Cart3(dS, anglsi, typ, p);
    Sl = estimateZ(dSi, edges', [0;0;0]);
end

function [Sl] = getLimitPoses(p, t_i, p_i, t_ii, p_ii)
        
    Sl = zeros(12, P);    
    anglsi = angles;

    if prnts2(p)
        th = thEdge(t_i);
        phi = phEdge(p_i);
        anglsi(:, prnts2(p)) = [th; phi];
        if prnts2(prnts2(p))
            anglsi(:, prnts2(prnts2(p))) = [thEdge(t_ii); phEdge(p_ii)];
        end        
    end

    if prnts2(p)
        v = squeeze(sepPlane{p}(t_i,p_i,:));
        v = v/norm(v(1:3));
        if ~any(isnan(v))
            e1 = v(1:3);
            e2 = squeeze(E2{p}(t_i,p_i,:));
            Ti = gramschmidt([e1,e2,cross(e1, e2)]);
            bnd = bounds{p}(t_i, p_i,:);
            crnr = bnd(sel);
%             crnr = [bnd(1) bnd(1) bnd(2) bnd(2); bnd(3) bnd(4) bnd(3) bnd(4)];
        end
    end
    for i=1:4
        if prnts2(p)==0
            
        elseif exist('crnr', 'var')
            Xh = Ti(:,2:3)*crnr(:,i);
            if norm(Xh)>1
                Xh = Xh/norm(Xh);
            end
            X_i = pointFromPlane2Sphere(Xh, v, 1);

            [thL, phiL, ~] = cart2sph(X_i(1), X_i(2), X_i(3));
            anglsi(:, p) = rad2deg([thL; phiL]);
        end
        
        dSi = jointAngles2Cart3(dS, anglsi, typ);
        Sl(3*i-2:3*i, :) = estimateZ(dSi, edges', [0;0;0]);
    end
end

function [myPrnt, myGPrnt, t_prnt, p_prnt, t_Gprnt, p_Gprnt] = getPrntsNAnglIndcs(p, angls)
    myPrnt = prnts2(p);
    if myPrnt
%         t_prnt = floor((angls(1,myPrnt)+180)/jmp + 1);
%         p_prnt = floor((angls(2,myPrnt)+90)/jmp + 1) - 5; 
        t_prnt = 27;
        p_prnt = 32;
%         p_prnt = floor((angls(2,myPrnt)+90)/jmp + 1);       
        

        myGPrnt = prnts2(myPrnt);
        if myGPrnt
            t_Gprnt = floor((angls(1,myGPrnt)+180)/jmp + 1);
            p_Gprnt = floor((angls(2,myGPrnt)+90)/jmp + 1);
        else
            t_Gprnt = 0;
            p_Gprnt = 0;
        end
    else
        myGPrnt = nan;
        t_prnt = 0;
        p_prnt = 0;
        t_Gprnt = 0;
        p_Gprnt = 0;
    end
end  
%% Initialize
angles = cart2JointAngles3(dS, typ);
sp = 2;
t_c = ceil(length(thEdge)/2)+1;
p_c = ceil(length(phEdge)/2)+1;
t_c = 27;
p_c = 32;
        
gap = 1/100;
Nv = floor(2/gap) + 1;
Nu = Nv;
    
u_c = 50;
v_c = 50;
% T = eye(3);
% ds = 0;
% prjS = 1;

N = 50;
[Sx,Sy,Sz] = sphere(N);
dt = delaunayTriangulation(Sx(:),Sy(:),Sz(:)); 
[tri, spX]= freeBoundary(dt); 
spX = spX';
[Sx,Sy,Sz] = sphere(15);

[myPrnt, myGPrnt, t_p, p_p, t_gp, p_gp] = getPrntsNAnglIndcs(sp, angles);
% t = timer('TimerFcn', @(~,~)myTimerFnc, 'Period', 0.03, 'ExecutionMode', 'fixedRate');
initlialize(sp);
end
