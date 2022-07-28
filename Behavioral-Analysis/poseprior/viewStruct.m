function viewStruct(S, Shat, edges)

    F = size(S,1)/3;
    x1 = min(min(S(1:3:end, :)));
    y1 = min(min(S(2:3:end, :)));
    z1 = min(min(S(3:3:end, :)));
    x2 = max(max(S(1:3:end, :)));
    y2 = max(max(S(2:3:end, :)));
    z2 = max(max(S(3:3:end, :)));
    if nargin<3
        load skeleton_17Pts;
    end
%     load vSkeleton    
    
    for i=1:F    
        plot3(S(3*i-2, :), S(3*i, :), S(3*i-1, :), 'k.', 'markersize', 15); 
        hold on;
        drawLines(S(3*i-2:3*i, :), 'k-',edges);
        if exist('Shat')
            plot3(Shat(3*i-2, :), Shat(3*i, :), Shat(3*i-1, :), 'b.'); 
            drawLines(Shat(3*i-2:3*i, :), 'b-',edges);
        end
        hold off;
        title(sprintf('Frame: %03d',i))
        view(180, 24);  
        grid on    
        axis equal off
%         axis([x1 x2 z1 z2 y1 y2]);
        
%         pause;
    end    
end

function drawLines(S,lineStyle,edges)
    
    for b = 1:size(edges,1)
        i = edges(b,1); j = edges(b,2);
        plot3([S(1,i), S(1,j)], [S(3,i), S(3, j)], [S(2,i), S(2, j)], lineStyle, 'linewidth', 1.5);            
    end        
end