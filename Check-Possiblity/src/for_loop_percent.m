function for_loop_percent(i, num_loops)
    arguments
        i(1,1) {mustBeInteger, mustBePositive};
        num_loops(1,1) {mustBeInteger, mustBePositive};
    end
    if i == 1
        fprintf('0 %%');
    else
        perc = round((i/num_loops*100));
        last_perc = round(((i-1)/num_loops*100));
        if (perc >= 100) && (last_perc >= 100)
            fprintf('\b\b\b\b\b%d %%', perc);
        elseif perc >= 10 && last_perc >= 10
            fprintf('\b\b\b\b%d %%', perc);
        else
            fprintf('\b\b\b%d %%', perc);
        end
    end
end