function generate_hists(E)
    for i=1:size(E,1)
        figure;
        %histogram(E{i},[[0,2],[2,5],[5,15], [15,30], [30,50] ])
        histogram(E{i}, 8)
        [min(E{i}), max(E{i}), mean(E{i}), median(E{i})]
    end
end
