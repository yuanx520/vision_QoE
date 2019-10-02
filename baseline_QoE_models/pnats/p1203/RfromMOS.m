function Q = RfromMOS(MOS)
    MOS(MOS > 4.5) = 4.5;
    Q = zeros(length(MOS), 1);
    h = zeros(length(MOS), 1);
    
    for iii = 1:length(Q)
        if MOS(iii) > 2.7505
            h(iii) = (1/3) .* (pi - atan(15 .* sqrt(-903522 + ...
                1113960 .* MOS(iii) - 202500 .* MOS(iii)^2) / (6750 .* MOS(iii) - 18566))); 
        else
            h(iii) = (1/3) .* (atan(15 .* sqrt(-903522 + ...
                1113960 .* MOS(iii) - 202500 .* MOS(iii)^2) / (6750 .* MOS(iii) - 18566)));
        end
    end
    Q = 20 .* (8 - sqrt(226) .* cos(h + (pi/3))) / 3;
end