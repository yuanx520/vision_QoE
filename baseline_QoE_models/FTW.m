function Q = FTW( l, n )
    % ftw QoE model
    % l: length of average stalling events
    % n: number of stalling events
    Q = (3.5 * exp(-(0.15*l+0.19)*n) + 1.5) * 20;
end